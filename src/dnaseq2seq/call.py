
import os
import time

import datetime
import logging
import string
import random
import traceback
from collections import defaultdict

from functools import partial
from pathlib import Path
from typing import List, Callable
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import torch
import torch.multiprocessing as mp
import queue
import pysam
import numpy as np

from dnaseq2seq.model import VarTransformer
from dnaseq2seq import buildclf
from dnaseq2seq import vcf
from dnaseq2seq import util
from dnaseq2seq import bam


logger = logging.getLogger(__name__)


DEVICE = torch.device("cuda") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")

import warnings
warnings.filterwarnings(action='ignore')

REGION_ERROR_TOKEN = "error"

# Profiling
TOTAL_TIME_CLF = 0

class RegionStopSignal:

    def __init__(self, total_sus_regions, total_sus_bp):
        self.tot_sus_regions = total_sus_regions
        self.tot_sus_bp = total_sus_bp

class CallingStopSignal:

    def __init__(self, regions_submitted, sus_region_count, sus_region_bp):
        self.regions_submitted = regions_submitted
        self.sus_region_count = sus_region_count
        self.sus_region_bp = sus_region_bp


class IndexedRegion:

    def __init__(self, chrom, start, end, index):
        self.chrom = chrom
        self.start = start
        self.end = end
        self.index = index


def randchars(n=6):
    """ Generate a random string of letters and numbers """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))


def gen_suspicious_spots(bamfile, chrom, start, stop, reference_fasta):
    """
    Generator for positions of a BAM / CRAM file that may contain a variant. This should be pretty sensitive and
    trigger on anything even remotely like a variant
    This uses the pysam Pileup 'engine', which seems less than ideal but at least it's C code and is probably
    fast. May need to update to something smarter if this
    :param bamfile: The alignment file in BAM format.
    :param chrom: Chromosome containing region
    :param start: Start position of region
    :param stop: End position of region (exclusive)
    :param reference_fasta: Reference sequences in fasta
    """
    aln = pysam.AlignmentFile(bamfile, reference_filename=reference_fasta)
    ref = pysam.FastaFile(reference_fasta)
    refseq = ref.fetch(chrom, start, stop)
    assert len(refseq) == stop - start, f"Ref sequence length doesn't match start - stop coords start: {chrom}:{start}-{stop}, ref len: {len(refseq)}"
    for col in aln.pileup(chrom, start=start, stop=stop, stepper='nofilter', multiple_iterators=False):
        # The pileup returned by pysam actually starts long before the first start position, but we only want to
        # report positions in the actual requested window
        if start <= col.reference_pos < stop:
            refbase = refseq[col.reference_pos - start]
            indel_count = 0
            base_mismatches = 0

            for i, read in enumerate(col.pileups):
                if read.indel != 0:
                    indel_count += 1

                if read.query_position is not None:
                    base = read.alignment.query_sequence[read.query_position]
                    if base != refbase:  # May want to check quality before adding a mismatch?
                        base_mismatches += 1

                if indel_count > 1 or base_mismatches > 2:
                    yield col.reference_pos
                    break



def load_model(model_path):
    """
    Create the VariantTransformer model using params / config settings from the given path
    Model is compiled with torch.compile and set to eval mode
    :returns: VariantTransformer model with parameters loaded
    """
    model_info = torch.load(model_path, map_location=DEVICE)
    statedict = model_info['model']
    modelconf = model_info['conf']
    new_state_dict = {}
    for key in statedict.keys():
      new_key = key.replace('_orig_mod.', '')
      new_state_dict[new_key] = statedict[key]
    statedict = new_state_dict

    model = VarTransformer(read_depth=modelconf['max_read_depth'],
                           feature_count=modelconf['feats_per_read'],
                           kmer_dim=util.FEATURE_DIM,  # Number of possible kmers
                           n_encoder_layers=modelconf['encoder_layers'],
                           n_decoder_layers=modelconf['decoder_layers'],
                           embed_dim_factor=modelconf['embed_dim_factor'],
                           encoder_attention_heads=modelconf['encoder_attention_heads'],
                           decoder_attention_heads=modelconf['decoder_attention_heads'],
                           d_ff=modelconf['dim_feedforward'],
                           device=DEVICE)

    model.load_state_dict(statedict)
    model.eval()
    model.to(DEVICE)
    
    model = torch.compile(model, fullgraph=True)
    return model


def cluster_positions_for_window(window, bamfile, reference_fasta, maxdist=100):
    """
    Generate a list of ranges containing a list of positions from the given window
    returns: list of (chrom, index, start, end) tuples
    """
    chrom, window_idx, window_start, window_end = window

    cpname = mp.current_process().name
    logger.debug(
        f"{cpname}: Generating regions from window {window_idx}: "
        f"{window_start}-{window_end} on chromosome {chrom}"
    )
    return [
        (chrom, window_idx, start, end)
        for start, end in util.cluster_positions(
            gen_suspicious_spots(bamfile, chrom, window_start, window_end, reference_fasta),
            maxdist=maxdist,
        )
    ]


def call(model_path: str, bam: str, bed: str, reference_fasta: str, vcf_out: str, classifier_path=None, **kwargs):
    """
    Use model in statedict to call variants in bam in genomic regions in bed file.
    Steps:
      1. build model
      2. break bed regions into windows with start positions determined by window_spacing and end positions
         determined by window_overlap (the last window in each bed region will likely be shorter than others)
      3. call variants in each window
      4. join variants after searching for any duplicates
      5. save to vcf file
    :param model_path: Path to haplotype generation model
    :param bam: Path to input BAM / CRAM file
    :param bed: Path to input BED file
    :param reference_fasta: Path to reference genome fasta
    :param vcf_out: Path to destination VCF
    :param classifier_path: Path to classifier model
    """
    seed = 1283769
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.set_num_threads(4) # Per-process ?
    mp.set_start_method('spawn')
    start_time = time.perf_counter()
    threads = kwargs.get('threads', 1)
    max_batch_size = kwargs.get('max_batch_size', 64)
    logger.info(f"Using {threads} threads for encoding")
    logger.info(f"Found torch device: {DEVICE}")
    logger.info(f"Writing variants to {Path(vcf_out).absolute()}")
    
    if 'cuda' in str(DEVICE):
        for idev in range(torch.cuda.device_count()):
            logger.info(f"Using CUDA device {idev} {torch.cuda.get_device_name({idev})}")

    logger.info(f"The model will be loaded from path {model_path}")

    vcf_header_extras = kwargs.get('cmdline')

    assert Path(model_path).is_file(), f"Model file {model_path} isn't a regular file"
    assert Path(bam).is_file(), f"Alignment file {bam} isn't a regular file"
    assert Path(bed).is_file(), f"BED file {bed} isn't a regular file"
    assert Path(reference_fasta).is_file(), f"Reference genome {reference_fasta} isn't a regular file"
    if classifier_path is None:
        logger.info("No classifier model provided, emitting uncalibrated qualities only. Specificity will be poor")
    else:
        assert Path(classifier_path).is_file(), f"Classifier model {classifier_path} isn't a regular file"

    # Verify model loading, we just want to fail fast here if there's an issue
    load_model(model_path)

    call_vars_in_parallel(
        bampath=bam,
        bed=bed,
        refpath=reference_fasta,
        model_path=model_path,
        classifier_path=classifier_path,
        threads=threads,
        max_batch_size=max_batch_size,
        vcf_out=vcf_out,
        vcf_header_extras=vcf_header_extras,
        show_progress=not kwargs.get('no_progress', False),
    )

    logger.info(f"All variants saved to {vcf_out}")
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    if elapsed_seconds > 3600:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds / 3600 :.2f} hours")
    elif elapsed_seconds > 120:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds / 60 :.2f} minutes")
    else:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds :.2f} seconds")


def region_priority(region_data: dict, chrom_order: List[str]) -> int:
    """
    Returns the priority value for the given region tensor
    Smaller-valued items are grabbed first by a PriorityQueue
    """
    chrom, start, end = region_data['region']
    idx = chrom_order.index(chrom)
    return int(1e9 * idx + start)


def worker_wrapper(func, callstate, *args, **kwargs):
    """
    Wrap a function with a try-catch block that sets an exception in the callstate dict
    """
    try:
        return func(*args, **kwargs)
    except Exception as ex:
        funcname = getattr(func, '__name__', 'unknown')
        logger.error(f"Caught exception in wrapped function {funcname}: {ex}")
        callstate['exception'] = ex
        callstate['traceback'] = traceback.format_exc()
        raise ex


def call_vars_in_parallel(
    bampath, bed, refpath, model_path, classifier_path, threads, max_batch_size, vcf_out, vcf_header_extras, show_progress,
):
    """
    Call variants in asynchronous fashion. There are three types of Processes that communicate via two mp.Queues
    The first process finds 'suspect' regions in the BAM file and adds them to the 'regions_queue', this is fast and
    there's just one Process that handles this
    The second type of process reads the regions_queue and generates region Tensors (data from BAM/CRAM files encoded
    into Tensors), and adds them to the 'tensors_queue'. There are 'threads' number of these Processes
    The final process reads from the tensors_queue and runs the forward pass of the model to generate haplotypes,
    then aligns those haplotypes to call variants. This is slow, but not sure we can parallelize it since there's
    (probably) only one GPU anyway?

    A total footgun here is that pytorch releases tensors generates by a Process when that process dies, even if they've been added to a shared queue (!!). So the 'generate_tensors'
    Processes must stay alive until the variant calling Process has completed. The calling process therefore waits until it receives 'threads' number of completion signals in the tensors_queue,
    then finishes processing everything, then adds signals (None objects) into the 'keepalive' queue (upon which the generate_tensors processes are waiting) to tell them they can finally die

    """

    regions_queue = mp.Queue(maxsize=1024)  # Hold BED file regions, generated in main process and sent to 'generate_tensors' workers
    tensors_queue =  mp.Queue(maxsize=5000)  # Holds tensors generated in 'generate tensors' workers, consumed by accumulate_regions_and_call
    region_keepalive_queue = mp.Queue()  # Signals to region_workers that they are permitted to die, since all tensors have been processed

    bed_chrom_order = util.unique_chroms(bed)
    priority_func = partial(region_priority, chrom_order=bed_chrom_order)
    progress_tracker = util.RegionProgressCounter(bed)

    with mp.Manager() as manager:
        callstate = manager.dict()
        callstate['exception'] = None
        callstate['region_keepalive'] = True

        # This one processes the input BED file and find 'suspect regions', and puts them in the regions_queue
        region_finder = mp.Process(target=worker_wrapper,
                                   args=(find_regions, callstate, regions_queue, bed, bampath, refpath, threads, show_progress, callstate))
        region_finder.start()

        region_workers = [mp.Process(target=worker_wrapper,
                                     args=(generate_tensors, callstate, regions_queue, tensors_queue, bampath, refpath, callstate))
                          for _ in range(threads)]

        for p in region_workers:
            p.start()

        model_proc = mp.Process(target=worker_wrapper,
                                args=(accumulate_regions_and_call, callstate, model_path, tensors_queue, priority_func, refpath, bampath, classifier_path, max_batch_size, vcf_out, vcf_header_extras, threads, callstate, progress_tracker, show_progress))
        model_proc.start()

        region_finder.join()
        logger.debug("Done finding regions")

        for p in region_workers:
            p.join()
        logger.debug("Region workers are done")

        model_proc.join()

        if callstate['exception'] is not None:
            logger.error(f"Found an exception in the calling process: {callstate['exception']}")
            raise callstate['exception']


def find_regions(regionq, inputbed, bampath, refpath, n_signals, show_progress, callstate):
    """
    Read the input BED formatted file and merge / split the regions into big chunks
    Then find regions that may contain a variant, and add all of these
    to the region_queue
    """
    torch.set_num_threads(2) # Must be here for it to work for this process
    region_count = 0
    tot_size_bp = 0
    sus_region_bp = 0
    sus_region_count = 0

    tot_regions, tot_bases = util.count_bed(inputbed)
    if show_progress:
        progbar = tqdm(total=100, position=0, desc="Region finding")
    else:
        progbar = None
    logger.info(f"Found {tot_regions} regions with {util.format_bp(tot_bases)} in {inputbed}")
    
    for idx, (chrom, window_start, window_end) in enumerate(util.split_large_regions(util.read_bed_regions(inputbed), max_region_size=10000)):
        if callstate['exception'] is not None:
            logger.error(f"Found an exception in the calling process: {callstate['exception']}")
            break

        try:
            region_count += 1
            tot_size_bp += window_end - window_start
            sus_regions = cluster_positions_for_window(
                (chrom, idx, window_start, window_end),
                bamfile=bampath,
                reference_fasta=refpath,
                maxdist=100,
            )
            sus_regions = util.merge_overlapping_regions(sus_regions)
            if progbar is not None:
                progbar.update(round(100 * (tot_size_bp) / tot_bases, 2) - progbar.n)
                progbar.refresh()
            else:
                logger.info(f"Identified regions {tot_size_bp} of {tot_bases} bp ({tot_size_bp / tot_bases * 100 :.2f} done)")
            for i, r in enumerate(sus_regions):
                sus_region_count += 1
                sus_region_bp += r[-1] - r[-2]
                regionq.put(IndexedRegion(chrom, r[-2], r[-1], idx))
        except Exception as ex:
            logger.error(f"Exception in region finder: {ex}")
            callstate['exception'] = ex
            raise ex

    logger.debug("Done finding regions")
    for i in range(n_signals):
        regionq.put(RegionStopSignal(sus_region_count, sus_region_bp))
    if progbar:
        progbar.close()


def encode_region(bampath, refpath, idxregion, max_read_depth, window_size, min_reads, batch_size, window_step):
    """
    Encode the reads in the given region and save the data along with the region and start offsets to a file
    and return the absolute path of the file
    """

    logger.debug(f"Encoding region {idxregion.chrom}:{idxregion.start}-{idxregion.end}")
    aln = pysam.AlignmentFile(bampath, reference_filename=refpath)
    reference = pysam.FastaFile(refpath)
    all_encoded = []
    all_starts = []
    for encoded_region, start_positions in _encode_region(aln, reference, idxregion.chrom, idxregion.start, idxregion.end, max_read_depth,
                                                     window_size=window_size, min_reads=min_reads, batch_size=batch_size, window_step=window_step):
        all_encoded.append(encoded_region)
        all_starts.extend(start_positions)
    logger.debug(f"Done encoding region {idxregion.chrom}:{idxregion.start}-{idxregion.end}, created {len(all_starts)} windows")
    if len(all_encoded) > 1:
        encoded = torch.concat(all_encoded, dim=0)
    elif len(all_encoded) == 1:
        encoded = all_encoded[0]
    else:
        logger.error(f"Uh oh, did not find any encoded paths!, region is {idxregion.chrom}:{idxregion.start}-{idxregion.end}")
        return None

    data = {
        'encoded_pileup': encoded,
        'region': (idxregion.chrom, idxregion.start, idxregion.end),
        'start_positions': all_starts,
        'index': idxregion.index,
    }
    return data


def generate_tensors(region_queue: mp.Queue, output_queue: mp.Queue, bampath, refpath, callstate: dict, max_read_depth=150, window_size=150):
    """
    Consume regions from the region_queue and generate input tensors for each and put them into the output_queue
    """
    min_reads = 5 # Abort if there are fewer than this many reads
    batch_size = 64 # Tensors hold this many regions at max, but since we're encoding a single region most tensors will have 4-8 individual windows
    window_step = 25
    torch.set_num_threads(2)  # Must be here for it to work for this process

    encoded_region_count = 0
    while True:
        if callstate['exception'] is not None:
            logger.error(f"Found an exception in the calling process: {callstate['exception']}")
            break

        region = region_queue.get()
        if isinstance(region, RegionStopSignal):
            logger.debug("Region worker found end token")
            output_queue.put(CallingStopSignal(regions_submitted=encoded_region_count, sus_region_count=region.tot_sus_regions, sus_region_bp=region.tot_sus_bp))
            break
        else:

            reg = (region.chrom, region.start, region.end)
            data = encode_region(bampath, refpath, region, max_read_depth, window_size, min_reads, batch_size=batch_size, window_step=window_step)
            if data is not None:
                data['encoded_pileup'].share_memory_()
                encoded_region_count += 1
                output_queue.put(data)

    
    # It is CRITICAL to keep these processes alive, even after they're done doing everything. Pytorch will clean up the 
    # the tensors *that have already been queued* when these processes die, even if the tensors haven't been processed yet
    # This will lead to errors when the calling process polls the queue, leading to missed variant calls. Instead, we wait for
    # the calling process to signal callstate dict that it's done by setting region_keepalive to True, then we can die
    logger.debug(f"Polling callstate dict after generating {encoded_region_count} tensors")
    while callstate['region_keepalive'] and callstate['exception'] is None:
        time.sleep(1)
    logger.debug(f"Region worker {os.getpid()} is shutting down after generating {encoded_region_count} encoded regions")


def merge_datas(datas):
    """
    Concat a list of datas dictionaries into a single tensor and lists of regions, start positions
    """
    batch_encoded = []
    batch_start_pos = []
    batch_regions = []

    for data in datas:
        # Load the data, parsing location + encoded data from file
        chrom, start, end = data['region']
        batch_encoded.append(data['encoded_pileup'])
        batch_start_pos.extend(data['start_positions'])
        batch_regions.extend((chrom, start, end) for _ in range(len(data['start_positions'])))
    
    if len(batch_encoded) > 1:
        allencoded = torch.concat(batch_encoded, dim=0)
    else:
        allencoded = batch_encoded[0]

    return allencoded, batch_start_pos, batch_regions


@torch.no_grad()
def call_multi_paths(datas, model, refpath, bampath, classifier_model, vcf_template, max_batch_size):
    """
    Concat a list of 'datas' objects, which contain encoded_pileups, batch offsets, and then call variants over all of them
    
    :return : List of VCFRecords for variants found
    """
    call_start = datetime.datetime.now()
    var_records = []  # Stores all variant records
    reference = pysam.FastaFile(refpath)
    
    allencoded, batch_start_pos, batch_regions = merge_datas(datas)

    hap0, hap1= call_and_merge(allencoded, batch_start_pos, batch_regions, model, reference, max_batch_size)

    var_records.extend(vars_hap_to_records(hap0, hap1, bampath, refpath, classifier_model, vcf_template))

    call_elapsed = datetime.datetime.now() - call_start
    logger.debug(
        f"Called variants over {len(batch_start_pos)} windows from {len(datas)} regions in {call_elapsed.total_seconds() :.2f} seconds"
    )
    return var_records


def accumulate_regions_and_call(modelpath: str,
                                inputq: mp.Queue,
                                priority_func: Callable,
                                refpath: str,
                                bampath: str,
                                classifier_path,
                                max_batch_size: int,
                                vcf_out_path: str,
                                header_extras: str,
                                n_region_workers: int,
                                callstate: dict,
                                progress_tracker: util.RegionProgressCounter,
                                show_progress: bool):
    """
    Continually poll the input queue to find new encoded regions, and call variants over those regions
    This function is typically called inside a subprocess and runs until it finds a None entry in the queue
    Variants are written to the output VCF file here
    """

    torch.set_num_threads(8)
    model = load_model(modelpath)
    model.eval()
    if classifier_path:
        classifier = buildclf.load_model(classifier_path)
    else:
        classifier = None

    vcf_header = vcf.create_vcf_header(sample_name="sample", lowcov=20, cmdline=header_extras)
    vcf_template = pysam.VariantFile("/dev/null", mode='w', header=vcf_header)

    vcf_out = open(vcf_out_path, "w")
    vcf_out.write(str(vcf_header))
    vcf_out.flush()

    datas = []
    bp_processed = 0

    # Not sure what the optimum is here - we accumulate tensors until we have at least this many, then process them in batches
    # If this number is too large, we will wait for too long before submitting the next batch to the GPU
    # If its to small, then we end up sending lots of tiny little batches to the GPU, which also isn't efficient
    n_finished_workers = 0
    max_consecutive_timeouts = 10
    timeouts = 0
    regions_found = 0
    regions_processed = 0
    tot_regions_submitted = 0
    vbuff = util.VariantSortedBuffer(outputfh=vcf_out, buff_size=10000)
    if show_progress:
        progbar = tqdm(total=100, position=1, desc="Variant calling")
    else:
        progbar = None

    wait_time_total = 0
    process_time_total = 0
    total_windows_found = 0
    while True:
        if callstate['exception'] is not None:
            logger.error(f"Found an exception in the calling process: {callstate['exception']}")
            break

        try:
            t0 = time.perf_counter()
            data = inputq.get(timeout=10) # Timeout is in seconds, we count these and error out if there are too many
            t1 = time.perf_counter()
            wait_time_total += t1 - t0
            timeouts = 0
        except queue.Empty:
            timeouts += 1
            data = None
            logger.debug(f"Got a timeout in model queue, have {timeouts} total")
        except Exception as ex:
            logger.error(f"Exception polling calling input queue: {ex}")
            raise ex

        if timeouts == max_consecutive_timeouts:
            logger.error(f"Found {max_consecutive_timeouts} timeouts, aborting model processing queue")
            break

        if not isinstance(data, CallingStopSignal) and data is not None:
            regions_found += 1
            logger.debug("Found a non-None data object, appending it")
            datas.append(data)
            total_windows_found += data['encoded_pileup'].shape[0]

        if type(data) == CallingStopSignal:
            n_finished_workers += 1
            tot_regions_submitted += data.regions_submitted
            logger.debug(f"Found a stop token, {n_finished_workers} of {n_region_workers} are done, tot regions submitted: {tot_regions_submitted}, tot processed: {regions_processed}")

        if total_windows_found >= max_batch_size:
            logger.debug(f"Calling variants from {len(datas)} objects, we've found {regions_found} regions and processed {regions_processed} of them so far")
            datas = sorted(datas, key=priority_func) # Sorting data chunks here helps ensure sorted output

            bp = sum(d['region'][2] - d['region'][1] for d in datas)
            bp_processed += bp
            logger.debug(
                f"Calling variants up to {datas[len(datas) // 2]['region'][0]}:{datas[-1]['region'][1]}-{datas[-1]['region'][2]}, total bp processed: {round(bp_processed / 1e6, 3)}MB"
            )
            t0 = time.perf_counter()
            to_submit, datas = datas[0:-3], datas[-3:]
            submitted_windows = sum(d['encoded_pileup'].shape[0] for d in to_submit)
            remaining_windows = sum(d['encoded_pileup'].shape[0] for d in datas)
    
            total_windows_found = remaining_windows
            records = call_multi_paths(to_submit, model, refpath, bampath, classifier, vcf_template, max_batch_size=max_batch_size)
            t1 = time.perf_counter()
            process_time_total += t1 - t0
            progress = 100 * progress_tracker.prog(datas[-1]['region'][0], datas[-1]['region'][2])

            if progbar is not None:
                progbar.update(round(progress, 2) - progbar.n)
                progbar.refresh()
            else:
                logger.info(f"Variant calling progress {progress:.2f}%")

            regions_processed += len(datas)
            # Store the variants in a buffer so we can sort big groups of them (no guarantees about sort order for
            # variants coming out of queue)
            vbuff.put_all(records)

        if n_finished_workers == n_region_workers:
            # We are finishing, so process anything left in the datas buffer and put the records in the variant buffer
            if datas:
                records = call_multi_paths(datas, model, refpath, bampath, classifier, vcf_template, max_batch_size=max_batch_size)
                vbuff.put_all(records)

            logger.debug(f"All region workers are done, datas length is {len(datas)}, exiting..")
            break

    logger.debug(f"Calling process is cleaning up, got {tot_regions_submitted} regions submitted, found {regions_found} regions and processed {regions_processed} regions")
    callstate['region_keepalive'] = False

    logger.debug(f"Writing final {len(vbuff)} variants...")
    if progbar:
        progbar.close()
    
    logger.info(f"Calling queue spent {wait_time_total :.3f} seconds waiting for info and {process_time_total :.3f} seconds processing data")
    logger.info(f"Spent {TOTAL_TIME_CLF :.3f} seconds in classifier")
    vbuff.flush()
    vcf_out.close()

    logger.debug("Calling worker is exiting")


def call_and_merge(batch, batch_offsets, regions, model, reference, max_batch_size):
    """
    Generate haplotypes for the batch, identify variants in each, and then 'merge genotypes' across the overlapping
    windows with the ad-hoc algo in the resolve_haplotypes function. This also filters out any variants not found
    in the 'regions' tuple

    Note that this function contains additional, unused logic to divide batch up into smaller regions and send those
    through the model individually. This might be useful if different regions require very different numbers of predicted
    tokens, for instance. However, since the time spent in forward passes of the model are almost constant in batch size
    this doesn't offer much speedup in a naive implementation (but since we don't do KV caching during decoding the cost
    for generating additional tokens increases linearly in the total token count... so maybe it makes sense to do this?)

    :param batch: Tensor of encoded regions
    :param batch_offsets: List of start positions, must have same length as batch.shape[0]
    :param regions: List of genomic regions, must have length equal to batch_offsets
    :param model: Model for haplotype prediction
    :param reference: pysam.FastaFile with reference genome
    :param max_batch_size: Maximum number of regions to call in one go
    :returns Dict[(chrom, start, end)] -> List[proto vars] for the variants found in each region
    """
    logger.debug(f"Predicting batch of size {batch.shape[0]} for chrom {regions[0][0]}:{regions[0][1]}-{regions[-1][2]}")
    dists = np.array([r[2] - bo for r, bo in zip(regions, batch_offsets)])
    #mid_dist = int(np.mean(dists))
    #logger.info(f"Dists: {dists} mid dist: {mid_dist}")
    # Identify distinct sub-batches for calling. In the future we might populate this with different indexes
    # Setting everything to 0 effectively turns it off for now
    subbatch_idx = np.zeros(len(dists), dtype=int)
    #subbatch_idx = (dists > mid_dist).astype(np.int_)
    #logger.info('\n'.join(f"{b, d}" for b,d in zip(subbatch_idx, dists)))
    byregion = defaultdict(list)
    # n_output_toks = min(150 // util.TGT_KMER_SIZE - 1, max(dists) // util.TGT_KMER_SIZE + 1)
    #logger.info(f"Dists: {dists}")
    #median_dist = np.median(dists)

    for sbi in range(max(subbatch_idx) + 1):
        which = np.where(subbatch_idx == sbi)[0]
        subbatch = batch[torch.tensor(which), :, :, :]
        subbatch_offsets = [b for i,b in zip(subbatch_idx, batch_offsets) if i == sbi]
        subbatch_dists =   [d for i,d in zip(subbatch_idx, dists) if i == sbi]
        subbatch_regions = [r for i,r in zip(subbatch_idx, regions) if i == sbi]
        max_dist = max(subbatch_dists)
        logger.debug(f"Max dist for sub-batch {sbi} : {max_dist}")
        n_output_toks = min(150 // util.TGT_KMER_SIZE - 1, max_dist // util.TGT_KMER_SIZE + 1)

        logger.debug(f"Sub-batch size: {len(subbatch_offsets)}   max dist: {max(subbatch_dists)},  n_tokens: {n_output_toks}")
        batchvars = call_batch(subbatch, subbatch_offsets, subbatch_regions, model, reference, n_output_toks, max_batch_size=max_batch_size)
        logger.debug(f"Called {len(batchvars)} in {subbatch_regions}")
        for region, bvars in zip(subbatch_regions, batchvars):
            byregion[region].append(bvars)

    hap0 = defaultdict(list)
    hap1 = defaultdict(list)
    for region, rvars in byregion.items():
        chrom, start, end = region
        h0, h1 = resolve_haplotypes(rvars)
        for k, v in h0.items():
            if start <= v[0].pos < end:
                hap0[k].extend(v)
        for k, v in h1.items():
            if start <= v[0].pos < end:
                hap1[k].extend(v)

    return hap0, hap1


def merge_multialts(v0, v1):
    """
    Merge two VcfVar objects into a single one with two alts

    ATCT   G
    A     GCAC
      -->
    ATCT   G,GCACT

    """
    assert v0.pos == v1.pos
    #assert v0.het and v1.het
    if v0.ref == v1.ref:
        v0.alts = (v0.alts[0], v1.alts[0])
        v0.qual = (v0.qual + v1.qual) / 2  # Average quality??
        v0.samples['sample']['GT'] = (1,2)
        return v0

    else:
        shorter, longer = sorted([v0, v1], key=lambda x: len(x.ref))
        extra_ref = longer.ref[len(shorter.ref):]
        newalt = shorter.alts[0] + extra_ref
        longer.alts = (longer.alts[0], newalt)
        longer.qual = (longer.qual + shorter.qual) / 2
        longer.samples['sample']['GT'] = (1, 2)
        return longer


def het(rec):
    return rec.samples[0]['GT'] == (0,1) or rec.samples[0]['GT'] == (1,0)


def merge_overlaps(overlaps, min_qual):
    """
     Attempt to merge overlapping VCF records in a sane way
     As a special case if there is only one input record we just return that
    """
    if len(overlaps) == 1:
        return [overlaps[0]]
    overlaps = list(filter(lambda x: x.qual > min_qual, overlaps))
    if len(overlaps) == 1:
        # An important case where two variants overlap but one of them is low quality
        # Should the remaining variant be het or hom?
        return [overlaps[0]]
    elif len(overlaps) == 0:
        return []

    result = []
    overlaps = sorted(overlaps, key=lambda x: x.qual, reverse=True)[0:2]  # Two highest quality alleles
    overlaps[0].samples['sample']['GT'] = (None, 1)
    overlaps[1].samples['sample']['GT'] = (1, None)
    result.extend(sorted(overlaps, key=lambda x: x.pos))
    return result


def collect_phasegroups(vars_hap0, vars_hap1, aln, reference, minimum_safe_distance=100):
    """
    Construct VcfVar objects with phase from Variant dict objects, assuming we can only
    correctly phase Variants within 'minimum_safe_distance' bp

    :returns: List of VcfVar objects
    """
    allkeys = list(k for k in vars_hap0.keys()) + list(k for k in vars_hap1.keys())
    allkeys = sorted(set(allkeys), key=lambda x: x[1])

    all_vcf_vars = []
    group0 = defaultdict(list)
    group1 = defaultdict(list)
    prevpos = -1000
    prevchrom = None
    for k in allkeys:
        chrom, pos, ref, alt = k
        if (chrom != prevchrom) or (pos - prevpos > minimum_safe_distance):
            vcf_vars = vcf.construct_vcfvars(
                vars_hap0=group0,
                vars_hap1=group1,
                aln=aln,
                reference=reference
            )
            all_vcf_vars.extend(vcf_vars)

            group0 = defaultdict(list)
            group1 = defaultdict(list)
            if k in vars_hap0:
                group0[k].extend(vars_hap0[k])
            if k in vars_hap1:
                group1[k].extend(vars_hap1[k])
            prevpos = pos
        else:
            if k in vars_hap0:
                group0[k].extend(vars_hap0[k])
            if k in vars_hap1:
                group1[k].extend(vars_hap1[k])
        prevchrom = chrom

    vcf_vars = vcf.construct_vcfvars(
        vars_hap0=group0,
        vars_hap1=group1,
        aln=aln,
        reference=reference
    )
    all_vcf_vars.extend(vcf_vars)
    return all_vcf_vars


def vars_hap_to_records(vars_hap0, vars_hap1, bampath, refpath, classifier_model, vcf_template):
    """
    Convert variant haplotype objects to variant records
    """

    # Merging vars can sometimes cause a poor quality variant to clobber a very high quality one, to avoid this
    # we hard-filter out very poor quality variants that overlap other, higher-quality variants
    # This value defines the min qual to be included when merging overlapping variants
    min_merge_qual = 0.01
    global TOTAL_TIME_CLF
    reference = pysam.FastaFile(refpath)
    aln = pysam.AlignmentFile(bampath, reference_filename=refpath)

    vcf_vars = collect_phasegroups(vars_hap0, vars_hap1, aln, reference, minimum_safe_distance=100)

    # covert variants to pysam vcf records
    vcf_records = [
        vcf.create_vcf_rec(var, vcf_template)
        for var in sorted(vcf_vars, key=lambda x: x.pos)
    ]

    if not vcf_records:
        return []

    for rec in vcf_records:
        rec.info["RAW_QUAL"] = rec.qual

    if classifier_model:
        clfstart = time.time()
        
        clf_preds = buildclf.predict_records(vcf_records, classifier_model, bampath, refpath, threads=16)
        for rec, pred in zip(vcf_records, clf_preds):
            rec.qual = pred

        clfend = time.time()
        TOTAL_TIME_CLF += clfend - clfstart
        logger.debug(f"Predicted variant quality for {len(vcf_records)} records in {(clfend - clfstart):6f} seconds ({(clfend - clfstart)/len(vcf_records) :6f} per record)")

    merged = []
    overlaps = [vcf_records[0]]
    for rec in vcf_records[1:]:
        if overlaps and util.records_overlap(overlaps[-1], rec):
            overlaps.append(rec)
        elif overlaps:
            result = merge_overlaps(overlaps, min_qual=min_merge_qual)
            merged.extend(result)
            overlaps = [rec]
        else:
            overlaps = [rec]

    if overlaps:
        merged.extend(merge_overlaps(overlaps, min_qual=min_merge_qual))
    else:
        merged.append(rec)

    return merged


def _call_safe(encoded_reads, model, n_output_toks, max_batch_size, enable_amp=True):
    """
    Predict the sequence for the encoded reads, but dont submit more than 'max_batch_size' samples
    at once
    """
    seq_preds = None
    probs = None
    start = 0
    
    while start < encoded_reads.shape[0]:
        end = min(encoded_reads.shape[0]+1, start + max_batch_size)
        logger.debug(f"Calling batch of size {end - start}")
        with torch.amp.autocast(device_type='cuda', enabled=enable_amp):
            preds, prbs = util.predict_sequence(encoded_reads[start:end, :, :, :].to(DEVICE).float(), model,
                                            n_output_toks=n_output_toks, device=DEVICE)
        if seq_preds is None:
            seq_preds = preds
        else:
            seq_preds = torch.concat((seq_preds, preds), dim=0)
        if probs is None:
            probs = prbs.detach().cpu().numpy()
        else:
            probs = np.concatenate((probs, prbs.detach().cpu().numpy()), axis=0)
        start += max_batch_size
    return seq_preds, probs


def call_batch(encoded_reads, offsets, regions, model, reference, n_output_toks, max_batch_size):
    """
    Call variants in a batch (list) of regions, by running a forward pass of the model and
    then aligning the predicted sequences to the reference genome and picking out any
    mismatching parts
    :returns : List of variants called in both haplotypes for every item in the batch as a list of 2-tuples
    """
    assert encoded_reads.shape[0] == len(regions), f"Expected the same number of reads as regions, but got {encoded_reads.shape[0]} reads and {len(regions)}"
    assert len(offsets) == len(regions), f"Should be as many offsets as regions, but found {len(offsets)} and {len(regions)}"

    seq_preds, probs = _call_safe(encoded_reads, model, n_output_toks, max_batch_size)

    calledvars = []
    for offset, (chrom, start, end), b in zip(offsets, regions, range(len(seq_preds))):
        hap0_t, hap1_t = seq_preds[b, 0, :, :], seq_preds[b, 1, :, :]
        hap0 = util.kmer_preds_to_seq(hap0_t, util.i2s)
        hap1 = util.kmer_preds_to_seq(hap1_t, util.i2s)
        probs0 = np.exp(util.expand_to_bases(probs[b, 0, :]))
        probs1 = np.exp(util.expand_to_bases(probs[b, 1, :]))

        refseq = reference.fetch(chrom, offset, offset + len(hap0))
        vars_hap0 = list(v for v in vcf.aln_to_vars(refseq, hap0, chrom, offset, probs=probs0) if start <= v.pos <= end)
        vars_hap1 = list(v for v in vcf.aln_to_vars(refseq, hap1, chrom, offset, probs=probs1) if start <= v.pos <= end)
        #print(f"Offset: {offset}\twindow {start}-{end} frame: {start % 4} hap0: {vars_hap0}\n       hap1: {vars_hap1}")
        #calledvars.append((vars_hap0, vars_hap1))
        calledvars.append((vars_hap0[0:5], vars_hap1[0:5]))

    return calledvars


def add_ref_bases(encbases, reference, chrom, start, end, max_read_depth):
    """
    Add the reference sequence as read 0
    """
    refseq = reference.fetch(chrom, start, end)
    ref_encoded = bam.string_to_tensor(refseq)
    return torch.cat((ref_encoded.unsqueeze(1), encbases), dim=1)[:, 0:max_read_depth, :]


def _encode_region(aln, reference, chrom, start, end, max_read_depth, window_size=150, min_reads=5, batch_size=64, window_step=25):
    """
    Generate batches of tensors that encode read data from the given genomic region, along with position information. Each
    batch of tensors generated should be suitable for input into a forward pass of the model - but the data will be on the
    CPU.
    Each item in the batch represents a pileup in a single 'window' into the given region of size 'window_size', and
    subsequent elements are encoded from a sliding window that advances by 'window_step' after each item.
    If start=100, window_size is 50, and window_step is 10, then the items will include data from regions:
    100-150
    110-160
    120-170,
    etc

    The start positions for each item in the batch are returned in the 'batch_offsets' element, which is required
    when variant calling to determine the genomic coordinates of the called variants.

    If the full region size is small this will probably generate just a single batch, but if the region is very large
    (or batch_size is small) this could generate multiple batches

    :param window_size: Size of region in bp to generate for each item
    :returns: Generator for tuples of (batch tensor, list of start positions)
    """
    window_start = int(start - 0.7 * window_size)  # We start with regions a bit upstream of the focal / target region
    batch = []
    batch_offsets = []
    readwindow = bam.ReadWindow(aln, chrom, start - 150, end + window_size)
    logger.debug(f"Encoding region {chrom}:{start}-{end}")
    returned_count = 0
    while window_start <= (end - 0.2 * window_size):
        try:
            #logger.debug(f"Getting reads from  readwindow: {window_start} - {window_start + window_size}")
            enc_reads = readwindow.get_window(window_start, window_start + window_size, max_reads=max_read_depth)
            encoded_with_ref = add_ref_bases(enc_reads, reference, chrom, window_start, window_start + window_size,
                                             max_read_depth=max_read_depth)
            batch.append(encoded_with_ref)
            batch_offsets.append(window_start)
            #logger.debug(f"Added item to batch from window_start {window_start}")
        except bam.LowReadCountException:
            logger.debug(
                f"Bam window {chrom}:{window_start}-{window_start + window_size} "
                f"had too few reads for variant calling (< {min_reads})"
            )
        window_start += window_step
        if len(batch) >= batch_size:
            encodedreads = torch.stack(batch, dim=0).cpu()
            returned_count += 1
            yield encodedreads, batch_offsets
            batch = []
            batch_offsets = []

    # Last few
    if batch:
        encodedreads = torch.stack(batch, dim=0).cpu() # Keep encoded tensors on cpu for now
        returned_count += 1
        yield encodedreads, batch_offsets

    if not returned_count:
        logger.debug(f"Region {chrom}:{start}-{end} has only low coverage areas, not encoding data")


def resolve_haplotypes(genos):
    """
    Rearrange variants across haplotypes with a heuristic algorithm to minimize the number of conflicting
    predictions.
    Genos is a list of two-tuples of Variant objects representing the outputs of calling from multiple overlapping windows
    like this:
    [ (hap0 variants from window 1, hap1 variants from window 1),
      (hap0 variants from window 2, hap1 variants from window 2),
      ...
    ]
    The goal is to rearrange variants across haplotypes to minimize conflicts
    :returns : Two-tuple of dicts of variants, each representing one haplotype
    """
    # All unique variant keys, sorted by pos
    allvars = sorted(list(v for g in genos for v in g[0]) + list(v for g in genos for v in g[1]), key=lambda v: v.pos)
    allkeys = set()
    varsnodups = []
    for v in allvars:
        if v.key not in allkeys:
            allkeys.add(v.key)
            varsnodups.append(v)

    results = [[], []]

    prev_het = None
    prev_het_index = None

    # Loop over every unique variant and decide which haplotype to put it on
    for p in varsnodups:
        homcount = 0
        hetcount = 0
        for g in genos:
            a = p.key in [v.key for v in g[0]]
            b = p.key in [v.key for v in g[1]]
            if a and b:
                homcount += 1
            elif a or b:
                hetcount += 1
        if homcount > hetcount:
            results[0].append(p)
            results[1].append(p)
        elif prev_het is None:
            results[0].append(p) # No previous hets, so just add it to hap0
            prev_het = p
            prev_het_index = 0
        else:
            # There was a previous het variant, so figure out where this new one should go
            # determine if p should be in cis or trans with prev_het
            cis = 0
            trans = 0
            for g in genos:
                g0keys = [v.key for v in g[0]]
                g1keys = [v.key for v in g[1]]
                if (p.key in g0keys and prev_het.key in g0keys) or (p.key in g1keys and prev_het.key in g1keys):
                    cis += 1
                elif (p.key in g0keys and prev_het.key in g1keys) or (p.key in g1keys and prev_het.key in g0keys):
                    trans += 1
            if trans >= cis: # If there's a tie, assume trans. This covers the case where cis==0 and trans==0, because trans is safer
                results[1 - prev_het_index].append(p)
                prev_het = p
                prev_het_index = 1 - prev_het_index
            else:
                results[prev_het_index].append(p)
                prev_het = p
                prev_het_index = prev_het_index

    # Build dictionaries with correct haplotypes...
    allvars0 = dict()
    allvars1 = dict()
    for v in results[0]:
        allvars0[v.key] = [t for t in allvars if t.key == v.key]
    for v in results[1]:
        allvars1[v.key] = [t for t in allvars if t.key == v.key]
    return allvars0, allvars1

