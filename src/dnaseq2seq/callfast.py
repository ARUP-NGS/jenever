import os
import time

import datetime
import logging
import string
import random
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pprint import pp, pprint

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
from dnaseq2seq import stage
from dnaseq2seq import hapvarcalling

LOG_FORMAT  ='[%(asctime)s] %(process)d  %(name)s  %(levelname)s %(funcName)s: l.%(lineno)d  %(message)s '

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


import warnings
warnings.filterwarnings(action='ignore')

@dataclass
class RegionStopSignal:
    total_sus_regions: int
    total_sus_bp: int

@dataclass
class IndexedRegion:
    chrom: str
    start: int
    end: int
    index: int


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
    # logger.info(f"Writing variants to {Path(vcf_out).absolute()}")
    
    if 'cuda' in str(DEVICE):
        for idev in range(torch.cuda.device_count()):
            logger.info(f"Using CUDA device {idev} {torch.cuda.get_device_name({idev})}")
    else:
        logger.warning("No CUDA device found, this will be slow")
        try:
            torch.cuda.current_device()
        except Exception as ex:
            logger.error(ex)

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

    test_parallel_call(bam, bed, reference_fasta, model_path, classifier_path, max_batch_size, vcf_out, vcf_header_extras)
    # call_vars_in_parallel(
    #     bampath=bam,
    #     bed=bed,
    #     refpath=reference_fasta,
    #     model_path=model_path,
    #     classifier_path=classifier_path,
    #     threads=threads,
    #     max_batch_size=max_batch_size,
    #     vcf_out=vcf_out,
    #     vcf_header_extras=vcf_header_extras,
    #     show_progress=not kwargs.get('no_progress', False),
    # )

    logger.info(f"All variants saved to {vcf_out}")
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    if elapsed_seconds > 3600:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds / 3600 :.2f} hours")
    elif elapsed_seconds > 120:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds / 60 :.2f} minutes")
    else:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds :.2f} seconds")


def test_parallel_call(bam, bed, reference_fasta, model_path, classifier_path, max_batch_size, vcf_out, vcf_header_extras):
    """
    Test the parallel calling function
    """
    inputbed = bed
    bampath = bam
    refpath = reference_fasta
    window_size = 150
    min_reads = 5
    batch_size = 64
    window_step = 25

    _, modelconf = hapvarcalling.load_model(model_path)
    
    region_finder = stage.InitialStage(
        "region-finder", 
        target_func=None, 
        iterator_factory=make_region_finder_iterator, 
        iterator_kwargs={'inputbed': inputbed, 'bampath': bampath, 'refpath': refpath}
    )

    region_encoder_stage = stage.Stage(
        "region-encoder", 
        make_region_encoder_func(bampath, refpath, modelconf['max_read_depth'], window_size, min_reads, batch_size, window_step), 
        n_workers=8
    )
    variant_caller = hapvarcalling.VarHapCaller(model_path, classifier_path, refpath, bampath, max_batch_size, vcf_out, vcf_header_extras)
    variant_caller_stage = stage.Stage(
        "variant-caller", 
        variant_caller, 
        n_workers=1
    )
    region_finder.connect(region_encoder_stage)
    region_encoder_stage.connect(variant_caller_stage)
    region_finder.run()
    region_encoder_stage.run()  
    variant_caller_stage.run()

    for i, result in enumerate(variant_caller_stage.drain()):
        print(f"Result {i}:")
        pprint(result)
        if i % 5 == 0:
            pprint(variant_caller_stage.get_stats())
    
    region_finder.join(propogate_downstream=True)
    print(f"Done encoding regions, found {i+1} regions")
    print(f"Region finder stats:")
    pprint(region_finder.get_stats())
    print(f"Region encoder stats:")
    pprint(region_encoder_stage.get_stats())
    print(f"Variant caller stats:")
    pprint(variant_caller_stage.get_stats())
    

def make_region_finder_iterator(inputbed, bampath, refpath):
    return find_regions(inputbed, bampath, refpath)

def make_region_encoder_func(bampath, refpath, max_read_depth: int, window_size: int, min_reads: int, batch_size: int, window_step: int):
    return partial(encode_region, bampath=bampath, refpath=refpath, max_read_depth=max_read_depth, window_size=window_size, min_reads=min_reads, batch_size=batch_size, window_step=window_step)

def encode_region(idxregion: IndexedRegion, bampath, refpath, max_read_depth: int, window_size: int, min_reads: int, batch_size: int, window_step: int):
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


def add_ref_bases(encbases, reference, chrom, start, end, max_read_depth):
    """
    Add the reference sequence as read 0
    """
    refseq = reference.fetch(chrom, start, end)
    ref_encoded = bam.string_to_tensor(refseq)
    return torch.cat((ref_encoded.unsqueeze(1), encbases), dim=1)[:, 0:max_read_depth, :]


def _encode_region(aln, reference, chrom: str, start: int, end: int, max_read_depth: int, window_size: int, min_reads: int, batch_size: int, window_step: int):
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


def find_regions(inputbed, bampath, refpath):
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
    logger.info(f"Found {tot_regions} regions with {util.format_bp(tot_bases)} in {inputbed}")
    
    for idx, (chrom, window_start, window_end) in enumerate(util.split_large_regions(util.read_bed_regions(inputbed), max_region_size=10000)):

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

            logger.info(f"Identified regions {tot_size_bp} of {tot_bases} bp ({tot_size_bp / tot_bases * 100 :.2f} done)")
            for i, r in enumerate(sus_regions):
                sus_region_count += 1
                sus_region_bp += r[-1] - r[-2]
                yield IndexedRegion(chrom, r[-2], r[-1], idx)

        except Exception as ex:
            logger.error(f"Exception in region finder: {ex}")
            raise ex

    logger.info(f"Done finding regions, found {sus_region_count} regions with {util.format_bp(sus_region_bp)} bp")


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

if __name__ == "__main__":
    model_path = "/home/22319/data/variant-transformer/variant_transformer_runs/mbig_haptnt_ff1536/mbig_haptnt_ff1536_step211_20251224_163824.pt"
    clasifier_path = None
    bam = "/mnt/ri_share/Data/variant-transformer/gem-bams/99702111878_NA12878_S89/99702111878_NA12878_S89.cram"
    ref = "/mnt/ri_share/Data/variant-transformer/ref/human_g1k_v37_decoy_phiXAdaptr.fasta.gz"
    bed = "test.bed"
    vcf_out = "test.vcf"
    call(model_path=model_path, bam=bam, bed=bed, reference_fasta=ref, vcf_out=vcf_out, classifier_path=clasifier_path, max_batch_size=64, threads=1)