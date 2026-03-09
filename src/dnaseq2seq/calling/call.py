
import time

import logging
import random
from dataclasses import dataclass
from pprint import pp, pprint

from pathlib import Path

import torch
import torch.multiprocessing as mp
import pysam
import numpy as np


from dnaseq2seq import util
from dnaseq2seq import bam
from dnaseq2seq.calling import stage
from dnaseq2seq.calling.hapvarcalling_optimized import EncodedRegion
from dnaseq2seq.calling import hapvarcalling_optimized as hapvarcalling
from dnaseq2seq.calling import vcfwriter

LOG_FORMAT  ='[%(asctime)s] %(process)d  %(name)s  %(levelname)s %(funcName)s: l.%(lineno)d  %(message)s '

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
    mp.set_start_method('spawn') # Spawn is safer, but slower than fork. Fork is the default on linux
    start_time = time.perf_counter()
    threads = kwargs.get('threads', 1)
    max_batch_size = kwargs.get('max_batch_size', 128)
    logger.info(f"Using {threads} threads for encoding")
    
    # logger.info(f"Writing variants to {Path(vcf_out).absolute()}")
    
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

    run_calling_workflow(bam, bed, reference_fasta, model_path, classifier_path, max_batch_size, vcf_out, vcf_header_extras)


    logger.info(f"All variants saved to {vcf_out}")
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    if elapsed_seconds > 3600:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds / 3600 :.2f} hours")
    elif elapsed_seconds > 120:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds / 60 :.2f} minutes")
    else:
        logger.info(f"Total running time of call subcommand is: {elapsed_seconds :.2f} seconds")


def run_calling_workflow(bam, bed, reference_fasta, model_path, classifier_path, max_batch_size, vcf_out, vcf_header_extras):
    """
    Test the parallel calling function
    """
    gpu_count = torch.cuda.device_count()
    inputbed = bed
    bampath = bam
    refpath = reference_fasta
    window_size = 150
    min_reads = 5
    max_batch_size = 256
    window_step = 25
    region_finder_workers = 3
    region_encoder_workers = 12 * gpu_count
    emit_stats_output = False
    
    gpu_profiler = util.GPUProfiler(gpu_index=0, interval=0.2)
    total_regions, total_bases = util.count_bed(inputbed)
    logger.info(f"Found {total_regions} regions with {util.format_bp(total_bases)} in {inputbed}")

    logger.info(f"Loading model from {model_path}")
    _, modelconf = hapvarcalling.load_model(model_path, 'cpu')
    logger.info(f"Model loaded, max_read_depth: {modelconf['max_read_depth']}")
    
    bed_chunker_stage = stage.InitialStage(
        "bed-chunker",
        target_func=BedChunker(inputbed),
        custom_item_counter=region_base_pairs,
    )

    logger.info(f"Creating region finder stage with {region_finder_workers} workers")
    region_finder_stage = stage.Stage(
        "region-finder",
        RegionFinderWorker(bampath, refpath),
        n_workers=region_finder_workers,
        custom_item_counter=region_base_pairs,
        stats_update_interval=1,
    )
    bed_chunker_stage.connect(region_finder_stage)
    bed_chunker_stage.run()

    logger.info(f"Creating region encoder stage with {region_encoder_workers} workers")
    region_encoder = RegionEncoderWorker(bampath, refpath, modelconf['max_read_depth'], window_size, min_reads, max_batch_size, window_step)
    region_encoder_stage = stage.Stage(
        "region-encoder", 
        region_encoder, 
        n_workers=region_encoder_workers,
        stats_update_interval=1,
    )
    region_finder_stage.connect(region_encoder_stage)
    region_finder_stage.run()

    logger.info(f"Creating variant caller stage")
    variant_caller = hapvarcalling.OptimizedVarHapCaller(model_path, refpath, max_batch_size, device='cuda', enable_double_buffering=False, use_pinned_memory=False)
    variant_caller_stage = stage.Stage(
        "variant-caller", 
        variant_caller, 
        n_workers=gpu_count,
        custom_item_counter=encoded_region_base_pairs,
        stats_update_interval=1,
    )
    region_encoder_stage.connect(variant_caller_stage)
    region_encoder_stage.run()

    vcf_writer_workers = 4
    logger.info(f"Creating vcf writer stage with {vcf_writer_workers} workers")
    vcf_writer = vcfwriter.VCFWriter(refpath=refpath, bampath=bampath, classifier_model=classifier_path)
    vcf_writer_stage = stage.Stage(
        "vcf-writer", 
        vcf_writer, 
        n_workers=vcf_writer_workers,
        stats_update_interval=1,
    )
    variant_caller_stage.connect(vcf_writer_stage)
    variant_caller_stage.run()

    logger.info(f"Creating vcf collector stage")
    vcf_collector = vcfwriter.VCFCollector(vcf_out, vcf_header_extras=vcf_header_extras)
    vcf_collector_stage = stage.Stage(
        "vcf-collector",
        vcf_collector,
        n_workers=1,
        stats_update_interval=1,
    )
    vcf_writer_stage.connect(vcf_collector_stage)
    vcf_writer_stage.run()
    vcf_collector_stage.run()
    
    gpu_profiler.start()
    
    last_log_time = time.perf_counter()
    log_interval = 5.0
    try:
        for i, result in enumerate(vcf_collector_stage.drain()):
            now = time.perf_counter()
            if now - last_log_time >= log_interval:
                last_log_time = now
                bp_chunked = bed_chunker_stage.stats['custom_counter']
                bp_identified = region_finder_stage.stats['custom_counter']
                regions_encoded = region_encoder_stage.stats['items_processed']
                bp_called = variant_caller_stage.stats['custom_counter']
                vcf_records_produced = vcf_writer_stage.stats['items_processed']
                # finder_queue_length = bed_chunker_stage.stats['items_processed'] - region_finder_stage.stats['items_received']
                # encoded_queue_length = region_finder_stage.stats['items_processed'] - region_encoder_stage.stats['items_received']
                # calling_queue_length = region_encoder_stage.stats['items_processed'] - variant_caller_stage.stats['items_received']
                # vcf_queue_length = variant_caller_stage.stats['items_processed'] - vcf_writer_stage.stats['items_received']
                # collector_queue_length = vcf_records_produced - vcf_collector_stage.stats['items_received']
                pct_complete = bp_identified / total_bases * 100.0
                # logger.info(f"Region finder received: {region_finder_stage.stats['items_received']}, processed: {region_finder_stage.stats['items_processed']}")
                # logger.info(f"Region encoder received: {region_encoder_stage.stats['items_received']}, processed: {region_encoder_stage.stats['items_processed']}")
                # logger.info(f"Variant caller received: {variant_caller_stage.stats['items_received']}, processed: {variant_caller_stage.stats['items_processed']}")
                # logger.info(f"VCF writer received: {vcf_writer_stage.stats['items_received']}, processed: {vcf_writer_stage.stats['items_processed']}")
                # logger.info(f"VCF collector received: {vcf_collector_stage.stats['items_received']}, processed: {vcf_collector_stage.stats['items_processed']}")
                logger.info(f"Iter {i} BP chunked: {util.format_bp(bp_chunked)}, BP identified: {util.format_bp(bp_identified)}, BP called: {util.format_bp(bp_called)} Regions encoded: {regions_encoded} VCF records: {vcf_records_produced}, Progress: {pct_complete:.2f}%")
                # logger.info(f"Finder queue: {finder_queue_length}, Encoder queue: {encoded_queue_length}, Caller queue: {calling_queue_length}, VCF queue: {vcf_queue_length}, Collector queue: {collector_queue_length}")
        
        bed_chunker_stage.join(propogate_downstream=True)
    except KeyboardInterrupt:
        logger.error("Keyboard interrupt received, stopping stages")
        bed_chunker_stage.abort()
        region_finder_stage.abort()
        region_encoder_stage.abort()
        variant_caller_stage.abort()
        vcf_writer_stage.abort()
        vcf_collector_stage.abort()
        raise KeyboardInterrupt
    gpu_profiler.stop()
    
    if emit_stats_output:
        from dnaseq2seq.calling import stats_display
        all_stats = [bed_chunker_stage.get_stats(), region_finder_stage.get_stats(), region_encoder_stage.get_stats(), variant_caller_stage.get_stats(), vcf_writer_stage.get_stats(), vcf_collector_stage.get_stats()]
        all_names = ["bed-chunker", "region-finder", "region-encoder", "variant-caller", "vcf-writer", "vcf-collector"]
        stats_display.print_stats(all_stats, stage_names=all_names)
        print(f"GPU profiler report:")
        pprint(gpu_profiler.get_report())


def region_base_pairs(idxregion: IndexedRegion):
    if idxregion is None:
        return 0
    return idxregion.end - idxregion.start
    
def encoded_region_base_pairs(encoded_region: EncodedRegion):
    if encoded_region is None:
        return 0
    return encoded_region.region[2] - encoded_region.region[1]

class RegionEncoderWorker:
    """
    Callable that takes an IndexedRegion (likely produced by a BedChunker) and encodes the reads in the region into a tensor.
    """
    def __init__(self, bampath, refpath, max_read_depth: int, window_size: int, min_reads: int, batch_size: int, window_step: int):
        self.bampath = bampath
        self.refpath = refpath
        self.max_read_depth = max_read_depth
        self.window_size = window_size
        self.min_reads = min_reads
        self.batch_size = batch_size
        self.window_step = window_step
        self.aln = None
        self.reference = None
    
    def _init_files(self):
        self.aln = pysam.AlignmentFile(self.bampath, reference_filename=self.refpath)
        self.reference = pysam.FastaFile(self.refpath)
        torch.set_num_threads(2)
    
    def __call__(self, idxregion: IndexedRegion):
        if self.aln is None or self.reference is None:
            self._init_files()
        
        all_encoded = []
        all_starts = []
        all_depths = []
        for encoded_region, start_positions, depth_profiles in _encode_region(self.aln, self.reference, idxregion.chrom, idxregion.start, idxregion.end, self.max_read_depth,
                                                     window_size=self.window_size, min_reads=self.min_reads, batch_size=self.batch_size, window_step=self.window_step):
            all_encoded.append(encoded_region)
            all_starts.extend(start_positions)
            all_depths.append(depth_profiles)
        if len(all_encoded) > 1:
            encoded = torch.concat(all_encoded, dim=0)
            depths = np.concatenate(all_depths, axis=0)
        elif len(all_encoded) == 1:
            encoded = all_encoded[0]
            depths = all_depths[0]
        else:
            logger.error(f"Uh oh, did not find any encoded paths!, region is {idxregion.chrom}:{idxregion.start}-{idxregion.end}")
            return None
        encoded.share_memory_()
        midpoint_depths = depths[:, depths.shape[1] // 2]
        return EncodedRegion(
            encoded_pileup=encoded,
            region=(idxregion.chrom, idxregion.start, idxregion.end),
            start_positions=all_starts,
            index=idxregion.index,
            midpoint_depths=midpoint_depths,
        )


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
    :returns: Generator for tuples of (batch tensor, list of start positions, depth profiles array)
    """
    window_start = int(start - 0.7 * window_size)  # We start with regions a bit upstream of the focal / target region
    batch = []
    batch_offsets = []
    batch_depths = []
    readwindow = bam.ReadWindowFast(aln, chrom, start - 150, end + window_size)
    logger.debug(f"Encoding region {chrom}:{start}-{end}")
    returned_count = 0
    while window_start <= (end - 0.2 * window_size):
        try:
            #logger.debug(f"Getting reads from  readwindow: {window_start} - {window_start + window_size}")
            enc_reads = readwindow.get_window(window_start, window_start + window_size, max_reads=max_read_depth)
            batch_depths.append(bam.depth_from_window(enc_reads))
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
            yield encodedreads, batch_offsets, np.stack(batch_depths)
            batch = []
            batch_offsets = []
            batch_depths = []

    # Last few
    if batch:
        encodedreads = torch.stack(batch, dim=0).cpu() # Keep encoded tensors on cpu for now
        returned_count += 1
        yield encodedreads, batch_offsets, np.stack(batch_depths)

    if not returned_count:
        logger.debug(f"Region {chrom}:{start}-{end} has only low coverage areas, not encoding data")


def gen_suspicious_spots(aln, chrom, start, stop, ref):
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



def cluster_positions_for_window(window, aln, ref, maxdist=100):
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
            gen_suspicious_spots(aln, chrom, window_start, window_end, ref),
            maxdist=maxdist,
        )
    ]


class BedChunker:
    """
    Iterable that reads a BED file, splits large regions into ~10kb chunks,
    and yields BedChunk items in genome order. Designed for use as the
    target_func of an InitialStage.
    """

    def __init__(self, inputbed, max_region_size=10000):
        self.inputbed = inputbed
        self.max_region_size = max_region_size

    def __iter__(self):
        for idx, (chrom, start, end) in enumerate(
            util.split_large_regions(util.read_bed_regions(self.inputbed), max_region_size=self.max_region_size)
        ):
            yield IndexedRegion(chrom=chrom, start=start, end=end, index=idx)


class RegionFinderWorker:
    """
    Callable that takes a BedChunk and finds suspicious regions within it
    using pileup-based variant detection. Returns a MultiResult of
    IndexedRegion objects, or SkipResult if no suspicious regions are found.

    Opens BAM/reference file handles lazily on first invocation so each
    worker process gets its own handles.
    """

    def __init__(self, bampath, refpath):
        self.bampath = bampath
        self.refpath = refpath
        self.aln = None
        self.ref = None

    def _init_files(self):
        torch.set_num_threads(2)
        self.aln = pysam.AlignmentFile(self.bampath, reference_filename=self.refpath)
        self.ref = pysam.FastaFile(self.refpath)

    def __call__(self, chunk: IndexedRegion):
        if self.aln is None or self.ref is None:
            self._init_files()

        sus_regions = cluster_positions_for_window(
            (chunk.chrom, chunk.index, chunk.start, chunk.end),
            aln=self.aln,
            ref=self.ref,
            maxdist=100,
        )
        merged = util.merge_overlapping_regions(sus_regions)

        if not merged:
            return stage.SkipResult()

        results = [
            IndexedRegion(chunk.chrom, r[-2], r[-1], chunk.index)
            for r in merged
        ]
        return stage.MultiResult(results)


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
    :returns: Generator for tuples of (batch tensor, list of start positions, depth profiles array)
    """
    window_start = int(start - 0.7 * window_size)  # We start with regions a bit upstream of the focal / target region
    batch = []
    batch_offsets = []
    batch_depths = []
    readwindow = bam.ReadWindow(aln, chrom, start - 150, end + window_size)
    logger.debug(f"Encoding region {chrom}:{start}-{end}")
    returned_count = 0
    while window_start <= (end - 0.2 * window_size):
        try:
            #logger.debug(f"Getting reads from  readwindow: {window_start} - {window_start + window_size}")
            enc_reads = readwindow.get_window(window_start, window_start + window_size, max_reads=max_read_depth)
            batch_depths.append(bam.depth_from_window(enc_reads))
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
            yield encodedreads, batch_offsets, np.stack(batch_depths)
            batch = []
            batch_offsets = []
            batch_depths = []

    # Last few
    if batch:
        encodedreads = torch.stack(batch, dim=0).cpu() # Keep encoded tensors on cpu for now
        returned_count += 1
        yield encodedreads, batch_offsets, np.stack(batch_depths)

    if not returned_count:
        logger.debug(f"Region {chrom}:{start}-{end} has only low coverage areas, not encoding data")

if __name__ == "__main__":
    start = time.perf_counter()
    model_path = "mbig_haptnt_ff1536_averaged2.pt"
    # model_path = "med_gqamodel_g44o20_averaged.pt"
    # model_path = "mbig_g44o20_averaged180-200.pt"
    model_prefix = Path(model_path).stem
    # model_path = "averaged_model.pt"
    clasifier_path = None
    bam = "/mnt/ri_share/Data/variant-transformer/gem-bams/99702111878_NA12878_S89/99702111878_NA12878_S89.cram"
    # bam = "/data2/brendan/99702111878_NA12878_S89.cram"
    ref = "/mnt/ri_share/Data/variant-transformer/ref/human_g1k_v37_decoy_phiXAdaptr.fasta.gz"
    bed = "perftest.bed"

    # chunker = BedChunker(bed)
    # for chunk in chunker:
    #     print(chunk)
    
    # bed = "/mnt/ri_share/Data/variant-transformer/chr21_22_valregion.bed"
    bed_suffix = Path(bed).stem
    vcf_out = f"{model_prefix}_NA12878_S89_{bed_suffix}.vcf"
    call(model_path=model_path, bam=bam, bed=bed, reference_fasta=ref, vcf_out=vcf_out, classifier_path=clasifier_path, max_batch_size=64, threads=1)
    end = time.perf_counter()
    print(f"Total running time of call subcommand is: {end - start :.3f} seconds")