
import random
import time
import traceback
import numpy as np
from typing import List, Tuple
import logging
from collections import defaultdict
import bisect

import pysam
import torch

from dnaseq2seq import util

logger = logging.getLogger(__name__)

FEATURE_NUM=10



class LowReadCountException(Exception):
    """
    Region of bam file has too few spanning reads for variant detection
    """
    pass

def readkey(read):
    suf = "-1" if read.is_read1 else "-2"
    suf2 = "-" + str(read.cigar) + "-" + str(read.reference_start)
    return read.query_name + suf + suf2


class ReadCache:
    """
    Simple cache to store read encodings
    """

    def __init__(self):
        self.cache = {}

    def __getitem__(self, read):
        key = readkey(read)
        if key not in self.cache:
            # self.cache[key] = (alnstart(read), encode_read(read).char(), read.get_aligned_pairs(matches_only=True), ) # gen_cumulative_offsets(read)
            self.cache[key] = ReadEncoder(read)

        return self.cache[key]

    def __contains__(self, item):
        return readkey(item) in self.cache


class ReadEncoder:

    def __init__(self, read: pysam.AlignedSegment):
        self.read = read
        self.alnpairs = read.get_aligned_pairs(matches_only=True)
        self.alnstart = next(a for a in self.alnpairs if a[1] is not None)[1]
        self.alnend = next(a for a in reversed(self.alnpairs) if a[1] is not None)[1]
        self.encoded_chunk = None
        self.first_base_to_encode = None # Read offset of first base to grab
        self.first_base_ref_coord = None # Reference coordinate of first base to grab
        self.window_start = None # Ref coord start of window
        self.window_end = None # Ref coord end of window

    def __len__(self):
        return self.read.query_length

    def extend(self, new_end):
        """
        Extend the current encoded chunk to the given reference position end in a semi-efficient manner
        We don't re-encode any of the bases that are part of the current chunk, just the new ones
        :param new_end: Reference coordinate of the end of the new window
        """
        assert self.encoded_chunk is not None, f"No encoded chunk to extend"
        bases_to_skip = self.first_base_to_encode + self.encoded_chunk.shape[0]
        new_enc = encode_read(self.read,
                              prepad=0,
                              tot_length=new_end - self.window_end, # Includes prepad bases, so always equal to full window length
                              skip=bases_to_skip)
        encoded = torch.cat((self.encoded_chunk, new_enc), dim=0)

        self.encoded_chunk = encoded
        self.window_end = new_end

        return encoded

    def shift(self, new_start, new_end):
        """
        Create a new encoded region by shifting the current region to the new location. If the new location is
        outside the current encoded region, we re-encode the whole thing
        """
        assert new_start > self.window_start, f"Can't shift to a position before the current window start (asked for {new_start} but current start is {self.window_start})"
        if self.encoded_chunk is None or new_start > self.window_end:
            return self.get_encoded(new_start, new_end)

        encoded = self.extend(new_end)[new_start - self.window_start:, :]
        self.encoded_chunk = encoded
        self.window_start = new_start
        self.window_end = new_end
        return encoded

    def _find_start(self, start):
        """ Use bisect library to find entry in aligned pairs that corresponds to the start of the window """
        idx = bisect.bisect_left(self.alnpairs, start, key=lambda x: x[1])  # Including the key function here restricts everything to python >=3.10
        return self.alnpairs[idx]

    def _from_scratch(self, ref_start, ref_end):
        """
        Discard existing information and re-encode the read from scratch
        :return: Encoded tensor representing the read in the given reference window
        """
        first_base_to_encode, first_base_ref_coord = self._find_start(ref_start)
        encoded = encode_read(self.read,
                              prepad=first_base_ref_coord - ref_start,
                              tot_length=ref_end - ref_start, # Includes prepad bases, so always equal to full window length
                              skip=first_base_to_encode)
        assert encoded.shape[0] == ref_end - ref_start, f"Encoded read length {encoded.shape[0]} doesn't match window length {ref_end - ref_start}"

        self.encoded_chunk = encoded
        self.first_base_to_encode = first_base_to_encode
        self.first_base_ref_coord = first_base_ref_coord
        self.window_start = ref_start
        self.window_end = ref_end
        return encoded

    def get_encoded(self, ref_start, ref_end):
        """
        Encode a portion of the read corresponding to the given reference window
        This will shift the current encoded region if the reference start coord is between the existing (previous) start and end, otherwise re-encode the whole thing
        :param ref_start: Reference coordinate of the start of the window
        :param ref_end: Reference coordinate of the end of the window
        :return: Encoded tensor
        """
        # TODO This is problematic because the first call from _from_scratch involves finding the 'anchor' position for the read, but the more
        # windows we generate via 'shift' operations the more that anchor becomes shifted and inaccurate due to indels in the read
        # So we need to find a way to re-anchor the read to the reference sequence at some point
        return self._from_scratch(ref_start, ref_end)

        # if self.encoded_chunk is None:
        #     return self._from_scratch(ref_start, ref_end)
        # elif self.window_start < ref_start < self.window_end:
        #     return self.shift(ref_start, ref_end)
        # else:
        #     return self._from_scratch(ref_start, ref_end)


def get_mapping_coords(window_start: int, window_end: int, read_length: int, read_idx_anchor: int, ref_idx_anchor: int) -> Tuple[int, int, int]:
    """
    Calculate the start and end indexes of the read that map to the window as well as the offset of the window start
    :param window_start: Start of the window
    :param window_end: End of the window
    :param read_length: total number of bases in read
    :param read_idx_anchor: Index into read that corresponds to ref_start reference coordinate
    :param ref_idx_anchor: Reference coordinate of the start of the read
    :return: Tuple of read offset, window offset, and length of bases that map to window
    """

    # Reference coordinate of first read base
    read_start_ref_coord = ref_idx_anchor - read_idx_anchor

    # Compute first base of read that maps to window
    read_idx_start = max(0, window_start - read_start_ref_coord)

    # Reference coordinate of first base of read that maps to window
    ref_idx_start = max(window_start, read_start_ref_coord)

    # Compute last base of read that maps to window
    read_idx_end = min(read_length,
                       read_idx_start + (window_end - ref_idx_start),
                       )

    # Compute position in window the first base of the read maps
    window_offset = max(0, ref_idx_start - window_start)

    # Compute number of bases included in window
    num_bases = read_idx_end - read_idx_start

    return read_idx_start, window_offset, num_bases


class ReadWindow:

    def __init__(self, aln, chrom, start, end, min_mq=-1, margin_size=150):
        self.aln = aln
        self.start = start
        self.end = end
        self.margin_size = margin_size # Should be about a read length
        self.chrom = chrom
        self.min_mq = min_mq
        self.cache = ReadCache()  # Cache for encoded reads
        self.bypos = self._fill() # Maps read start positions to actual reads

    def _fill(self):
        bypos = defaultdict(list)
        for i, read in enumerate(self.aln.fetch(self.chrom, self.start - self.margin_size, self.end)):
            if read is not None and read.mapping_quality > self.min_mq:
                bypos[alnstart(read)].append(read)
        return bypos

    def get_window(self, start, end, max_reads, downsample_read_count=None):
        assert self.start <= start < self.end, f"Start coordinate must be between beginning and end of window"
        assert self.start < end <= self.end, f"End coordinate must be between beginning and end of window"
        start_time = time.perf_counter()
        allreads = []
        for pos in range(start - self.margin_size, end):
            for read in self.bypos[pos]: #
                # if pos > end or (pos + read.query_length) < start: # Check to make sure read overlaps window
                #     continue
                overlap = read.get_overlap(start, end) # Can return None sometimes
                if overlap is not None and overlap > 1:
                    allreads.append((pos, read))

        # TODO realign reads around window by using get_blocks() or get_aligned_pairs() to identify aligned regions around start of window
        # I think this should alter the 'read_start' entry that is the first part of the tuple in allreads
        if len(allreads) < 5:
            raise LowReadCountException(f"Only {len(allreads)} reads in window")
            
        if downsample_read_count:
            num_reads_to_sample = downsample_read_count
        else:
            num_reads_to_sample = max_reads

        if len(allreads) > num_reads_to_sample:
            logger.debug(f"Window has {len(allreads)}, downsampling to {num_reads_to_sample}")
            allreads = random.sample(allreads, num_reads_to_sample)
            allreads = sorted(allreads, key=lambda x: x[0])

        window_size = end - start
        t = torch.zeros(window_size, max_reads, 10, device='cpu', dtype=torch.int8)
        for i, (readstart, read) in enumerate(allreads):
            readenc = self.cache[read]
            t[:, i, :] = readenc.get_encoded(start, end)

        end_time = time.perf_counter()
        logger.debug(f"Encoded {len(allreads)} reads in {end_time - start_time:.2f} seconds (region {start}-{end})")
        return t


def encode_read(read, prepad=0, tot_length=None, skip=0):
    """
    Encode the given read into a tensor
    :param read: Read to be encoded (typically pysam.AlignedSegment)
    :param prepad: Leading zeros to prepend
    :param skip: Skip this many bases at the beginning of the read
    :param tot_length: If not None, desired total 'length' (dimension 0) of tensor
    """
    if tot_length:
        assert prepad < tot_length, f"Cant have more padding than total length"
    bases = []
    for i in range(prepad):
        bases.append(torch.zeros(FEATURE_NUM))

    try:
        for i, t in enumerate(iterate_bases(read, skip=skip)):
            bases.append(t)
            if tot_length is not None:
                if len(bases) >= tot_length:
                    break
    except StopIteration:
        pass

    if tot_length is not None:
        while len(bases) < tot_length:
            bases.append(torch.zeros(FEATURE_NUM))
    return torch.stack(tuple(bases)).char()


def base_index(base):
    base = base.upper()
    if base == 'A':
        return 0
    elif base == 'C':
        return 1
    elif base == 'G':
        return 2
    elif base == 'T':
        return 3
    raise ValueError(f"Expected [ACTG], got {base}")


def update_from_base(base, tensor):
    if base == 'A':
        tensor[0] = 1
    elif base == 'C':
        tensor[1] = 1
    elif base == 'G':
        tensor[2] = 1
    elif base == 'T':
        tensor[3] = 1
    elif base == 'N':
        tensor[0:4] = 1
    elif base == '-':
        tensor[0:4] = 0
    return tensor


def encode_basecall(base, qual, consumes_ref_base, consumes_read_base, strand, clipped, mapq):
    ebc = torch.zeros(10).char() # Char is a signed 8-bit integer, so ints from -128 - 127 only
    ebc = update_from_base(base, ebc)
    ebc[4] = int(round(qual / 10))
    ebc[5] = consumes_ref_base # Consumes a base on reference seq - which means not insertion
    ebc[6] = consumes_read_base # Consumes a base on read - so not a deletion
    ebc[7] = 1 if strand else 0
    ebc[8] = 1 if clipped else 0
    ebc[9] = int(round(mapq / 10))
    return ebc


def decode(t):
    t = t.squeeze()
    if torch.sum(t[0:4]) == 0.0:
        return '-'
    else:
        return util.INDEX_TO_BASE[t[0:4].argmax()]


def string_to_tensor(bases):
    return torch.vstack([encode_basecall(b, 50, 0, 0, 0, 0, 50) for b in bases])


def target_string_to_tensor(bases):
    """
    Encode the string into a tensor with base index values, like class labels, for each position
     The tensor looks like [0,1,2,1,0,2,3,0...]
    """
    result = torch.tensor([base_index(b) for b in bases]).long()
    return result


def pad_zeros(pre, data, post):
    if pre:
        prepad = torch.zeros(pre, data.shape[-1], dtype=data.dtype)
        data = torch.cat((prepad, data))
    if post:
        postpad = torch.zeros(post, data.shape[-1], dtype=data.dtype)
        data = torch.cat((data, postpad))
    return data


def iterate_bases(rec, skip=0):
    """
    Generate encoded base calls for the given variant record, this version does NOT
    insert gaps into the bases if there's a deletion in the cigar - it just reads right on thru
    :param rec: pysam VariantRecord
    :return: Generator for encoded base calls
    """
    cigtups = rec.cigartuples
    if cigtups is None:
        cigtups = [(0, len(rec.query_sequence))]
    bases = rec.query_sequence
    quals = rec.query_qualities
    cig_index = 0
    n_bases_cigop = cigtups[cig_index][1]
    cigop = cigtups[cig_index][0]
    is_ref_consumed = cigop in {0, 2, 4, 5, 7}  # 2 is deletion
    is_seq_consumed = cigop in {0, 1, 3, 4, 7}  # 1 is insertion, 3 is 'ref skip'
    is_clipped = cigop in {4, 5}
    if bases is None:
        logger.warning(f"No bases for {rec.query_name}, skipping")
        return
    if quals is None:
        logger.warning(f"No quals for {rec.query_name}, skipping")
        return

    for i, (base, qual) in enumerate(zip(bases, quals)):
        if i < skip:
            continue
        yield encode_basecall(base, qual, is_ref_consumed, is_seq_consumed, rec.is_reverse, is_clipped, rec.mapping_quality)
        n_bases_cigop -= 1
        if n_bases_cigop <= 0:
            cig_index += 1
            if cig_index >= len(cigtups):
                break
            n_bases_cigop = cigtups[cig_index][1]
            cigop = cigtups[cig_index][0]
            is_ref_consumed = cigop in {0, 2, 4, 5, 7}
            is_seq_consumed = cigop in {0, 1, 3, 4, 7}
            is_clipped = cigop in {4, 5}


def rec_tensor_it(read, minref):
    for i in range(alnstart(read) - minref):
        yield torch.zeros(FEATURE_NUM)

    try:
        for t in iterate_bases(read):
            yield t
    except StopIteration:
        pass

    while True:
        yield torch.zeros(FEATURE_NUM)


def emit_tensor_aln(t):
    """
    Expecting t [read, position, bases]
    """
    for read_idx in range(t.shape[1]):
        for pos_idx in range(t.shape[0]):
            b = decode(t[pos_idx, read_idx, :])
            print(b, end='')
        print()


def alnstart(read):
    """
    If the first cigar element is hard or soft clip, return read.reference_start - size of first cigar element,
    otherwise return read.reference_start
    """
    if read.cigartuples is not None and read.cigartuples[0][0] in {4, 5}:
        return read.reference_start - read.cigartuples[0][1]
    else:
        return read.reference_start


def _consume_n(it, n):
    """ Yield the first n elements of the given iterator """
    for i in range(n):
        yield next(it)


def encode_pileup3(reads, start, end):
    """
    Convert a list of reads (pysam VariantRecords) into a single tensor

    :param reads: List of pysam reads
    :param start: Genomic start coordinate
    :param end: Genomic end coordinate
    :return: Tensor with shape [position, read, features]
    """
    # minref = min(alnstart(r) for r in reads)
    # maxref = max(alnstart(r) + r.query_length for r in reads)
    isalt = ["alt" in r.query_name for r in reads]
    everything = []
    for readnum, read in enumerate(reads):
        try:
            readencoded = [enc.char() for enc in _consume_n(rec_tensor_it(read, start), end-start)]
            everything.append(torch.stack(readencoded))
        except Exception as ex:
            logger.warn(f"Error processing read {read.query_name}: {ex}, skipping it")
            traceback.print_exception(type(ex), ex, ex.__traceback__)
            raise ex

    return torch.stack(everything).transpose(0,1), torch.tensor(isalt)


def ensure_dim(readtensor, seqdim, readdim):
    """
    Trim or zero-pad the readtensor to make sure it has exactly 'seqdim' size for the sequence length
    and 'readdim' size for the read dimension
    Assumes readtensor has dimension [seq, read, features]
    :return:
    """
    if readtensor.shape[0] >= seqdim:
        readtensor = readtensor[0:seqdim, :, :]
    else:
        pad = torch.zeros(seqdim - readtensor.shape[0], readtensor.shape[1], readtensor.shape[2], dtype=readtensor.dtype)
        readtensor = torch.cat((readtensor, pad), dim=0)

    if readtensor.shape[1] >= readdim:
        readtensor = readtensor[:, 0:readdim, :]
    else:
        pad = torch.zeros(readtensor.shape[0], readdim - readtensor.shape[1], readtensor.shape[2], dtype=readtensor.dtype)
        readtensor = torch.cat((readtensor, pad), dim=1)
    return readtensor


def format_cigar(cig):
    return cig.replace("M", "M ").replace("S", "S ").replace("I", "I ").replace("D", "D ")


def reads_spanning(bam, chrom, pos, max_reads):
    """
    Return a list of reads spanning the given position, generally attempting to take
    reads in which 'pos' is approximately in the middle of the read
    :return : list of reads spanning the given position
    """
    start = pos - 10
    bamit = bam.fetch(chrom, start)
    reads = []
    try:
        read = next(bamit)
        while read.reference_start < pos:
            if read.reference_end is not None and read.reference_start < pos < read.reference_end:
                reads.append(read)
            read = next(bamit)
    except StopIteration:
        pass
    mid = len(reads) // 2
    return reads[max(0, mid-max_reads//2):min(len(reads), mid+max_reads//2)]


def reads_spanning_range(bam, chrom, start, end):
    """
    Return a list of reads spanning the given position, generally attempting to take
    reads in which 'pos' is approximately in the middle of the read
    :return : list of reads spanning the given position
    """
    bamit = bam.fetch(chrom, start)
    reads = []
    try:
        read = next(bamit)
        while read.reference_start < end:
            if read.reference_end is not None and read.reference_end > start:
                reads.append(read)
            read = next(bamit)
    except StopIteration:
        pass
    return reads


def encode_with_ref(chrom, pos, ref, alt, bam, fasta, maxreads):
    """
    Fetch reads from the given BAM file, encode them into a single tensor, and also
    fetch & create the corresponding ref sequence and alternate sequence based on the given chrom/pos/ref/alt coords
    :returns: Tuple of encoded reads, reference sequence, alt sequence
    """
    reads = reads_spanning(bam, chrom, pos, max_reads=maxreads)
    if len(reads) < 5:
        raise ValueError(f"Not enough reads spanning {chrom} {pos}, aborting")

    minref = min(alnstart(r) for r in reads)
    maxref = max(alnstart(r) + r.query_length for r in reads)
    reads_encoded, _ = encode_pileup3(reads, minref, maxref)
    pos = pos - 1 # Believe fetch() is zero-based, but input typically in 1-based VCF coords?
    refseq = fasta.fetch(chrom, minref, maxref) 
    assert refseq[pos - minref: pos-minref+len(ref)] == ref, f"Ref sequence / allele mismatch (found {refseq[pos - minref: pos-minref+len(ref)]})"
    altseq = refseq[0:pos - minref] + alt + refseq[pos-minref+len(ref):]
    assert len(refseq) == reads_encoded.shape[0], f"Length of reference sequence doesn't match width of encoded read tensor ({len(refseq)} vs {reads_encoded.shape[0]})"

    ref_encoded = string_to_tensor(refseq)
    encoded_with_ref = torch.cat((ref_encoded.unsqueeze(1), reads_encoded), dim=1)[:, 0:maxreads, :]

    return encoded_with_ref, refseq, altseq


def encode_and_downsample(chrom, start, end, bam, refgenome, maxreads, num_samples, downsample_frac=0.2):
    """
    Returns 'num_samples' tuples of read tensors and corresponding reference sequence and alt sequence for the given
    chrom/pos/ref/alt. Each sample is for the same position, but contains a random sample of 'maxreads' from all of the
    reads overlapping the position.
    :param maxreads: Number of reads to downsample to
    :returns: Tuple of encoded reads, reference sequence, alt sequence
    """
    allreads = reads_spanning_range(bam, chrom, start, end)
    if len(allreads) < 5:
        raise ValueError(f"Not enough reads in {chrom}:{start}-{end}, aborting")

    if (len(allreads) // maxreads) < num_samples:
        num_samples = max(1, len(allreads) // maxreads)

    #logger.info(f"Taking {num_samples} samples from {chrom}:{start}-{end}  ({len(allreads)} total reads")
    readwindow = ReadWindow(bam, chrom, start, end, margin_size=50000)
    for i in range(num_samples):
        reads_to_sample = maxreads
        if np.random.rand() < downsample_frac:
            reads_to_sample = maxreads // 2
        reads_encoded = readwindow.get_window(start, end, max_reads=maxreads, downsample_read_count=reads_to_sample)
        refseq = refgenome.fetch(chrom, start, end)
        ref_encoded = string_to_tensor(refseq)
        encoded_with_ref = torch.cat((ref_encoded.unsqueeze(1), reads_encoded), dim=1)[:, 0:maxreads, :]

        yield encoded_with_ref, (start, end)

def read_query_length(read: pysam.AlignedSegment):
    return read.query_alignment_length

if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



    from functools import partial
    from dnaseq2seq.call import gen_suspicious_spots, old_gen_suspicious_spots
    from susposfinder import get_sus_positions, get_sus_positions_cached
    # bam = "/Users/brendan/data/WGS/hg002_chr22.bam"
    bam = "/Users/brendan/data/WGS/99702152385_GM24385_500ng_S92_chr21and22.cram"
    ref = "/Users/brendan/data/ref_genome/human_g1k_v37_decoy_phiXAdaptr.fasta.gz"
    chrom = "22"
    aln = pysam.AlignmentFile(bam, reference_filename=ref)

    start = 27710000
    end =   27710150

    # for r in aln.fetch(chrom, start, end):
    #     l = lc[r]
    #     print(f"{r.query_name} : {l}")
    #
    # for r in aln.fetch(chrom, start, end):
    #     l = lc[r]
    #     print(f"{r.query_name} : {l}")

    # t0 = time.perf_counter()
    # for r in aln.fetch(chrom, start, end):
    #     print(r.query_name, r.reference_start, r.reference_end, r.cigarstring, r.mapping_quality)
    # t1 = time.perf_counter()
    # print(f"Time to fetch reads: {t1-t0:.4f} seconds")

    # sp = util.get_sus_positions(r, pysam.FastaFile(ref), min_qual=10)
    # print(sp)

    # t0 = time.perf_counter()
    # logger.info("Starting old method")
    # pl_old = old_gen_suspicious_spots(bam, chrom, start, end, reference_fasta=ref, min_indel_count=3, min_mismatch_count=3)
    # for p in pl_old:
    #     print(p + 1)
    t1 = time.perf_counter()
    start = 27717050

    for i in range(100):
        pl = gen_suspicious_spots(bam, chrom, start, start + 150, reference_fasta=ref, min_indel_count=5, min_mismatch_count=3)
        for p in pl:
            print(f"{p}")
        start += 5000
    t2 = time.perf_counter()
    #
    # print(f"Old method took {t1-t0:.2f} seconds")
    print(f"New method took {t2-t1:.2f} seconds")


    # c = [(start, end) for start, end in util.cluster_positions(
    #         gen_suspicious_spots(bam, chrom, 27717050, 27728250, reference_fasta=ref, min_indel_count=3, min_mismatch_count=3),
    #         maxdist=50,
    # )]
    #
    # print(c)
    # tot = sum([end-start for start, end in c])
    # print(f"Total of {tot} bases in suspicious regions, region count: {len(c)}")
    # for b in gen_suspicious_spots(bam, "22", 27717050, 27717250, reference_fasta=ref):
    #     print(b)

    # rw = ReadWindow(aln, chrom, 27718050, 27718580, margin_size=50000)
    # t = rw.get_window(27718050, 27718150, 100)
    # p = util.to_pileup(t)
    # print(p)
    #
    # print("The next window")
    # t = rw.get_window(27718100, 27718200, 100)
    # p = util.to_pileup(t)
    # print(p)

    # ws = 27717800
    # we = 27717820
    # fetcher = aln.fetch("22", ws, we)
    # r = next(fetcher)
    # re = ReadEncoder(r)
    # enc0 = re.get_encoded(27717800, 27717810)
    # print(util.to_pileup(enc0.unsqueeze(1)))
    # enc1 = re.extend(27717820)
    # print(util.to_pileup(enc1.unsqueeze(1)))
    # enc2 = re.shift(27717810, 27717830)
    # print(''.join(["."] * 10) + util.to_pileup(enc2.unsqueeze(1)))


    # encs = []
    # for i, r in enumerate(fetcher):
    #     re = ReadEncoder(r)
    #     enc = re.get_encoded(ws, we)
    #     encs.append(enc)
    #     if i > 100:
    #         break
    #
    #
    # t = torch.stack(tuple(encs), dim=1)

    # p = util.to_pileup(t)
    # print(p)


