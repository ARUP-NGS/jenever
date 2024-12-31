
import random
import traceback
import numpy as np
import pysam
import torch
import logging
from collections import defaultdict

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
            self.cache[key] = (alnstart(read), encode_read(read))

        return self.cache[key][1]

    def clear_to_pos(self, min_pos):
        """
        Remove any items from the cache that have a alnstart of less than min_pos
        """
        newcache = {}
        for key, val in self.cache.items():
            if val[0] >= min_pos:
                newcache[key] = val
        self.cache = newcache

    def __contains__(self, item):
        if type(item) == str:
            return item in self.cache
        elif type(item) == pysam.AlignedSegment:
            return item.query_name in self.cache
        else:
            return False


class ReadWindow:

    def __init__(self, aln, chrom, start, end, min_mq=-1):
        self.aln = aln
        self.start = start
        self.end = end
        self.margin_size = 150 # Should be about a read length
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
        allreads = []
        for pos in range(start - self.margin_size, end):
            for read in self.bypos[pos]:
                if pos > end or (pos + read.query_length) < start: # Check to make sure read overlaps window
                    continue
                allreads.append((pos, read))
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
            encoded = self.cache[read].char() # Char is the same as int8
            enc_start_offset = max(0,  start - readstart)
            enc_end_offset = min(encoded.shape[0], window_size - (readstart - start))
            t_start_offset = max(0, readstart - start)
            t_end_offset = t_start_offset + (enc_end_offset - enc_start_offset)
            t[t_start_offset:t_end_offset, i, :] = encoded[enc_start_offset:enc_end_offset]

        return t


def encode_read(read, prepad=0, tot_length=None):
    """
    Encode the given read into a tensor
    :param read: Read to be encoded (typically pysam.AlignedSegment)
    :param prepad: Leading zeros to prepend
    :param tot_length: If not None, desired total 'length' (dimension 0) of tensor
    """
    if tot_length:
        assert prepad < tot_length, f"Cant have more padding than total length"
    bases = []
    for i in range(prepad):
        bases.append(torch.zeros(FEATURE_NUM))

    try:
        for t in iterate_bases(read):
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

def base_onehot(base):
    t = torch.zeros(4, dtype=torch.int8)
    update_from_base(base, t)
    return t

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

def seq_to_onehot(seq):
    return torch.concat([base_onehot(b) for b in seq], dim=0)

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


def iterate_bases(rec):
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
    for i, (base, qual) in enumerate(zip(bases, quals)):
        readpos = i/150 if not rec.is_reverse else 1.0 - i/150
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
    readwindow = ReadWindow(bam, chrom, start, end)
    for i in range(num_samples):
        reads_to_sample = maxreads
        if np.random.rand() < downsample_frac:
            reads_to_sample = maxreads // 2
        reads_encoded = readwindow.get_window(start, end, max_reads=maxreads, downsample_read_count=reads_to_sample)
        refseq = refgenome.fetch(chrom, start, end)
        ref_encoded = string_to_tensor(refseq)
        encoded_with_ref = torch.cat((ref_encoded.unsqueeze(1), reads_encoded), dim=1)[:, 0:maxreads, :]

        yield encoded_with_ref, (start, end)
