
from itertools import product, permutations
from skbio.alignment import StripedSmithWaterman
# from skbio.alignment import pair_align, PairAlignPath
import logging
import pysam
from intervaltree import IntervalTree
from collections import defaultdict

from dnaseq2seq.training import loader

logger = logging.getLogger(__name__)


class Haplotype(object):

    def __init__(self, score, seq, allele_indices, raw_read_support):
        self.score = score
        self.seq = seq
        self.allele_indices = allele_indices
        self.raw_read_support = raw_read_support

    def __eq__(self, other):
        return self.score == other.score and self.seq == other.seq and self.allele_indices == other.allele_indices

    def __hash__(self):
        return sum([hash(self.score), hash(self.seq), hash(self.allele_indices)])

    def __str__(self):
        return "{} : {} : {}".format(self.allele_indices, self.score, self.seq[0:min(6, len(self.seq))])

    def all_ref(self):
        """
        Return true this haplotype represents all reference alleles
        """
        return all(i==0 for i in self.allele_indices)


class Genotype(object):

    def __init__(self, haplotypes):
        self.haplotypes = haplotypes

    def __eq__(self, other):
        if not len(self.haplotypes) == len(other.haplotypes):
            return False
        for a,b in zip(self.haplotypes, other.haplotypes):
            if not a==b:
                return False
        return True

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __hash__(self):
        return sum(hash(h) for h in self.haplotypes)

    def __str__(self):
        return " / ".join(str(h.allele_indices) for h in self.haplotypes)

    @property
    def score(self):
        return sum(h.score for h in self.haplotypes)

    @property
    def raw_read_support(self):
        return sum(h.raw_read_support for h in self.haplotypes)

    def ref_haplotype(self):
        """
        Return the first haplotype that is all reference alleles, or None if no haplotype is all ref
        """
        for hap in self.haplotypes:
            if hap.all_ref():
                return hap
        return None


def trim_common_prefix(a,b):
    """
    Remove the longest string that starts both a and b and return its length and the trimmed a and b
    :return: Tuple of prefix length, trimmed a, trimmed b
    """
    if min(len(a), len(b))==0:
        return 0,a,b
    for idx in range(min(len(a), len(b))):
        if a[idx] != b[idx]:
            return idx, a[idx:], b[idx:]
    idx += 1
    return idx, a[idx:], b[idx:]


def trim_common_suffix(a,b):
    """
    Trim the longest string that appears at the end of both a and b,
    :return: Tuple of number of bases trimmed, and a and b with the sequence removed
    """
    idx = -1
    for idx in range(1, min(len(a), len(b))):
        if a[len(a)-idx] != b[len(b)-idx]:
            return idx-1, a[0:len(a)-idx+1], b[0:len(b)-idx+1]
    return idx+1, a[0:len(a)-idx-1], b[0:len(b)-idx-1]


def construct_haplotype(variants, allele_indexes, ref_sequence, ref_offset):
    """
    Given a list of variants and the allele indexes to use, create a new Haplotype object with the variants inserted

    :param variants: List of pysam VCF records
    :param allele_indexes: List of integers describing which allele to use (can include 0 which indicates reference)
    :param ref_sequence: Reference fasta in neighborhood of variants
    :param ref_offset: 0-indexed genomic position at which ref_sequence begins
    :return: Sequence containing given alleles from variants
    """
    displacement = 0
    seq = ref_sequence
    for var, allele_index in zip(variants, allele_indexes):
        trim, vref, valt = trim_common_prefix(var.ref, var.alleles[allele_index])
        vref_start = var.start - ref_offset + displacement + trim  # Start is 0-indexed, pos is 1-indexed
        vref_end = vref_start + len(vref)
        seq = seq[0:vref_start] + valt + seq[vref_end:]
        displacement += len(valt) - len(vref)
    return Haplotype(score=-1.0, seq=seq, allele_indices=allele_indexes, raw_read_support=-1)


def gen_genotype_combos(variants):
    """
    Construct all possible combinations of allele indices into genotypes. For instance, say there are two

    :param variants:
    :return: List of list of possible allele combinations, like this:
     [
       [(0,1), (1,0)],  # trans configuration
       [(0,0), (1,1)],  # cis configuration
     ]
     ..for instance, if there are two input variants, both het

     TODO: Test on multiallelic variants
    """
    gts = [permutations(v.samples[0]['GT']) for v in variants] # List of tuples of genotypes
    combos = []
    gtset = set()
    for combo in product(*gts):
        flt_combo = []
        for k in combo:
            flt_combo.append(tuple(j if j is not None else 0 for j in k)) # Convert half-calls to hets?
        gt = sorted([k for k in zip(*flt_combo)])

        sgt = str(gt)
        if sgt in gtset:
            continue

        gtset.add(sgt)
        combos.append(gt)

    return combos



def hap_is_possible(hap, variants):
    """
    Examine alleles in haplotype and determine if any two alleles occupy the same reference bases
    If so, then return False
    :param hap: Haplotype
    :param variants: Variants included in haplotype
    :return: True if there are no incompatibilities
    """
    intervals = IntervalTree()
    insertionpoints = set()
    for allele_index, var in zip(hap, variants):

        # This is probably not technically correct, but we often assume that a 0/1 genotype doesn't actually
        # assert referenceness very strongly on the non-alt haplotype. For instance, we egregiously convert
        # half-calls and somatic calls to 0/1 regularly and just assume the '0' part doesn't matter
        if allele_index == 0:
            continue

        ref = var.ref
        alt = var.alleles[allele_index]
        _, trimref, trimalt = trim_common_suffix(ref, alt)
        trimlen, trimref, trimalt = trim_common_prefix(trimref, trimalt)
        start = var.start + trimlen
        end = start + len(trimref)

        # We handle insertions differently than deletions and SNPs, since
        # insertions dont occupy any reference bases. For insertions, we just track the 'insertion point',
        # which is the base after which the newly inserted bases appear. For SNPs and dels, we store the range
        # of bases they occupy
        if end == start:
            if start in insertionpoints:
                return False
            insertionpoints.add(start)
        else:
            overlaps = intervals[start:end]
            if overlaps:
                return False
            else:
                intervals[start:end] = var

    return True


def eliminate_incompatible_haps(genotypes, variants):
    """
    The haplotypes generated by gen_genotypes_somatic and gen_genotype_combos are a list of all potential
    haplotypes generated by the variants (given their genotypes), but it's still possible for some variant (allele, really)
    combinations to be incompatible, for instance if two alleles have overlapping reference positions.
    I think the correct course here is to eliminate any genotype that contains any haplotypes which contain incompatible
    variants (we can't just eliminate the haplotype since for germline we need two haplotypes per genotype)
    :param genotypes: List of genotype combos (tuples of allele indices), as generate by gen_genotype_combos or gen_genotype_somatic
    :param variants: List of variants associated with genotypes
    :return: Subset of input genotypes not including genotypes with 'incompatible' haplotypes
    """
    ok_genotypes = []
    for geno in genotypes:
        if all(hap_is_possible(hap.allele_indices, variants) for hap in geno.haplotypes):
            ok_genotypes.append(geno)
        else:
            logging.debug("Eliminating genotype {} since it contains at least one impossible haplotype near position {}:{}".format(geno, variants[0].chrom, variants[0].pos))
    return ok_genotypes


def gen_seqs_multi(ref_sequence, region_start, variants):
    """
    Create all haplotype sequences from the variants and the reference
    :param ref: Reference fasta
    :param variants: List of pysam VCF records
    :return: List of tuples, each item is (haplotype sequence, allele index 0, allele index 1, etc etc)
    """
    # NEIGHBORHOOD_SIZE = 100  # Additional reference bases before and after variants to include, should be about read length

    for var in variants[1:]:
        if var.chrom != variants[0].chrom:
            raise ValueError("Variants are on different chroms")

    if not variants == sorted(variants, key=lambda v: v.start):
        raise ValueError('Input variants must be sorted by start coordinate')

    # We want to create all possible haplotypes given the variants. If both are hets and unphased, this means 4
    # (ref, ref), (ref, alt), (alt, ref), and (alt, alt)

    # Create list of tuples, each list item is (constructed haplotype, var allele index 0, var allele index 1, ...)

    geno_gen = gen_genotype_combos(variants)

    all_genos = []
    for gt_indexes in geno_gen:
        haps = [construct_haplotype(variants, idx, ref_sequence, region_start) for idx in gt_indexes]
        all_genos.append(Genotype(haplotypes=haps))

    # Remove any genotypes that contain 'incompatible' haplotypes (those that have two overlapping alleles)
    result_genos = eliminate_incompatible_haps(all_genos, variants)

    return result_genos


def score_genotypes(aln, ref_sequence, region_start, variants):
    """
    Construct all of the haplotypes that the called variants suggest exist, the fetch reads from the alignment
    and align them all to each haplotype, keeping track of the highest scoring haplotype for each read. Return
    the genotypes (compatible haplotype configurations) align with their scores
    :param aln: pysam alignment file object
    :param ref: pysam Fasta reference file
    :param variants: List of pysam variant records
    :return: List of genotypes with scores, each genotype contains list of haplotypes (with scores)
    """
    ploidy = len(variants[0].samples[0]['GT'])
    if any(len(v.samples[0]['GT']) != ploidy for v in variants):
        raise ValueError("All variants must have the same ploidy to generate genotypes, sorry")

    genotypes = gen_seqs_multi(ref_sequence, region_start, variants)
    read_support = [
        0 for gt in genotypes
        for _ in range(len(gt.haplotypes))
    ]
    ssws = [
        [StripedSmithWaterman(hap.seq) for hap in gts.haplotypes]
        # [pair_align(hap.seq, ref_sequence, mode='local', sub_score=(1, -2), gap_cost=(4, 0.2), free_ends=True, trim_ends=True) for hap in gts.haplotypes]
        for gts in genotypes
    ]

    ref_start = min(v.start for v in variants)
    ref_end = max(v.start + len(v.ref) for v in variants)
    # Fetch each overlapping read and align it to all possible haplotypes and keep track of the highest-scoring one
    try:
        read_iterator = aln.fetch(variants[0].chrom, ref_start, ref_end)
    except KeyError:
        read_iterator = aln.fetch(variants[0].chrom.replace("chr", ""), ref_start, ref_end)

    for read in read_iterator:
        #print(f"read start: {read.reference_start} end: {read.reference_end}")
        # Skip reads that don't overlap the whole region of interest
        if read.reference_start is None \
                or read.reference_end is None \
                or read.reference_start > ref_start \
                or read.reference_end < ref_end:
            continue

        # readscores is a list of the alignment score for every haplotype across all genotypes
        readscores = [hap_ssw(read.seq).optimal_alignment_score
                      for gt_scores, genotype_aligners in zip(read_support, ssws)
                      for hap_ssw in genotype_aligners]

        maxscore = max(readscores)
        best_indices = [i for i, score in enumerate(readscores) if score == maxscore]
        # best = readscores.index( max(readscores))
        for index in best_indices:
            read_support[index] += 1.0 / len(best_indices)

    # Tabulate and normalize scores
    scoresum = sum(read_support)
    if scoresum == 0:
        return []
    scores = [s/scoresum for s in read_support]

    # Assign scores to haplotypes
    # Pretty fragile - assumes this will iterate over haplotypes in same order as alignment function above,
    # which isn't guaranteed
    for i, gt in zip(range(0, len(scores), ploidy), genotypes):
        for j, hap in zip(range(i, i+ploidy), gt.haplotypes):
            hap.score = scores[j]
            hap.raw_read_support = read_support[j]

    return sorted(genotypes, key=lambda g: -g.score)


def project_vars(variants, allele_indexes, ref_sequence, ref_offset):
    """
    Given a list of variants and the allele indexes to use, create a new Haplotype object with the variants inserted

    :param variants: List of pysam VCF records
    :param allele_indexes: List of integers describing which allele to use (can include 0 which indicates reference)
    :param ref_sequence: Reference fasta in neighborhood of variants
    :param ref_offset: 0-indexed genomic position at which ref_sequence begins
    :return: Sequence containing given alleles from variants
    """
    displacement = 0
    seq = ref_sequence
    for var, allele_index in zip(variants, allele_indexes):
        trim, vref, valt = trim_common_prefix(var.ref, var.alleles[allele_index])
        vref_start = var.start - ref_offset + displacement + trim  # Start is 0-indexed, pos is 1-indexed
        assert vref_start >= 0, f"Negative start coord for variant {var}!"
        vref_end = vref_start + len(vref)
        seq = seq[0:vref_start] + valt + seq[vref_end:]
        displacement += len(valt) - len(vref)
    return seq


def gen_haplotypes(bam, ref_sequence, chrom, region_start, region_end, variants):
    """
    Generate the two most likely haplotypes given the variants and reads for the region
    In many cases this is trivial and we don't need to look at the reads, but if there are
    multiple heterozygous variants we need to get fancy and look at which potential haplotypes
    are actually best supported by the reads

    : returns: Two strings representing the most probable haplotypes in the region
    """
    hap0 = ref_sequence
    hap1 = ref_sequence
    if len(variants) == 0:
        return hap0, hap1

    het_count = sum(1 if len(set(v.samples[0]['GT']))>1 else 0 for v in variants )

    if het_count == 0:  # Every variant is homozygous
        hap_alt = project_vars(variants,
                               [1] * len(variants),
                               ref_sequence,
                               region_start)
        return hap_alt, hap_alt

    elif het_count == 1 and len(variants) == 1:
        hap0 = project_vars(variants,
                               [variants[0].samples[0]['GT'][0]],
                               ref_sequence,
                               region_start)
        hap1 = project_vars(variants,
                               [variants[0].samples[0]['GT'][1]],
                               ref_sequence,
                               region_start)
        return hap0, hap1

    elif het_count == 1 and len(variants) > 1:
        # All homs except 1 het
        hap0_indices = [v.samples[0]['GT'][0] for v in variants]
        hap1_indices = [v.samples[0]['GT'][1] for v in variants]

        hap0 = project_vars(variants,
                            hap0_indices,
                            ref_sequence,
                            region_start)
        hap1 = project_vars(variants,
                            hap1_indices,
                            ref_sequence,
                            region_start)
        return hap0, hap1

    else:
        ref_start = min(v.start for v in variants if len(set(v.samples[0]['GT']))>1)
        ref_end = max(v.start + len(v.ref) for v in variants if len(set(v.samples[0]['GT']))>1)
        if ref_end - ref_start > 100:
            raise ValueError(f"Variants are too far apart to phase, skipping {chrom}:{region_start}-{region_end}")
        genotypes = score_genotypes(bam, ref_sequence, region_start, variants)
        best_genotype = genotypes[0] # They come back ordered by score
        return best_genotype.haplotypes[0].seq, best_genotype.haplotypes[1].seq


def parse_rows_classes(bed):
    """
    Iterate over every row in rows and create a list of class indexes for each element
    , where class index is currently row.vtype-row.status. So a class will be something like
    snv-TP or small_del-FP, and each class gets a unique number
    :returns : 0. List of chrom / start / end tuples from the input BED
               1. List of class indexes across rows.
               2. A dictionary keyed by class index and valued by class names.
    """
    classcount = 0
    classes = defaultdict(int)
    idxs = []
    rows = []
    with open(bed) as fh:
        for line in fh:
            row = line.strip().split("\t")
            chrom = row[0]
            start = int(row[1])
            end = int(row[2])
            clz = row[3]
            rows.append((chrom, start, end))
            if clz in classes:
                idx = classes[clz]
            else:
                idx = classcount
                classcount += 1
                classes[clz] = idx
            idxs.append(idx)
    class_names = {v: k for k, v in classes.items()}
    return rows, idxs, class_names



