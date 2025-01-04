
import numpy as np
from dataclasses import dataclass
import logging
import pysam

from skbio.alignment import StripedSmithWaterman
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    """
    Represents a variant found in a single calling window. These are 0-based, have no reference padding, and
    model a single alt allele
    """
    chrom: str
    ref: str
    alt: str
    pos: int
    qual: float
    hap_model: int = None
    step: int = None
    window_offset: int = None
    var_index: int = None
    var_count: int = None
    aln_score: int = None
    tnpred: float = None

    def __str__(self):
        a = self.alt
        if len(a) > 20:
            a = self.alt[:20] + "..."
        return f"Var {self.chrom}:{self.pos} {self.ref}->{a}"

    def __eq__(self, other):
        return self.chrom == other.chrom and self.ref == other.ref and self.alt == other.alt and self.pos == other.pos

    def __hash__(self):
        return hash(f"{self.chrom}&{self.ref}&{self.alt}&{self.pos}&{self.qual}")

    def __gt__(self, other):
        return self.pos > other.pos

    @property
    def end(self):
        return self.pos + len(self.ref)

    @property
    def key(self):
        return self.chrom, self.pos, self.ref, self.alt


@dataclass
class Cigar:
    op: str
    len: int


@dataclass
class VcfVar:
    """
    A variant found across multiple calling windows.
    These are VCF-style, with 1-based coords and reference padding on indels, and a list of alts
    """
    chrom: str
    pos: int
    ref: str
    quals: list
    qual: float
    filter: list
    depth: int
    phased: bool
    phase_set: int
    haplotype: int
    window_var_count: int
    window_cis_vars: int
    window_trans_vars: int
    window_offset: List[int]
    var_index: int
    call_count: int
    step_count: int
    genotype: tuple
    het: bool
    duplicate: bool
    alts: list
    tnpred: List[float]

    @property
    def alt(self):
        return self.alts[0]

    def __hash__(self):
        return hash("{self.chrom}&{self.pos}&{self.ref}&{self.alts}")



def _cigtups(cigstr):
    """
    Generator for Cigar objects from a cigar string
    :param cig: cigar string
    :return: Generator of Cigar objects
    """
    digits = []
    for c in cigstr:
        if c.isdigit():
            digits.append(c)
        else:
            cig = Cigar(op=c, len=int("".join(digits)))
            digits = []
            yield cig


def _geomean(probs):
    """
    Geometric mean of numbers
    """
    return np.exp(np.log(probs).mean())


def align_sequences(query, target, gap_open_penalty=3, gap_extend_penalty=1, match_score=2, mismatch_score=-1):
    """
    Return Smith-Watterman alignment of both sequences
    """
    # TODO SSW in skbio is deprecated (see https://github.com/scikit-bio/scikit-bio/issues/1814)
    ssw = StripedSmithWaterman(query,
                               gap_open_penalty=gap_open_penalty,
                               gap_extend_penalty=gap_extend_penalty,
                               match_score=match_score,
                               mismatch_score=mismatch_score)
    return ssw(target)


def _mismatches_to_vars(query, target, chrom, cig_offset, window_offset, probs):
    """
    Zip both sequences and look for mismatches, if any are found convert them to Variant objects
    and return them
    This is for finding variants that are inside an "Match" region according to the cigar from an alignment result
    :returns: Generator over Variants from the paired sequences
    """
    mismatches = []
    mismatch_quals = []
    mismatchstart = None
    for i, (a, b) in enumerate(zip(query, target)):
        if a == b:
            if mismatches:
                yield Variant(chrom=chrom,
                              ref="".join(mismatches[0]).replace("-", ""),
                              alt="".join(mismatches[1]).replace("-", ""),
                              pos=mismatchstart,
                              qual=_geomean(mismatch_quals),  # Geometric mean?
                              window_offset=mismatchstart - cig_offset + window_offset,
                              )
            mismatches = []
            mismatch_quals = []
        else:
            if mismatches:
                mismatches[0] += a
                mismatches[1] += b
                mismatch_quals.append(probs[i])
            else:
                mismatches = [a, b]
                mismatch_quals = [probs[i]]
                mismatchstart = i + cig_offset

    # Could be mismatches at the end
    if mismatches:
        yield Variant(chrom=chrom,
                      ref="".join(mismatches[0]).replace("-", ""),
                      alt="".join(mismatches[1]).replace("-", ""),
                      pos=mismatchstart,
                      qual=_geomean(mismatch_quals),
                      window_offset=mismatchstart - cig_offset + window_offset)


def _display_aln(aln):
    """
    Utility function for printing an alignment
    """
    qit = iter(aln.query_sequence)
    tit = iter(aln.target_sequence)
    tbases = 0
    for cig in _cigtups(aln.cigar):
        if cig.op == "M":
            for _ in range(cig.len):
                q = next(qit)
                t = next(tit)
                if q == t:
                    print(f"{tbases}\tM   {q} = {t}")
                else:
                    print(f"{tbases}\tM   {q} ! {t}")
                tbases += 1
        elif cig.op == "I":
            for _ in range(cig.len):
                q = next(qit)
                t = "-"
                print(f"{tbases}\tI  {q}   {t}")
        elif cig.op == "D":
            for _ in range(cig.len):
                q = "-"
                t = next(tit)
                print(f"{tbases}   D  {q}   {t}")
                tbases += 1
        else:
            raise ValueError(f"Unknown cigar op {cig.op}")


def aln_to_vars(refseq, altseq, chrom, offset=0, probs=None):
    """
    Smith-Watterman align the given sequences and return a generator over Variant objects
    that describe differences between the sequences
    :param refseq: String of bases representing reference sequence
    :param altseq: String of bases representing alt sequence
    :param offset: This amount will be added to each variant position
    :return: Generator over variants
    """
    num_vars = 0
    if probs is not None:
        assert len(probs) == len(altseq), f"Probabilities must contain same number of elements as alt sequence"
    else:
        probs = np.ones(len(altseq))
    aln = align_sequences(altseq, refseq, gap_open_penalty=4, gap_extend_penalty=0.2, match_score=1, mismatch_score=-1)
    ref_seq_consumed = 0
    q_offset = 0
    t_offset = 0

    variant_pos_offset = 0
    if aln.query_begin > 0:
        q_offset += aln.query_begin # Maybe we don't want this? query is alt sequence, so nonzero indicates first alt base matches downstream of first ref base
    if aln.target_begin > 0:
        ref_seq_consumed += aln.target_begin
        t_offset += aln.target_begin

    variants = []
    for cig in _cigtups(aln.cigar):
        if cig.op == "M":
            for v in _mismatches_to_vars(
                        refseq[t_offset:t_offset+cig.len],
                        altseq[q_offset:q_offset+cig.len],
                        chrom,
                        offset + t_offset,
                        t_offset,
                        probs[q_offset:q_offset+cig.len]
            ):
                v.var_index = num_vars
                variants.append(v)
                num_vars += 1
            q_offset += cig.len
            variant_pos_offset += cig.len
            t_offset += cig.len

        elif cig.op == "I":
            variants.append(
                Variant(
                        chrom=chrom,
                        ref='',
                        alt=altseq[q_offset:q_offset+cig.len],
                        pos=offset + variant_pos_offset + aln.target_begin,
                        qual=_geomean(probs[q_offset:q_offset+cig.len]),
                        window_offset=variant_pos_offset,
                        var_index=num_vars)
                )
            num_vars += 1
            q_offset += cig.len
            # variant_pos_offset += cig.len

        elif cig.op == "D":
            variants.append(
                Variant(chrom=chrom,
                        ref=refseq[t_offset:t_offset + cig.len],
                        alt='',
                        pos=offset + t_offset,
                        qual=_geomean(probs[q_offset-1:q_offset+cig.len]),
                        window_offset=t_offset,
                        var_index=num_vars)
                )
            num_vars += 1
            t_offset += cig.len
            variant_pos_offset += cig.len

    for v in variants:
        v.aln_score = aln.optimal_alignment_score
        v.var_count = len(variants)

    return variants


def var_depth(chrom, pos, aln):
    """
    Get read depth at variant start position from bam  pysam AlignmentFile to get depth at
    :param var: local variant tuple just (pos, ref, alt)
    :param chrom:
    :param aln: bam pysam AlignmentFile
    :return:
    """
    # get bam depths for now, not same as tensor depth
    count = aln.count(
         contig=chrom,
         start=pos,  # pos of var
         stop=pos + 1,  # pos - 1
         read_callback="all",
    )
    return count


def construct_vcfvars(vars_hap0, vars_hap1, aln, reference, mindepth=30):
    """
    Aggregate sets of Variant objects into VcfVar objects.
     This calculates most of the fields that are emitted in the final pysam.VariantRecord objects and used for
     the classifier model calculation

    :param vars_hap0: Dict of (chrom, pos, ref, alt) -> [Variants] on hap 0
    :param vars_hap1: Dict of (chrom, pos, ref, alt) -> [Variants] on hap 1
    :param aln: bam AlignmentFile (for depth calculation)
    :param mindepth: read depth cut off in bam to be labled as LowCov in filter field
    :return: single dict of vars for call window
    """

    # index vars by (pos, ref, alt) to make dealing with homozygous easier
    vcfvars_hap0 = {}
    allkeys = list(set(list(vars_hap0.keys()) + list(vars_hap1.keys())))
    phase_set = min(v[1] for v in allkeys) if len(allkeys) > 1 else None
    for var in vars_hap0.keys():
        chrom, pos, ref, alt = var
        depth = var_depth(chrom, pos, aln)
        vcfvars_hap0[var] = VcfVar(
            chrom=chrom,
            pos=pos + 1,
            ref=ref,
            alts=[alt],
            quals=[call.qual for call in vars_hap0[var]],
            qual=float(np.mean([call.qual for call in vars_hap0[var]])),
            filter=[],
            depth=depth,
            phased=False,  # default for now
            phase_set=phase_set,  # default to first var in window for now
            haplotype=0,  # defalt vars_hap0 to haplotype 0 for now
            window_var_count=len(set(vars_hap0.keys()) | set(vars_hap1.keys())),
            window_cis_vars=min(v.var_count for v in vars_hap0[var]),
            window_trans_vars=len(vars_hap1),
            call_count=len(vars_hap0[var]),
            step_count=len(vars_hap0[var]),
            genotype=(0, 1),  # default to haplotype 0
            het=True,  # default to het but check later
            duplicate=False,  # initialize but check later
            window_offset=[call.window_offset for call in vars_hap0[var]],
            var_index=[call.var_index for call in vars_hap0[var]],
            tnpred=[call.tnpred for call in vars_hap0[var]],
        )

    vcfvars_hap1 = {}
    for var in vars_hap1.keys():
        chrom, pos, ref, alt = var
        logger.debug(f"Computing {var}")
        depth = var_depth(chrom, pos, aln)
        vcfvars_hap1[var] = VcfVar(
            chrom=chrom,
            pos=pos + 1,
            ref=ref,
            alts=[alt],
            quals=[call.qual for call in vars_hap1[var]],
            qual=float(np.mean([call.qual for call in vars_hap1[var]])),
            filter=[],
            depth=depth,
            phased=False,  # default value for now
            phase_set=phase_set,  # default to first var position in window
            haplotype=1,  # defalt vars_hap1 to haplotype 1 for now
            window_var_count=len(set(vars_hap0.keys()) | set(vars_hap1.keys())),
            window_cis_vars=min(v.var_count for v in vars_hap1[var]),
            window_trans_vars=len(vars_hap0),
            call_count=len(vars_hap1[var]),
            step_count=len(vars_hap1[var]),
            genotype=(1, 0),  # default to haplotype 1
            het=True,  # default to het but check later
            duplicate=False,  # initialize but check later
            window_offset=[call.window_offset for call in vars_hap1[var]],
            var_index=[call.var_index for call in vars_hap1[var]],
            tnpred=[call.tnpred for call in vars_hap1[var]],
        )

    # check for homozygous vars
    homs = list(set(vcfvars_hap0.keys()) & set(vcfvars_hap1.keys()))
    for var in homs:
        # modify hap0 var info
        vcfvars_hap0[var].qual = (vcfvars_hap0[var].qual + vcfvars_hap1[var].qual) / 2  # Mean of each call??
        vcfvars_hap0[var].quals = vcfvars_hap0[var].quals + vcfvars_hap1[var].quals  # combine haps
        vcfvars_hap0[var].window_cis_vars = min(vcfvars_hap0[var].window_cis_vars, vcfvars_hap1[var].window_cis_vars)  # cis and trans are now cis
        vcfvars_hap0[var].window_trans_vars = 0  # moved all vars to cis
        vcfvars_hap0[var].call_count += vcfvars_hap1[var].call_count  # combine call counts in both haplotypes
        vcfvars_hap0[var].genotype = (1, 1)
        vcfvars_hap0[var].het = False
        vcfvars_hap0[var].window_offset = sorted(set(vcfvars_hap0[var].window_offset + vcfvars_hap1[var].window_offset))
        vcfvars_hap0[var].tnpred =  [0] #np.mean(x for x in vcfvars_hap0[var].tnpred + vcfvars_hap1[var].tnpred)
        # then remove from hap1 vars
        vcfvars_hap1.pop(var)

    # combine haplotypes
    assert len(set(vcfvars_hap0) & set(vcfvars_hap1)) == 0, (
        "There should be no vars shared between vcfvars_hap0 and vcfvars_hap1 (hom vars should all be in hap0)"
    )

    vcfvars = {**vcfvars_hap0, **vcfvars_hap1}

    for key, var in vcfvars.items():
        # adjust filters
        if var.depth < mindepth:
            var.filter.append("LowCov")
        if var.call_count < 2:
            var.filter.append("SingleCallHet")
        elif var.step_count < 2:
            var.filter.append("SingleCallHom")

        # adjust insertions and deletions so no blank ref or alt
        if var.ref == "" or any(v == "" for v in var.alts):
            leading_ref_base = reference.fetch(reference=var.chrom, start=var.pos - 2, end=var.pos - 1)
            var.pos = var.pos - 1
            var.ref = leading_ref_base + var.ref
            newalts = [leading_ref_base + a for a in var.alts]
            var.alts = newalts


    # go back and set phased to true if more than one het variant remains
    het_vcfvars = [k for k, v in vcfvars.items() if v.het]
    if len(het_vcfvars) > 1:
        for var in het_vcfvars:
            vcfvars[var].phased = True

    return vcfvars.values()


def create_vcf_header(sample_name="sample", lowcov=30, cmdline=None):
    """
    Initialize pysam VariantFile vcf object and create it's header
    :param path: vcf file path
    :param sample_name: sample name for FORMAT field
    :param lowcov: info for FILTER field "LowCov" header entry
    :return: VariantFile object
    """

    # Create a VCF header
    vcfh = pysam.VariantHeader()
    if cmdline is not None:
        vcfh.add_meta('COMMAND', cmdline)
    # Add a sample named "sample"
    vcfh.add_sample(sample_name)
    # Add contigs
    vcfh.add_meta('contig', items=[('ID', 1)])
    vcfh.add_meta('contig', items=[('ID', 2)])
    vcfh.add_meta('contig', items=[('ID', 3)])
    vcfh.add_meta('contig', items=[('ID', 4)])
    vcfh.add_meta('contig', items=[('ID', 5)])
    vcfh.add_meta('contig', items=[('ID', 6)])
    vcfh.add_meta('contig', items=[('ID', 7)])
    vcfh.add_meta('contig', items=[('ID', 8)])
    vcfh.add_meta('contig', items=[('ID', 9)])
    vcfh.add_meta('contig', items=[('ID', 10)])
    vcfh.add_meta('contig', items=[('ID', 11)])
    vcfh.add_meta('contig', items=[('ID', 12)])
    vcfh.add_meta('contig', items=[('ID', 13)])
    vcfh.add_meta('contig', items=[('ID', 14)])
    vcfh.add_meta('contig', items=[('ID', 15)])
    vcfh.add_meta('contig', items=[('ID', 16)])
    vcfh.add_meta('contig', items=[('ID', 17)])
    vcfh.add_meta('contig', items=[('ID', 18)])
    vcfh.add_meta('contig', items=[('ID', 19)])
    vcfh.add_meta('contig', items=[('ID', 20)])
    vcfh.add_meta('contig', items=[('ID', 21)])
    vcfh.add_meta('contig', items=[('ID', 22)])
    # FILTER values other than "PASS"
    vcfh.add_meta('FILTER', items=[('ID', "LowCov"),
                                   ('Description', f'cov depth at var start pos < {lowcov}')])
    vcfh.add_meta('FILTER', items=[('ID', "SingleCallHet"),
                                   ('Description', f'var had single het call in the multi-step detection')])
    vcfh.add_meta('FILTER', items=[('ID', "SingleCallHom"),
                                   ('Description', f'var had single hom call in the multi-step detection')])
    # FORMAT items
    vcfh.add_meta('FORMAT', items=[('ID', "GT"), ('Number', 1), ('Type', 'String'),
                                   ('Description', 'Genotype')])
    vcfh.add_meta('FORMAT', items=[('ID', "DP"), ('Number', 1), ('Type', 'Integer'),
                                   ('Description', 'Bam depth at variant start position')])
    vcfh.add_meta('FORMAT', items=[('ID', "PS"), ('Number', "."), ('Type', 'Integer'),
                                   ('Description', 'Phase set equal to POS of first record in phased set')])
    # INFO items
    vcfh.add_meta('INFO', items=[('ID', "WIN_VAR_COUNT"), ('Number', "."), ('Type', 'Integer'),
                                 ('Description', 'Total variants called in same window(s)')])
    vcfh.add_meta('INFO', items=[('ID', "WIN_CIS_COUNT"), ('Number', "."), ('Type', 'Integer'),
                                 ('Description', 'Total cis variants called in same window(s)')])
    vcfh.add_meta('INFO', items=[('ID', "WIN_TRANS_COUNT"), ('Number', "."), ('Type', 'Integer'),
                                 ('Description', 'Total cis variants called in same window(s)')])
    vcfh.add_meta('INFO', items=[('ID', "QUALS"), ('Number', "."), ('Type', 'Float'),
                                 ('Description', 'QUAL value(s) for calls in window(s)')])
    vcfh.add_meta('INFO', items=[('ID', "CALL_COUNT"), ('Number', "."), ('Type', 'Integer'),
                                 ('Description', 'Number of model haplotype calls of this var in multi-step detection')])
    vcfh.add_meta('INFO', items=[('ID', "STEP_COUNT"), ('Number', "."), ('Type', 'Integer'),
                                 ('Description', 'Number of overlapping steps where var detected in multi-step detection')])
    vcfh.add_meta('INFO', items=[('ID', "DUPLICATE"), ('Number', 1), ('Type', 'String'),
                                 ('Description', 'Duplicate of call made in previous window')])
    vcfh.add_meta('INFO', items=[('ID', "WIN_OFFSETS"), ('Number', "."), ('Type', 'Integer'),
                                 ('Description', 'Position of call within calling window')])
    vcfh.add_meta('INFO', items=[('ID', "VAR_INDEX"), ('Number', "."), ('Type', 'Integer'),
                                 ('Description', 'Order of call in window')])
    vcfh.add_meta('INFO', items=[('ID', "RAW_QUAL"), ('Number', 1), ('Type', 'Float'),
                                 ('Description', 'Original quality if classifier used to update QUAL field')])
    vcfh.add_meta('INFO', items=[('ID', "TNPRED"), ('Number', "."), ('Type', 'Float'),
                                 ('Description', 'Predicted probability of being a true negative')])
    # write to new vcf file object
    return vcfh


def prob_to_phred(p, max_qual=1000.0):
    """
    Convert a probability in 0..1 to phred scale
    """
    if np.isnan(p):
        logger.warning(f"Found NaN probability for variant, returning quality 0")
        return 0.0
    assert 0.0 <= p <= 1.0, f"Probability must be in [0, 1], but found {p}"
    if p == 1.0:
        return max_qual
    else:
        return min(max_qual, -10 * np.log10(1.0 - p))


def create_vcf_rec(var, vcf_file):
    """
    create single variant record from pandas row
    :param var:
    :param vcf_file:
    :return:
    """
    # Create record
    vcf_filter = var.filter if var.filter else "PASS"
    if np.isnan(var.qual):
        logger.error(f"Quality is NaN for variant {var}")
    r = vcf_file.new_record(contig=var.chrom, start=var.pos - 1, stop=var.pos - 1 + len(var.ref), 
                       alleles=(var.ref, *var.alts), filter=vcf_filter, qual=int(round(prob_to_phred(var.qual))))
    # Set FORMAT values
    r.samples['sample']['GT'] = var.genotype
    r.samples['sample'].phased = var.phased  # note: need to set phased after setting genotype
    r.samples['sample']['DP'] = var.depth
    r.samples['sample']['PS'] = var.phase_set
    # Set INFO values
    r.info['WIN_VAR_COUNT'] = var.window_var_count
    r.info['WIN_CIS_COUNT'] = var.window_cis_vars
    r.info['WIN_TRANS_COUNT'] = var.window_trans_vars
    r.info['QUALS'] = [float(x) for x in var.quals]
    r.info['CALL_COUNT'] = var.call_count
    r.info['STEP_COUNT'] = var.step_count
    r.info['WIN_OFFSETS'] = [int(x) for x in var.window_offset]
    r.info['VAR_INDEX'] = [int(x) for x in var.var_index]
    r.info['TNPRED'] = 0 #[float(f"{x :.4f}") for x in var.tnpred]
    if var.duplicate:
        r.info['DUPLICATE'] = ()
    return r


def vars_to_vcf(vcf_file, variants):
    """
    create variant records from pyranges variant table and write all to pysam VariantFile
    :param vcf_file:
    :param pr_vars: List of variants to write

    :return: None
    """
    for var in variants:
        r = create_vcf_rec(var, vcf_file)
        vcf_file.write(r)


