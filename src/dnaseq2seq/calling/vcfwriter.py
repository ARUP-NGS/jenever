from collections import defaultdict
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import json
import time
import pysam

from dnaseq2seq.util import SortedVariantWriter
from dnaseq2seq.calling import vcf
from dnaseq2seq import util
from dnaseq2seq.calling.vcf import Variant
from dnaseq2seq.calling import stage


logger = logging.getLogger(__name__)


@dataclass
class VCFRecord:
    """
    A serializable VCF record that holds all standard VCF fields, because pysam records are not pickle-able.
    When created from a pysam record, the pre-formatted VCF line is stored so that
    SortedVariantWriter can write it directly via str().
    """
    chrom: str
    pos: int
    ref: str
    alts: List[str]
    id: Optional[str] = None
    qual: Optional[float] = None
    filter: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    format: List[str] = field(default_factory=list)
    samples: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    vcf_line: Optional[str] = None

    def __str__(self) -> str:
        if self.vcf_line is not None:
            return self.vcf_line
        raise ValueError("VCFRecord has no vcf_line set; cannot convert to string")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this VCFRecord to a plain dictionary.
        
        :returns: Dictionary with all VCF fields
        """
        return asdict(self)

    def to_json(self) -> str:
        """
        Serialize this VCFRecord to a JSON string.
        
        :returns: JSON string representation
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VCFRecord':
        """
        Create a VCFRecord from a dictionary.
        
        :param data: Dictionary with VCF fields
        :returns: VCFRecord instance
        """
        required_fields = ['chrom', 'pos', 'ref', 'alts']
        for fld in required_fields:
            if fld not in data:
                raise ValueError(f"Missing required field: {fld}")
        
        return cls(
            chrom=data['chrom'],
            pos=data['pos'],
            ref=data['ref'],
            alts=data.get('alts', []),
            id=data.get('id'),
            qual=data.get('qual'),
            filter=data.get('filter', []),
            info=data.get('info', {}),
            format=data.get('format', []),
            samples=data.get('samples', {})
        )

    @classmethod
    def from_pysam(cls, rec: pysam.VariantRecord) -> 'VCFRecord':
        """
        Create a VCFRecord from a pysam.VariantRecord.
        Captures the pre-formatted VCF line so str() works with SortedVariantWriter.
        
        :param rec: pysam.VariantRecord instance
        :returns: VCFRecord instance
        """
        filter_list = list(rec.filter.keys()) if rec.filter else []
        info_dict = {key: rec.info[key] for key in rec.info.keys()}
        format_list = list(rec.format.keys()) if rec.format else []

        samples_dict = {}
        for sample_name in rec.samples:
            sample = rec.samples[sample_name]
            samples_dict[sample_name] = {key: sample[key] for key in sample.keys()}
        
        return cls(
            chrom=rec.chrom,
            pos=rec.pos,
            ref=rec.ref,
            alts=list(rec.alts) if rec.alts else [],
            id=rec.id,
            qual=rec.qual,
            filter=filter_list,
            info=info_dict,
            format=format_list,
            samples=samples_dict,
            vcf_line=str(rec),
        )


def init_vcf_output(vcf_out_path, vcf_header_extras):
    """
    Initialize the VCF output file and template, these items are not pickle-able
    """
    vcf_header = vcf.create_vcf_header(sample_name="sample", lowcov=20, cmdline=vcf_header_extras)
    vcf_template = pysam.VariantFile("/dev/null", mode='w', header=vcf_header)
    vcf_out = open(vcf_out_path, "w")
    vcf_out.write(str(vcf_header))
    vcf_out.flush()
    return vcf_out, vcf_template


class VCFWriter:
    """
    Stage worker that converts haplotype variant calls into serializable VCFRecord objects.
    Returns a MultiResult of VCFRecords so that multiple workers can run in parallel.
    The actual file writing is handled by the downstream VCFCollector stage.
    """
    def __init__(self, refpath=None, bampath=None, classifier_model=None):
        self.bampath = bampath
        self.refpath = refpath
        self.classifier_model = classifier_model
        self.bam = None
        self.reference = None
        self.vcf_template = None

    def _init_resources(self):
        self.bam = pysam.AlignmentFile(self.bampath, reference_filename=self.refpath)
        self.reference = pysam.FastaFile(self.refpath)
        vcf_header = vcf.create_vcf_header(sample_name="sample", lowcov=20, cmdline=None)
        self.vcf_template = pysam.VariantFile("/dev/null", mode='w', header=vcf_header)

    def __call__(self, hapvars):
        if self.bam is None:
            self._init_resources()

        if hapvars is None:
            return stage.SkipResult()

        records = vars_hap_to_records(hapvars.hap0, hapvars.hap1, self.bam, self.reference, self.classifier_model, self.vcf_template)
        if not records:
            return stage.SkipResult()
        vcf_records = [VCFRecord.from_pysam(rec) for rec in records]
        return stage.MultiResult(vcf_records)


class VCFCollector:
    """
    Final stage worker that collects serializable VCFRecord objects from
    (potentially parallel) VCFWriter workers and writes them through a single
    SortedVariantWriter. Must run with exactly one worker since
    SortedVariantWriter is not thread-safe.
    """
    def __init__(self, vcf_path, chrom_order=None, vcf_header_extras=None):
        self.vcf_path = vcf_path
        self.chrom_order = chrom_order
        self.vcf_header_extras = vcf_header_extras
        self.vcf_out_fh = None
        self.writer = None

    def _init_output(self):
        self.vcf_out_fh, _ = init_vcf_output(self.vcf_path, self.vcf_header_extras)
        self.writer = SortedVariantWriter(self.vcf_out_fh, self.chrom_order)

    def __call__(self, record: VCFRecord):
        if self.vcf_out_fh is None:
            self._init_output()
        self.writer.put(record)
        return 1

    def flush(self):
        logger.info("VCFCollector flushing and closing file handle")
        if self.writer:
            self.writer.flush()
        if self.vcf_out_fh:
            self.vcf_out_fh.close()
        return stage.SkipResult()


def merge_multialts(v0: VCFRecord, v1: VCFRecord) -> VCFRecord:
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

def vars_hap_to_records(vars_hap0, vars_hap1, aln, reference, classifier_model, vcf_template):
    """
    Convert variant haplotype objects to variant records
    """

    # Merging vars can sometimes cause a poor quality variant to clobber a very high quality one, to avoid this
    # we hard-filter out very poor quality variants that overlap other, higher-quality variants
    # This value defines the min qual to be included when merging overlapping variants
    min_merge_qual = 0.01
    global TOTAL_TIME_CLF

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

    # if classifier_model:
    #     clfstart = time.time()
        
    #     clf_preds = buildclf.predict_records(vcf_records, classifier_model, bampath, refpath, threads=16)
    #     for rec, pred in zip(vcf_records, clf_preds):
    #         rec.qual = pred

    #     clfend = time.time()
    #     TOTAL_TIME_CLF += clfend - clfstart
    #     logger.debug(f"Predicted variant quality for {len(vcf_records)} records in {(clfend - clfstart):6f} seconds ({(clfend - clfstart)/len(vcf_records) :6f} per record)")

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

