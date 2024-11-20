#!/usr/bin/env python
import random
from enum import Enum
from typing import List
import yaml
import numpy as np
import bisect
import pysam
import pickle
#import sklearn
import multiprocessing as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging
import argparse
from itertools import islice

logger = logging.getLogger(__name__)

SUPPORTS_REF=0
SUPPORTS_ALT=1
SUPPORTS_OTHER=2


class CigarOperator(Enum):
    BAM_CMATCH = 0  # 'M'
    BAM_CINS = 1  # 'I'
    BAM_CDEL = 2  # 'D'
    BAM_CREF_SKIP = 3  # 'N'
    BAM_CSOFT_CLIP = 4  # 'S'
    BAM_CHARD_CLIP = 5  # 'H'
    BAM_CPAD = 6  # 'P'
    BAM_CEQUAL = 7  # '='
    BAM_CDIFF = 8  # 'X'
    BAM_CBACK = 9  # 'B'

class OffsetOutOfBoundsException(Exception):
    """
    Raised when computing read position offset if we end up with an invalid value
    """
    pass

def trim_suffix(ref, alt):
    r = ref
    a = alt
    while r[-1] == a[-1] and min(len(r), len(a)) > 1:
        r = r[0:-1]
        a = a[0:-1]
    return r, a


def trim_prefix(ref, alt):
    offset = 0
    if len(ref) == 1 or len(alt) == 1:
        return 0, ref, alt
    for r, a in zip(ref[:-1], alt[:-1]):
        if r != a:
            return offset, ref[offset:], alt[offset:]
        offset += 1
    return offset, ref[offset:], alt[offset:]


def full_prefix_trim(ref, alt):
    """
    Trim prefix, leaving empty alleles if necessary
    """
    for i, (r, a) in enumerate(zip(ref + "$", alt + "$")):
        if r != a:
            break
    return i, ref[i:], alt[i:]


def find_var(varfile, chrom, pos, ref, alt):
    """
    Search varfile for VCF record with matching chrom, pos, ref, alt
    If chrom/pos/ref/alt match is found, return that record and allele index of matching alt
    """
    ref, alt = trim_suffix(ref, alt)
    poffset, trimref, trimalt = trim_prefix(ref, alt)
    pos = pos + poffset
    for var in varfile.fetch(chrom, pos-5, pos+3):
        for i, varalt in enumerate(var.alts):
            varref, varalt = trim_suffix(var.ref, varalt)
            offset, r, a = trim_prefix(varref, varalt)
            if var.pos + offset == pos and r == trimref and a == trimalt:
                return var, i



@lru_cache(maxsize=1000000)
def var_af(varfile, chrom, pos, ref, alt):
    result = find_var(varfile, chrom, pos, ref, alt)
    af = 0.0
    if result is not None:
        var, alt_index = result
        af = var.info['AF'][alt_index]
        if af is None or "PASS" not in var.filter:
            af = 0.0
        else:
            af = float(af)
    return af

def get_query_pos(read, ref_pos):
    """
    Returns the offset in the read sequence corresponding to the given reference position. Follows the
    rules given by bisect.bisect_left, so if multiple read reference positions correspond to the ref_pos,
    then the index of the leftmost is returned.
    :param read: pysam.AlignedSegment
    :param ref_pos: Reference genomic coordinate
    :return: Index in read.get_reference_positions() that corresponds to ref_pos
    """
    ref_mapping = read.get_reference_positions()
    if not ref_mapping:
        raise OffsetOutOfBoundsException('ref_mapping is empty for read {}'.format(read))
    offset = bisect.bisect_left(ref_mapping, ref_pos)
    if offset > len(ref_mapping):
        raise OffsetOutOfBoundsException('offset {} > len(ref_mapping): {}'.format(offset, len(ref_mapping)))
    elif ref_pos < ref_mapping[0]:
        raise OffsetOutOfBoundsException('ref_pos {} < ref_mapping[0] {}'.format(ref_pos, ref_mapping[0]))

    return offset

def cigop_at_offset(read, offset):
    """
    Return the cigar operation / length at the given offset in the read
    """
    if read.cigartuples is None:
        raise ValueError("Yikes, cigartuples is none!")
    for cigop, length in read.cigartuples:
        if offset <= length:
            return cigop, length
        elif cigop != CigarOperator.BAM_CINS.value: # Insertions dont count against ref bases so dont subtract them
            offset -= length
    return -1, -1


def read_support(read, offset, ref, alt):
    """
    Returns True if the given read appears to support the given alt allele
    Expects TRIMMED ref + alt variants as input
    :return: 0 if read supports ref allele, 1 if read supports alt, 2 oth
    """
    if len(ref) == 1 and len(alt) == 1:
        base = read.query_sequence[offset]
        if base == ref:
            return SUPPORTS_REF
        elif base == alt:
            return SUPPORTS_ALT
        else:
            return SUPPORTS_OTHER
    elif len(ref) > 0 and len(alt) == 0:
        # Deletion...
        cigop, length = cigop_at_offset(read, offset+1)
        if cigop == CigarOperator.BAM_CDEL.value and length == len(ref):
            return SUPPORTS_ALT
        elif cigop == CigarOperator.BAM_CDEL.value and length != len(ref):
            return SUPPORTS_OTHER
        else:
            return SUPPORTS_REF
    elif len(ref) == 0 and len(alt) > 0:
        # Insertion...
        cigop, length = cigop_at_offset(read, offset + 1)
        if cigop == CigarOperator.BAM_CINS.value and length == len(alt):
            return SUPPORTS_ALT
        elif cigop == CigarOperator.BAM_CINS.value and length != len(alt):
            return SUPPORTS_OTHER
        else:
            return SUPPORTS_REF
    else:
        # Multinucleotide var - look to see if the X bases match ref or alt, where X is min length of ref, alt
        # Since we trim matches bases on ref and alt this seems OK
        bases = read.query_sequence[offset:offset + min(len(ref), len(alt))]
        if bases == ref:
            return SUPPORTS_REF
        elif bases == alt:
            return SUPPORTS_ALT
        else:
            return SUPPORTS_OTHER




def bamfeats(chrom, start, ref, alt, aln):
    tot_reads = 0
    pos_ref = 0
    pos_alt = 0
    neg_ref = 0
    neg_alt = 0
    min_mq = 10
    highmq_ref = 0
    highmq_alt = 0
    pos_offset, trimref, trimalt = full_prefix_trim(ref, alt)
    if len(trimref) == 0 and len(trimalt) == 0:
        return 0, 0, 0, 0, 0, 0, 0

    varstart = start + pos_offset
    for read in aln.fetch(chrom, start-1, start+1):
        try:
            offset = get_query_pos(read, varstart)
            support_val = read_support(read, offset, trimref, trimalt)
            tot_reads += 1
        except:
            continue

        if read.is_forward:
            if support_val == SUPPORTS_REF:
                pos_ref += 1
            elif support_val == SUPPORTS_ALT:
                pos_alt += 1
        else:
            if support_val == SUPPORTS_REF:
                neg_ref += 1
            elif support_val == SUPPORTS_ALT:
                neg_alt += 1
        if read.mapping_quality > min_mq:
            if support_val == SUPPORTS_REF:
                highmq_ref += 1
            if support_val == SUPPORTS_ALT:
                highmq_alt += 1

    return tot_reads, pos_ref, pos_alt, neg_ref, neg_alt, highmq_ref, highmq_alt



def var_feats(var, bam_features):
    feats = []
    amreads, pos_ref, pos_alt, neg_ref, neg_alt, highmq_ref, highmq_alt = bam_features # bamfeats(var.chrom, var.start, var.ref, var.alts[0], aln)
    if amreads > 0:
        vaf = (pos_alt + neg_alt) / amreads
    else:
        vaf = -1.0 # In some important / complex cases we can't find any reads, and setting vaf = -1 flags these?

    feats.append(var.qual)
    feats.append(len(var.ref))
    feats.append(max(len(a) for a in var.alts))
    feats.append(min(var.info['QUALS']))
    feats.append(max(var.info['QUALS']))
    feats.append(var.info['WIN_VAR_COUNT'][0])
    feats.append(var.info['WIN_CIS_COUNT'][0])
    #feats.append(var.info['WIN_TRANS_COUNT'][0]) # Not used in latest lcbm model
    feats.append(var.info['STEP_COUNT'][0])
    feats.append(var.info['CALL_COUNT'][0])
    feats.append(min(var.info['WIN_OFFSETS']))
    feats.append(max(var.info['WIN_OFFSETS']))
    feats.append(var.samples[0]['DP'])
    
    feats.append(1 if 0 in var.samples[0]['GT'] else 0)
    feats.append(vaf)
    feats.append(amreads)
    feats.append(pos_alt + neg_alt)
    return np.array(feats)


def feat_names():
    return [
            "qual",
            "ref_len",
            "alt_len",
            "min_qual",
            "max_qual",
            "var_count",
            "cis_count",
            #"trans_count",
            "step_count",
            "call_count",
            "min_win_offset",
            "max_win_offset",
            "dp",
            #"af",
            "het",
            "vaf",
            "vafreads",
            "altreads",
          #  "strandbias_stat",
        ]


def varstr(var):
    return f"{var.chrom}:{var.pos}-{var.ref}-{var.alts[0]}"

def is_snv(var):
    return len(var.ref) == 1 and len(var.alts[0]) == 1

def is_del(var):
    return len(var.ref) > 1 and len(var.alts[0]) == 1

def is_ins(var):
    return len(var.ref) == 1 and len(var.alts[0]) > 1

def vcf_sampling_iter(vcf, max_snvs=float("inf"), max_dels=float("inf"), max_ins=float("inf"), skip=0):
    snvcount = 0
    delcount = 0
    inscount = 0
    varfile = pysam.VariantFile(vcf, ignore_truncation=True)
    if varfile is None:
        raise ValueError(f"Unable to open variant file {vcf}")
    for i, var in enumerate():
        if i < skip:
            continue
        if is_snv(var):
            snvcount += 1
            if snvcount > max_snvs:
                continue
        if is_del(var):
            delcount += 1
            if delcount > max_dels:
                continue
        if is_ins(var):
            inscount += 1
            if inscount > max_ins:
                continue
        if snvcount > max_snvs and delcount > max_dels and inscount > max_ins:
            break

        yield var


def rec_extract_feats(var, aln):
    bam_features = bamfeats(var.chrom, var.start, var.ref, var.alts[0], aln)
    feats = var_feats(var, bam_features)
    fstr = ",".join(str(x) for x in [varstr(var)] + list(feats))
    return feats, fstr


def extract_feats(vcf, aln):
    allfeats = []
    featstrs = []
    for var in vcf:
        feats, fstr = rec_extract_feats(var, aln)
        allfeats.append(feats)
        featstrs.append(fstr)
    return allfeats, featstrs


def save_model(mdl, path):
    logger.info(f"Saving model to {path}")
    with open(path, 'wb') as fh:
        pickle.dump(mdl, fh)
        

def load_model(path):
    logger.debug(f"Loading model from {path}")
    if str(path).endswith(".json") or str(path).endswith(".xgb"):
        bst = xgboost.Booster()
        bst.load_model(path)
        return bst
    else:
        with open(path, 'rb') as fh:
            return pickle.load(fh)

def _find_var(chrom, pos, ref, alts, vcf):
    assert type(alts) == tuple, "alts must be a tuple"
    alts = set(alts)
    for v in vcf.fetch(chrom, pos-2, pos+2):
        if v.chrom == chrom and v.pos == pos and v.ref == ref and set(v.alts) == alts:
            return v
    return None

def _process_sample(args):
    sample, bampath, reference_filename, varoutputs, tps, fps = args

    logger.info(f"Processing sample {sample}")
    aln = pysam.AlignmentFile(bampath, reference_filename=reference_filename)
    allfeats = []
    featstrs = []
    labels = []
    max_tp_snvs = 10000
    max_tp_dels = 5000
    max_tp_ins = 5000
    tpsnvs = 0
    tpins = 0
    tpdels = 0
    tp_downsample_freq = 0.1
    for outputfile, tp, fp in zip(varoutputs, tps, fps):
        tpfile = pysam.VariantFile(tp)
        fpfile = pysam.VariantFile(fp)
        for var in pysam.VariantFile(outputfile):
            base = var.info.get('BASE', '')
            if "CA" in base:
                continue
            call = var.info.get('CALL')
            if call is None:
                continue
            if call == 'TP' and random.random() < tp_downsample_freq:
                if ((is_snv(var) and tpsnvs < max_tp_snvs)
                        or (is_ins(var) and tpins < max_tp_ins)
                        or (is_del(var) and tpdels < max_tp_dels)):
                    tpvar = _find_var(var.chrom, var.pos, var.ref, var.alts, tpfile)
                    if tpvar is None:
                        logger.warning(f"Couldn't find true pos match for {var}")
                        continue
                    feats, fstr = rec_extract_feats(tpvar, aln)
                    allfeats.append(feats)
                    featstrs.append(fstr)
                    labels.append(1)
                    if is_snv(var):
                        tpsnvs += 1
                    elif is_ins(var):
                        tpins += 1
                    elif is_del(var):
                        tpdels += 1
            elif call == 'FP':
                alleles = list(var.samples['CALLS']['GT'])
                #print(f"Looking at {str(var).strip()}, alleles are: {alleles}")
                alts = tuple(var.alleles[a] for a in alleles if (a is not None) and a != 0 ) # Half calls like ./1 have None as an allele
                #print(f"     parsed alts: {alts}")
                fpvar = _find_var(var.chrom, var.pos, var.ref, alts, fpfile)
                if fpvar is None:
                    logger.warning(f"Couldn't find FP match for find variant {var}")
                    continue
                feats, fstr = rec_extract_feats(fpvar, aln)
                allfeats.append(feats)
                featstrs.append(fstr)
                labels.append(0)

    logger.info(f"Done with {sample} : TPs: {sum(labels)} FPs: {np.sum(1 - np.array(labels))}")
    return allfeats, featstrs, labels


def train_model(conf, threads, feat_csv=None, labels_csv=None, reference_filename=None):
    if feat_csv:
        logger.info(f"Writing feature dump to {feat_csv}")
        feat_fh = open(feat_csv, "w")
        feat_fh.write("varstr," + ",".join(feat_names()) + "\n")
    else:
        feat_fh = None
    if labels_csv:
        label_fh = open(labels_csv, "w")
        label_fh.write("label\n")
    else:
        label_fh = None

    with mp.Pool(threads) as pool:
        results = pool.map(_process_sample, ((sample, conf[sample]['bam'], reference_filename, conf[sample].get('vars'), conf[sample].get('tps'), conf[sample].get('fps')) for sample in conf.keys()))

    y = []
    feats = []
    for samplefeats, fstrs, labs in results:
        feats += samplefeats
        y.extend(labs)
        if feat_fh:
            for fstr in fstrs:
                feat_fh.write(fstr + "\n")
        if label_fh:
            for l in labs:
                label_fh.write(str(l) + "\n")

    if feat_fh:
        feat_fh.close()
    if label_fh:
        label_fh.close()

    y = np.array(y)
    logger.info(f"Loaded {np.sum(1 - y)} TP and {np.sum(y)} FPs")

    feat_train, feat_test, lab_train, lab_test = train_test_split(feats, y, test_size=0.1)
    print(f"Test set size: {len(lab_test)}")
    clf = RandomForestClassifier(n_estimators=100, max_depth=25, random_state=0, max_features=None, class_weight="balanced", n_jobs=threads)
    #clf = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=1, objective='binary:logistic')
    
    clf.fit(feat_train, lab_train)

    preds = clf.predict_proba(feat_test)[:,1]
    threshold = 0.1
    ppv, ppa, fscore, support = precision_recall_fscore_support(lab_test, preds > threshold)
    print(f"Metrics at threshold : {threshold}")
    print(f"PPA : {ppa[1] :.5f}")
    print(f"PPV : {ppv[1] :.5f}")
    print(f"F1 : {fscore[1] :.5f}")

    return clf


def predict(model, vcf, **kwargs):
    model = load_model(model)
    bam = pysam.AlignmentFile(kwargs.get('bam'))
    vcf = pysam.VariantFile(vcf, ignore_truncation=True)
    print(vcf.header, end='')
    for var in vcf:
        proba = predict_one_record(model, var, bam)
        var.qual = proba
        print(var, end='')


def batch(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


def process_batch(batch, bampath, refpath):
    results = []
    aln = pysam.AlignmentFile(bampath, reference_filename=refpath)
    for chrom, start, ref, alt in batch:
        result = bamfeats(chrom, start, ref, alt, aln)
        results.append(result)
    return results


def var_keys(var: pysam.VariantRecord):
    return var.chrom, var.start, var.ref, var.alts[0]


def collect_bam_features(var_records: List[pysam.VariantRecord], bampath: str, refpath: str, threads: int):
    """ 
    """
    futs = []
    batchsize = max(1, len(var_records) // threads)
    logger.debug(f"Processing {len(var_records)} with {threads} threads, batch size is: {batchsize}")
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for batch_records in batch(var_records, batchsize):
            vkeys = [var_keys(v) for v in batch_records]
            futs.append(pool.submit(process_batch, vkeys, bampath, refpath))

    bam_feat_results = []
    for i, fut in enumerate(futs):
        results = fut.result(timeout=60)        
        bam_feat_results.extend(results)  

    return bam_feat_results


def predict_records(varrecs: List[pysam.VariantRecord], loaded_model: RandomForestClassifier, bampath: str, refpath: str, threads: int):
    logger.debug(f"Predicting classifications for {len(varrecs)} records")
    bam_features = collect_bam_features(varrecs, bampath, refpath, threads)
    assert len(bam_features) == len(varrecs), f"Unequal numbers of features {len(bam_features)} and records: {len(varrecs)}"

    allfeats = []
    for var, bfeats in zip(varrecs, bam_features):
        feats = var_feats(var, bfeats)
        allfeats.append(feats)

    a = np.array(allfeats)
    if a.ndim == 1:
        a = a[np.newaxis, ...]  # Reshape to 2D array with one row
    
    predictions = loaded_model.predict_proba(a)[:, 1]
    return predictions.tolist()


def predict_one_record(var, loaded_model, bampath, refpath):
    """
    given a loaded model object and a pysam variant record, return classifier quality
    :param loaded_model: loaded model object for classifier
    :param var: single pysam vcf record
    :param kwargs:
    :return: classifier quality
    """
    aln = pysam.AlignmentFile(bampath, reference_filename=refpath)

    bam_features = bamfeats(var.chrom, var.start, var.ref, var.alts[0], aln)
    feats = var_feats(var, bam_features)

    #logger.debug(f"Feats for record: {var_rec.chrom}:{var_rec.pos} {var_rec.ref}->{var_rec.alts[0]} : {feats}")
    if isinstance(loaded_model, RandomForestClassifier):
        prediction = loaded_model.predict_proba(feats[np.newaxis, ...])
        return prediction[0, 1]
    else:
        prediction = loaded_model.predict(xgboost.DMatrix(feats[np.newaxis, ...]))
        return prediction


def train(conf, output, **kwargs):
    logger.info(f"Loading configuration from {conf}")
    conf = yaml.safe_load(open(conf).read())
    model = train_model(conf, 
            threads=kwargs.get('threads'), 
            feat_csv=kwargs.get('feat_csv'),
            labels_csv=kwargs.get('labels_csv'),
            reference_filename=kwargs.get('reference'))
    save_model(model, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", help="Number of threads to use", type=int, default=-1) # -1 means all threads
    
    subparser = parser.add_subparsers()

    trainparser = subparser.add_parser("train", help="Train a new model")
    trainparser.add_argument("-c", "--conf", help="Configuration file")
    trainparser.add_argument("-t", "--threads", help="thread count", default=24, type=int)
    trainparser.add_argument("-o", "--output", help="Output path")
    trainparser.add_argument("-r", "--reference", help="Reference genome fasta")
    trainparser.add_argument("--feat-csv", help="Feature dump CSV")
    trainparser.add_argument("--labels-csv", help="Label dump CSV")
    trainparser.set_defaults(func=train)

    predictparser = subparser.add_parser("predict", help="Predict")
    predictparser.add_argument("-m", "--model", help="Model file")
    predictparser.add_argument("-b", "--bam", help="Path to the bam")
    predictparser.add_argument("-v", "--vcf", help="Input VCF")
    predictparser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(**vars(args))


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s]  %(name)s  %(levelname)s  %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO)

    main()


