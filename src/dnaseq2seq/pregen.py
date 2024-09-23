

from functools import partial
from collections import defaultdict
import multiprocessing as mp
import itertools
import logging
import random
import traceback as tb
import yaml
from string import ascii_letters, digits
import lz4.frame
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pysam
import torch

import util
from bam import target_string_to_tensor, encode_and_downsample, ensure_dim
import phaser
import loader

logger = logging.getLogger(__name__)


def load_conf(confyaml):
    logger.info(f"Loading configuration from {confyaml}")
    conf = yaml.safe_load(open(confyaml).read())
    assert 'reference' in conf, "Expected 'reference' entry in training configuration"
    assert 'data' in conf, "Expected 'data' entry in training configuration"
    return conf


def default_vals_per_class():
    """
    Multiprocess will instantly deadlock if a lambda or any callable not defined on the top level of the module is given
    as the 'factory' argument to defaultdict - but we have to give it *some* callable that defines the behavior when the key
    is not present in the dictionary, so this returns the default "vals_per_class" if a class is encountered that is not
    specified in the configuration file. I don't think there's an easy way to make this user-settable, unfortunately
    """
    return 0

def pregen_one_sample(dataloader, batch_size, output_dir):
    """
    Pregenerate tensors for a single sample
    """
    TRUNCATE_LEN = 148 # Truncate target sequence in bases to this length, which should be evenly divisible from kmer length
    uid = "".join(random.choices(ascii_letters + digits, k=8))
    src_prefix = "src"
    tgt_prefix = "tgkmers"
    tn_prefix = "tntgt"

    logger.info(f"Saving tensors from {dataloader.bam} to {output_dir}/ with uid: {uid}")
    for i, (src, tgt, tntgt, varsinfo) in enumerate(dataloader.iter_once(batch_size)):
        tgt_kmers = util.tgt_to_kmers(tgt[:, :, 0:TRUNCATE_LEN]).float()
        logger.info(f"Saving batch {i} with uid {uid}")
        logger.info(f"Src dtype is {src.dtype}")

        # For debugging on only!
        #uid = Path(dataloader.bam).name.replace(".cram", "")

        for data, prefix in zip([src, tgt_kmers, tntgt],
                                [src_prefix, tgt_prefix, tn_prefix]):
            with lz4.frame.open(output_dir / f"{prefix}_{uid}-{i}.pt.lz4", "wb") as fh:
                torch.save(data, fh)

def pregen(config, **kwargs):
    """
    Pre-generate tensors from BAM files + labels and save them in 'datadir' for quicker use in training
    (this takes a long time)
    """
    conf = load_conf(config)
    batch_size = kwargs.get('batch_size', 64)
    reads_per_pileup = kwargs.get('read_depth', 150)
    samples_per_pos = kwargs.get('samples_per_pos', 2)
    processes = kwargs.get('threads', 1)
    jitter = kwargs.get('jitter', 0)

    vals_per_class = defaultdict(default_vals_per_class)
    vals_per_class.update(conf['vals_per_class'])

    logger.info(f"Full config: {conf}")
    output_dir = Path(kwargs.get('dir'))

    logger.info(f"Generating training data using config from {config} vals_per_class: {vals_per_class}")
    dataloaders = [
            loader.LazyLoader(c['bam'], c['bed'], c['vcf'], conf['reference'], reads_per_pileup, samples_per_pos, vals_per_class, max_jitter_bases=jitter)
        for c in conf['data']
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Submitting {len(dataloaders)} jobs with {processes} process(es)")

    if processes == 1:
        for dl in dataloaders:
            pregen_one_sample(dl, batch_size, output_dir)

    else:
        futures = []
        with ProcessPoolExecutor(max_workers=processes) as executor:
            for dl in dataloaders:
                futures.append(executor.submit(pregen_one_sample, dl, batch_size, output_dir))
        for fut in futures:
            fut.result()


# def find_sus_regions(bam, bed, reference, threads):
#     aln = pysam.AlignmentFile(bam, reference_filename=reference)
#     cluster_positions_func = partial(
#         call.cluster_positions_for_window,
#         bamfile=aln,
#         reference_fasta=reference,
#         maxdist=100,
#     )
#     sus_regions = mp.Pool(threads).map(cluster_positions_func, bed)
#     sus_regions = util.merge_overlapping_regions(list(itertools.chain(*sus_regions)))
#     return sus_regions


def trim_pileuptensor(src, tgt, width):
    """
    Trim or zero-pad the sequence dimension of src and target (first dimension of src, second of target)
    so they're equal to width
    :param src: Data tensor of shape [sequence, read, features]
    :param tgt: Target tensor of shape [haplotype, sequence]
    :param width: Size to trim to
    """
    assert src.shape[0] == tgt.shape[-1], f"Unequal src and target lengths ({src.shape[0]} vs. {tgt.shape[-1]}), not sure how to deal with this :("
    if src.shape[0] < width:
        z = torch.zeros(width - src.shape[0], src.shape[1], src.shape[2], dtype=src.dtype)
        src = torch.cat((src, z))
        t = torch.zeros(tgt.shape[0], width - tgt.shape[1]).long()
        tgt = torch.cat((tgt, t), dim=1)
    else:
        start = src.shape[0] // 2 - width // 2
        src = src[start:start+width, :, :]
        tgt = tgt[:, start:start+width]

    return src, tgt


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
            rows.append((chrom, start, end, clz))
            if clz in classes:
                idx = classes[clz]
            else:
                idx = classcount
                classcount += 1
                classes[clz] = idx
            idxs.append(idx)
    class_names = {v: k for k, v in classes.items()}
    return rows, idxs, class_names


def resample_classes(classes, class_names, rows, vals_per_class):
    """
    Randomly sample each class so that there are 'vals_per_class' members of each class
    then return a list of class assignments and the selected rows
    """
    byclass = defaultdict(list)
    for clz, row in zip(classes, rows):
        byclass[clz].append(row)

    for name in class_names.values():
        if name not in vals_per_class:
            logger.warning(f"Class name '{name}' not found in vals per class, will select default of {vals_per_class['asldkjas']} items from class {name}")

    result_rows = []
    result_classes = []
    for clz, classrows in byclass.items():

        # If there are MORE instances on this class than we want, grab a random sample of them
        logger.info(f"Variant class {class_names[clz]} contains {len(classrows)} variants")
        if len(classrows) > vals_per_class[class_names[clz]]:
            logger.info(f"Randomly sampling {vals_per_class[class_names[clz]]} variants for variant class {class_names[clz]}")
            rows = random.sample(classrows, vals_per_class[class_names[clz]])
            result_rows.extend(rows)
            result_classes.extend([clz] * vals_per_class[class_names[clz]])
        else:
            # If there are FEWER instances of the class than we want, then loop over them
            # repeatedly - no random sampling since that might ignore a few of the precious few samples we do have
            logger.info(f"Loop over {len(classrows)} variants repeatedly to generate {vals_per_class[class_names[clz]]} {class_names[clz]} variants")
            for i in range(vals_per_class[class_names[clz]]):
                result_classes.append(clz)
                idx = i % len(classrows)
                result_rows.append(classrows[idx])

    return result_classes, result_rows


def upsample_labels(bed, vals_per_class):
    """
    Generate class assignments for each element in rows (see assign_class_indexes),
     then create a new list of rows that is "upsampled" to normalize class frequencies
     The new list will be multiple times larger then the old list and will contain repeated
     elements of the low frequency classes
    :param rows: List of elements with vtype and status attributes (probably read in from a labels CSV file)
    :param vals_per_class: Number of instances to retain for each class, OR a Mapping with class names and values
    :returns: List of rows, with less frequent rows included multiple times to help normalize frequencies
    """
    rows, label_idxs, label_names = parse_rows_classes(bed)
    label_idxs, rows = resample_classes(
        np.array(label_idxs), label_names, rows, vals_per_class=vals_per_class
    )
    # classes, counts = np.unique(label_idxs, return_counts=True

    random.shuffle(rows)
    return rows


def load_from_csv(bampath, refpath, bed, vcfpath, max_reads_per_aln, samples_per_pos, vals_per_class, max_jitter_bases=0):
    """
    Generator for encoded pileups, reference, and alt sequences obtained from a
    alignment (BAM) and labels csv file. This performs some class normalized by upsampling / downsampling
    rows from the CSV based on "class", which is something like "snv-TP" or "ins-FP" (defined by assign_class_indexes
    function), may include VAF at some point?
    :param bampath: Path to input BAM file
    :param refpath: Path to ref genome
    :param csv: CSV file containing C/S/R/A, variant status (TP, FP, FN..) and type (SNV, del, ins, etc)
    :param max_reads_per_aln: Max reads to import for each pileup
    :param samples_per_pos: Max number of samples to create for a given position when read resampling
    :return: Generator
    """
    refgenome = pysam.FastaFile(refpath)
    bam = pysam.AlignmentFile(bampath, reference_filename=refpath)
    vcf = pysam.VariantFile(vcfpath)

    upsampled_labels = upsample_labels(bed, vals_per_class=vals_per_class)
    logger.info(f"Will save {len(upsampled_labels)} with up to {samples_per_pos} samples per site from {bed}")
    for region in upsampled_labels:
        chrom, start, end = region[0:3]
        rowlabel = region[3]
        row_is_tn = rowlabel.strip().startswith("tn-") # Indicator for whether this row contains any variants

        # Add some jitter?
        if max_jitter_bases > 0:
            jitter = np.random.randint(0, max_jitter_bases) - max_jitter_bases // 2
            start, end = start + jitter, end + jitter

        variants = list(vcf.fetch(chrom, start, end))
        #logger.info(f"Number of variants in {chrom}:{start}-{end} : {len(variants)}")
        try:
            for encoded, (minref, maxref) in encode_and_downsample(chrom,
                                                 start,
                                                 end,
                                                 bam,
                                                 refgenome,
                                                 max_reads_per_aln,
                                                 samples_per_pos):

                hap0, hap1 = phaser.gen_haplotypes(bam, refgenome, chrom, minref, maxref + 100, variants)
                regionsize = end - start
                midstart = max(0, start - minref)
                midend = midstart + regionsize

                tgt0 = target_string_to_tensor(hap0[midstart:midend])
                tgt1 = target_string_to_tensor(hap1[midstart:midend])
                tgt_haps = torch.stack((tgt0, tgt1))

                if encoded.shape[0] > regionsize:
                    encoded = encoded[midstart:midend, :, :]
                if encoded.shape[0] != tgt_haps.shape[-1]:
                    # This happens sometimes at the edges of regions where the reads only overlap a few bases at
                    # the start of the region.. raising an exception just causes this region to be skipped
                    raise ValueError(f"src shape {encoded.shape} doesn't match haplotype shape {tgt_haps.shape}, skipping")

                yield encoded, tgt_haps, region, row_is_tn

        except Exception as ex:
            logger.warning(f"Error encoding position {chrom}:{start}-{end} for bam: {bampath}, skipping it: {ex}")
            tb.print_exc()


def encode_chunks(bampath, refpath, bed, vcf, chunk_size, max_reads_per_aln, samples_per_pos, vals_per_class, max_to_load=1e9, max_jitter_bases=0):
    """
    Generator for creating batches of src, tgt (data and label) tensors from a CSV file

    :param bampath: Path to BAM file
    :param refpath: Path to human genome reference fasta file
    :param csv: Path to labels CSV file
    :param chunk_size: Number of samples / instances to include in one generated tensor
    :param max_reads_per_aln: Max read depth per tensor (defines index 2 of tensor)
    :param samples_per_pos: Randomly resample reads this many times, max, per position
    :param vals_per_class:
    :returns : Generator of src, tgt tensor tuples
    """
    allsrc = []
    alltgt = []
    allrowtn = []
    varsinfo = []
    count = 0
    seq_len = 150
    logger.info(f"Creating new data loader from {bampath}, vals_per_class: {vals_per_class}")
    for enc, tgt, region, tnlabel in load_from_csv(bampath, refpath, bed, vcf,
                                          max_reads_per_aln=max_reads_per_aln,
                                          samples_per_pos=samples_per_pos,
                                          vals_per_class=vals_per_class,
                                          max_jitter_bases=max_jitter_bases):
        src, tgt = trim_pileuptensor(enc, tgt, seq_len)
        src = ensure_dim(src, seq_len, max_reads_per_aln)
        allrowtn.append(tnlabel)
        assert src.shape[0] == seq_len, f"Src tensor #{count} had incorrect shape after trimming, found {src.shape[0]} but should be {seq_len}"
        assert tgt.shape[-1] == seq_len, f"Tgt tensor #{count} had incorrect shape after trimming, found {tgt.shape[-1]} but should be {seq_len}"
        allsrc.append(src)
        alltgt.append(tgt)

        count += 1

        if count == max_to_load:
            #logger.info(f"Stopping tensor load after {max_to_load}")
            yield torch.stack(allsrc).char(), torch.stack(alltgt).long(), torch.tensor(allrowtn), varsinfo
            break

        if len(allsrc) >= chunk_size:
            yield torch.stack(allsrc).char(), torch.stack(alltgt).long(), torch.tensor(allrowtn), varsinfo
            allsrc = []
            alltgt = []
            allrowtn = []
            varsinfo = []

    if len(allsrc):
        yield torch.stack(allsrc).char(), torch.stack(alltgt).long(), torch.tensor(allrowtn), varsinfo

    logger.info(f"Done loading {count} tensors from {bampath}")


