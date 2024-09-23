
import logging

logger = logging.getLogger(__name__)

import random
import math
from pathlib import Path
from collections import defaultdict
from itertools import chain
import lz4.frame
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import io
import functools

import numpy as np
import torch
import torch.multiprocessing as mp

import util
import pregen

class ReadLoader:
    """
    The simplest loader, this one just has a src and tgt tensor and iterates over them
    Assumes first index in each is the batch index
    """

    def __init__(self, src, tgt, device):
        assert src.shape[0] == tgt.shape[0]
        self.src = src
        self.tgt = tgt
        self.device = device

    def __len__(self):
        return self.src.shape[0]

    def iter_once(self, batch_size):
        offset = 0
        while offset < self.src.shape[0]:
            yield self.src[offset:offset + batch_size, :, :, :].to(self.device), self.tgt[offset:offset + batch_size, :, :].to(self.device), None, None
            offset += batch_size


class LazyLoader:
    """
    A loader that doesn't load anything until iter_once() is called, and doesn't save anything in memory
    This will be pretty slow but good if we can't fit all of the data into memory
    Useful for 'pre-gen' where we just want to iterate over everything once and save it to a file
    """

    def __init__(self, bam, bed, vcf, reference, reads_per_pileup, samples_per_pos, vals_per_class, max_jitter_bases):
        self.bam = bam
        self.bed = bed
        self.vcf = vcf
        self.reference = reference
        self.reads_per_pileup = reads_per_pileup
        self.samples_per_pos = samples_per_pos
        self.vals_per_class = vals_per_class
        self.max_jitter_bases = max_jitter_bases

    def iter_once(self, batch_size):
        logger.info(f"Encoding tensors from {self.bam} and {self.vcf}")
        for src, tgt, vaftgt, varsinfo in pregen.encode_chunks(self.bam,
                                                        self.reference,
                                                        self.bed,
                                                        self.vcf,
                                                        batch_size,
                                                        self.reads_per_pileup,
                                                        self.samples_per_pos,
                                                        self.vals_per_class,
                                                        max_to_load=1e9,
                                                        max_jitter_bases=self.max_jitter_bases):
            yield src, tgt, vaftgt, varsinfo



def decomp_single(path):
    with open(path, 'rb') as fh:
        return torch.load(io.BytesIO(lz4.frame.decompress(fh.read())), map_location='cpu')


def decompress_multi_ppe(paths, threads):
    """
    Read & decompress all of the items in the paths with lz4, then load them into Tensors
    but keep them on the CPU
    :returns : List of Tensors (all on CPU)
    """
    logger.info("Hey we are in the function")
    start = datetime.now()
    result = []
    futs = []
    q = mp.Queue()
    dfunc = functools.partial(decomp_single, queue=q)
    logger.info(f"Making the pool")
    with ProcessPoolExecutor(threads) as pool:
        for path in paths:
            logger.info(f"Submitting path: {path}")
            fut = pool.submit(dfunc, path)
            futs.append(fut)

    logger.info(f"Submitted all the funcs, now waiting for results, length: {len(q)}")
    for f in futs:
        logger.info("Waiting for item...")
        f.result(timeout=20)
        r = q.get()
        logger.info(f"Got : {r}")
        result.append(r)

    #for i, r in enumerate(result):
    #    logger.info(f"Result item {i}: {r.shape}")
        
    elapsed = datetime.now() - start
    logger.info(
        f"Decompressed {len(result)} items in {elapsed.total_seconds():.3f} seconds ({elapsed.total_seconds() / len(result):.3f} secs per item)"
    )
    return result


def decompress_multi_map(paths, threads):
    """
    Read & decompress all of the items in the paths with lz4, then load them into Tensors
    but keep them on the CPU
    :returns : List of Tensors (all on CPU)
    """
    torch.set_num_threads(1)
    start = datetime.now()
    #decompressed = []
    with mp.Pool(threads) as pool:
        result = pool.map(decomp_single, paths)
    
    #result = [torch.load(d, map_location='cpu') for d in decompressed]
           
    elapsed = datetime.now() - start
    logger.info(
        f"Decompressed {len(result)} items in {elapsed.total_seconds():.3f} seconds ({elapsed.total_seconds() / len(result):.3f} secs per item)"
    )
    return result


def decomp_profile(paths, threads):
    result = []
    paths = list(paths)
    read_sum = timedelta(0)
    decomp_sum = timedelta(0)
    load_sum = timedelta(0)
    for path in paths:
        start = datetime.now()
        with open(path, 'rb') as fh:
            r = fh.read()
        read_dt = datetime.now()
        d = io.BytesIO(lz4.frame.decompress(r))
        decomp_dt = datetime.now()
        t = torch.load(d, map_location='cpu')
        load_dt = datetime.now()
        result.append(t)        
        
        read_sum = read_sum + (read_dt - start)
        decomp_sum = decomp_sum + (decomp_dt - read_dt)
        load_sum = load_sum + (load_dt - decomp_dt)

    tot = read_sum + decomp_sum + load_sum
    logger.info(f"Decomped {len(paths)} items")
    logger.info(f"Total decomp time (secs): {tot.total_seconds() :.6f}")
    logger.info(f"Read frac: {read_sum.total_seconds() / tot.total_seconds() * 100 :.3f}")
    logger.info(f"Decomp frac: {decomp_sum.total_seconds() / tot.total_seconds() * 100 :.3f}")
    logger.info(f"Load frac: {load_sum.total_seconds() / tot.total_seconds() * 100 :.3f}")
    return result



def iterate_dir(device, pathpairs, batch_size, max_decomped, threads):
    """
    Iterate the data (pathpairs, which is a list of tuples of (src tensors, target tensors), decompressing sets
    of the data in parallel using 'threads' threads. Yield (src, tgt, None, None, None) values (the Nones are used
    for additional labels or debugging)
    """
    src, tgt = [], []
    for i in range(0, len(pathpairs), max_decomped):
        logger.info(f"Decompressing {i}-{i + max_decomped} files of {len(pathpairs)}")
        decomp_start = datetime.now()
        paths = pathpairs[i:i + max_decomped]
        decomped = decompress_multi_map(chain.from_iterable(paths), threads)
        decomp_end = datetime.now()
        decomp_time = (decomp_end - decomp_start).total_seconds()

        for j in range(0, len(decomped), 2):
            src.append(decomped[j])
            tgt.append(decomped[j + 1])

        total_size = sum([s.shape[0] for s in src])
        if total_size < batch_size:
            # We need to decompress more data to make a batch
            continue

        # Make a big tensor.
        src_t = torch.cat(src, dim=0)
        tgt_t = torch.cat(tgt, dim=0)

        nbatch = total_size // batch_size
        remain = total_size % batch_size

        # Slice the big tensors for batches
        for n in range(0, nbatch):
            start = n * batch_size
            end = (n + 1) * batch_size
            yield (
                src_t[start:end].to(device).float(),
                tgt_t[start:end].to(device).long(),
                None,  # vaftgt_t[start:end].to(self.device),
                None,
                {"decomp_time": decomp_time},
            )
            decomp_time = 0.0

        if remain:
            # The remaining data points will be in next batch.
            src = [src_t[nbatch * batch_size:]]
            tgt = [tgt_t[nbatch * batch_size:]]
        else:
            src, tgt = [], []

    if len(src) > 0:
        # We need to yield the last batch.
        yield (
            torch.cat(src, dim=0).to(device).float(),
            torch.cat(tgt, dim=0).to(device).long(),
            None,
            None,
            {"decomp_time": 0.0},
        )
    logger.info(f"Done iterating data")

def load_files(datadir, src_prefix="src", tgt_prefix=""):
    pathpairs = util.find_files(datadir, src_prefix, tgt_prefix)
    logger.info(f"Loaded {len(pathpairs)} from {datadir}")
    random.shuffle(pathpairs)
    return pathpairs

class CurriculumLoader:

    def __init__(self, device, datadirs, switchpoints, threads, max_decomped_batches=10):
        self.device = device
        self.datadirs = datadirs
        self.switchpoints = switchpoints
        self.threads = threads


    def iter_once(self, batch_size):
        self.load_files()  # Search for new data with every iteration ?
        for result in iterate_dir(self.device, self.pathpairs, batch_size, self.max_decomped, self.threads):
            yield result


class PregenLoader:

    def __init__(self, device, datadir, threads, max_decomped_batches=10, src_prefix="src", tgt_prefix="tgt", vaftgt_prefix="vaftgt", pathpairs=None):
        """
        Create a new loader that reads tensors from a 'pre-gen' directory
        :param device: torch.device
        :param datadir: Directory to read data from
        :param threads: Max threads to use for decompressing data
        :param max_decomped_batches: Maximum number of batches to decompress at once. Increase this on machines with tons of RAM
        :param pathpairs: List of (src path, tgt path, vaftgt path) tuples to use for data
        """
        self.device = device
        self.datadir = Path(datadir) if datadir else None
        self.src_prefix = src_prefix
        self.tgt_prefix = tgt_prefix
        if pathpairs and datadir:
            raise ValueError(f"Both datadir and pathpairs specified for PregenLoader - please choose just one")
        if pathpairs:
            self.pathpairs = pathpairs
        else:
            self.pathpairs = load_files(self.datadir, self.src_prefix, self.tgt_prefix)
            
        self.threads = threads
        self.max_decomped = max_decomped_batches # Max number of decompressed items to store at once - increasing this uses more memory, but allows increased parallelization
        logger.info(f"Creating PreGen data loader with {self.threads} threads")
        logger.info(f"Found {len(self.pathpairs)} batches in {datadir}")
        logger.info(f"Possible sharing strategies: {mp.get_all_sharing_strategies()}")
        #mp.set_sharing_strategy("file_system")
        logger.info(f"Current sharing strategy: {mp.get_sharing_strategy()}")
        if not self.pathpairs:
            raise ValueError(f"Could not find any files in {datadir}")


    def retain_val_samples(self, fraction):
        """
        Remove a fraction of the samples from this loader and return them
        :returns : List of (src, tgt) PATH tuples (not loaded tensors)
        """
        num_to_retain = int(math.ceil(len(self.pathpairs) * fraction))
        val_samples = random.sample(self.pathpairs, num_to_retain)
        newdata = []
        for sample in self.pathpairs:
            if sample not in val_samples:
                newdata.append(sample)
        self.pathpairs = newdata
        logger.info(f"Number of batches left for training: {len(self.pathpairs)}")
        return val_samples

    def iter_once(self, batch_size):
        """
        Make one pass over the training data, in this case all of the files in the 'data dir'
        Training data is compressed and on disk, which makes it slow. To increase performance we
        load / decomp several batches in parallel, then train over each decompressed batch
        sequentially
        :param batch_size: The number of samples in a minibatch.
        """
        self.pathpairs = load_files(self.datadir, self.src_prefix, self.tgt_prefix) # Search for new data with every iteration ?
        for result in iterate_dir(self.device, self.pathpairs, batch_size, self.max_decomped, self.threads):
            yield result


class ReadShuffleLoader:

    def __init__(self, loader):
        self.loader = loader

    def iter_once(self, batch_size):
        for src, tgt, *_, in self.loader.iter_once(batch_size):
            # Permute the entries in dimension 1 of the src tensor, but don't change the special element at index 0
            # This is fast but naive - each element of the batch is permuted the exact same way
            perm = (torch.randperm(src.size(1)-1) + 1)
            z = torch.zeros(1).int()
            idx = torch.cat((z, perm))
            src = src[:, idx, :, :]

            yield src, tgt, *_
