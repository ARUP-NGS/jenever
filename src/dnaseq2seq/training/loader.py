
import logging

logger = logging.getLogger(__name__)

import random
import math
from pathlib import Path
from itertools import chain
import lz4.frame
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import io
import functools
from typing import Union, List
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import torch
import torch.multiprocessing as mp

from dnaseq2seq import util
from dnaseq2seq.training.lmdbdataset import LMDBDataset
from dnaseq2seq.training.varsinfodatawrapper import VarsInfoDataWrapper

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
    src, tgt, tntgt = [], [], []
    for i in range(0, len(pathpairs), max_decomped):
        logger.info(f"Decompressing {i}-{i + max_decomped} files of {len(pathpairs)}")
        decomp_start = datetime.now()
        paths = pathpairs[i:i + max_decomped]
        decomped = decompress_multi_map(chain.from_iterable(paths), threads)
        decomp_end = datetime.now()
        decomp_time = (decomp_end - decomp_start).total_seconds()

        for j in range(0, len(decomped), 3):
            src.append(decomped[j])
            tgt.append(decomped[j + 1])
            tntgt.append(decomped[j + 2])

        total_size = sum([s.shape[0] for s in src])
        if total_size < batch_size:
            # We need to decompress more data to make a batch
            continue

        # Make a big tensor.
        src_t = torch.cat(src, dim=0)
        tgt_t = torch.cat(tgt, dim=0)
        tntgt_t = torch.cat(tntgt, dim=0)

        nbatch = total_size // batch_size
        remain = total_size % batch_size

        # Slice the big tensors for batches
        for n in range(0, nbatch):
            start = n * batch_size
            end = (n + 1) * batch_size
            yield {
                "read": src_t[start:end].to(device).float(),
                "tgkmers": tgt_t[start:end].to(device).long(),
                "tntgt": tntgt_t[start:end].to(device).float(),
                "altmask": None,
                "log_info": {"decomp_time": decomp_time},
            }
            decomp_time = 0.0

        if remain:
            # The remaining data points will be in next batch.
            src = [src_t[nbatch * batch_size:]]
            tgt = [tgt_t[nbatch * batch_size:]]
            tntgt = [tntgt_t[nbatch * batch_size:]]
        else:
            src, tgt, tntgt = [], [], []

    if len(src) > 0:
        # We need to yield the last batch.
        yield {
            "read": torch.cat(src, dim=0).to(device).float(),
            "tgkmers": torch.cat(tgt, dim=0).to(device).long(),
            "tntgt": torch.cat(tntgt, dim=0).to(device).float(),
            "altmask": None,
            "log_info": {"decomp_time": 0.0},
        }
    logger.info(f"Done iterating data")

def load_files(datadir, src_prefix, tgt_prefix, tn_prefix):
    pathpairs = util.find_files(datadir, src_prefix, tgt_prefix, tn_prefix)
    logger.info(f"Loaded {len(pathpairs)} from {datadir}")
    random.shuffle(pathpairs)
    return pathpairs


class PregenLoader:

    def __init__(self, device, datadir, threads, batch_size, max_decomped_batches=10, src_prefix="src", tgt_prefix="tgt", tn_prefix="tntgt", pathpairs=None):
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
        self.tn_prefix = tn_prefix
        self.batch_size = batch_size
        if pathpairs and datadir:
            raise ValueError(f"Both datadir and pathpairs specified for PregenLoader - please choose just one")
        if pathpairs:
            self.pathpairs = pathpairs
        else:
            self.pathpairs = load_files(self.datadir, self.src_prefix, self.tgt_prefix, self.tn_prefix)
            
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

    def __len__(self):
        """
        Return the number of batches available in this loader
        """
        return len(self.pathpairs)


    def __iter__(self):
        """
        Make this loader a standard Python iterable by yielding items from iter_once
        Uses a default batch_size of 1 for iteration
        """
        # Use a reasonable default batch size for iteration
        # This could be made configurable if needed
        for result in self.iter_once(self.batch_size):
            yield result

    def iter_once(self, batch_size=None):
        """
        Make one pass over the training data, in this case all of the files in the 'data dir'
        Training data is compressed and on disk, which makes it slow. To increase performance we
        load / decomp several batches in parallel, then train over each decompressed batch
        sequentially
        :param batch_size: The number of samples in a minibatch.
        """
        if batch_size is None:
            batch_size = self.batch_size
        self.pathpairs = load_files(self.datadir, self.src_prefix, self.tgt_prefix, self.tn_prefix) # Search for new data with every iteration ?
        for result in iterate_dir(self.device, self.pathpairs, batch_size, self.max_decomped, self.threads):
            yield result



class TruncateDepthLoader:
    """
    A loader that truncates the read depth of the tensors to a fixed value
    Remember that the tensors are (batch, sequence, read, features) so we are truncating dimension 2
    """

    def __init__(self, loader, max_read_depth):
        self.loader = loader
        self.max_read_depth = max_read_depth
        logger.info(f"Truncating read depth to {self.max_read_depth}")

    def iter_once(self):
        for itemdict in self.loader.iter_once():
            yield {
                "read": itemdict["read"][:, :, 0:self.max_read_depth, :],
                "tgkmers": itemdict["tgkmers"],
                "tntgt": itemdict["tntgt"],
                "altmask": itemdict["altmask"],
                "log_info": itemdict["log_info"],
            }
    
    def __len__(self):
        """
        Return the number of batches available in this loader
        """
        return len(self.loader)

    def __iter__(self):
        """
        Make this loader a standard Python iterable by yielding items from iter_once
        """
        for item in self.iter_once():
            yield item

def is_lmdb_dir(datadir):
    """
    Check if the directory is an LMDB dir
    :param datadir: Directory to check
    :returns : True if the directory is an LMDB dir, False otherwise
    """
    return (Path(datadir) / "data.mdb").exists()


def make_loader(datadir: Union[str, List[str]], **kwargs):
    """
    If they supply a single string value, we test to see if it's an LMDB dir or a pre-gen dir
    If they supply a list of strings, we assume they are LMDB dirs
    :param datadir: Directory to read data from
    :param kwargs: Additional arguments to pass to the loader
    :returns : Loader object
    """
    max_read_depth = kwargs.get('max_read_depth', -1)
    # Check if we're in a distributed context
    use_distributed = (torch.distributed.is_initialized() and 
                      torch.distributed.get_world_size() > 1)
    
    if isinstance(datadir, str):
        if is_lmdb_dir(datadir):
            # Configure LMDB dataset with appropriate reader limits
            max_readers = kwargs.get('max_readers', 126)
            dataset = LMDBDataset(datadir, max_read_depth=max_read_depth, max_readers=max_readers)
            logger.info(f"Created LMDB dataset with {len(dataset)} samples")
            
            # Use DistributedSampler only in distributed context, otherwise use SequentialSampler
            if use_distributed:
                sampler = DistributedSampler(dataset)
                shuffle = False  # DistributedSampler handles shuffling
            else:
                sampler = SequentialSampler(dataset)
                shuffle = kwargs.get('shuffle', False)
            
            dataset = VarsInfoDataWrapper(dataset)
            loader = DataLoader(
                dataset, 
                batch_size=kwargs.get('batch_size'), 
                sampler=sampler,
                shuffle=shuffle,
                num_workers=kwargs.get('num_workers', 1),
                pin_memory=kwargs.get('pin_memory', True),
                drop_last=kwargs.get('drop_last', True),
                prefetch_factor=kwargs.get('prefetch_factor', 2),
                persistent_workers=kwargs.get('persistent_workers', False),  # Disable persistent workers for LMDB
                multiprocessing_context='spawn',  # Use spawn to avoid LMDB issues
            )
            return loader
        else:
            loader = PregenLoader(
                        datadir, 
                        tgt_prefix='tgkmers',
                        **kwargs)
            if max_read_depth != -1:
                loader = TruncateDepthLoader(loader, max_read_depth)
            return loader
    else:
        # Configure LMDB datasets with appropriate reader limits
        max_readers = kwargs.get('max_readers', 126)
        datasets = [LMDBDataset(d, max_read_depth=max_read_depth, max_readers=max_readers) for d in datadir]
        total_samples = sum([len(d) for d in datasets])
        logger.info(f"Created {len(datasets)} LMDB datasets with {total_samples} samples")
        concat_dataset = torch.utils.data.ConcatDataset(datasets)
        concat_dataset = VarsInfoDataWrapper(concat_dataset)
        
        # Use DistributedSampler only in distributed context, otherwise use SequentialSampler
        if use_distributed:
            sampler = DistributedSampler(concat_dataset)
            shuffle = False  # DistributedSampler handles shuffling
        else:
            sampler = SequentialSampler(concat_dataset)
            shuffle = None
        
        loader = DataLoader(
            concat_dataset, 
            batch_size=kwargs.get('batch_size'), 
            sampler=sampler,
            shuffle=shuffle,
            num_workers=kwargs.get('num_workers', 1),
            pin_memory=kwargs.get('pin_memory', True),
            drop_last=kwargs.get('drop_last', True),
            prefetch_factor=kwargs.get('prefetch_factor', 2),
            persistent_workers=kwargs.get('persistent_workers', False),  # Disable persistent workers for LMDB
            multiprocessing_context='spawn',  # Use spawn to avoid LMDB issues
        )
        return loader
    