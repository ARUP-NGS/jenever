from this import d
import time
import json
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Any, List, Optional, Tuple
from torch import Tensor
import pysam
import logging
from dnaseq2seq import util
from dnaseq2seq.calling import buildclf
from dnaseq2seq.model import VarTransformer
from dnaseq2seq.calling import stage

from dnaseq2seq.calling import vcf
from dnaseq2seq.calling import vcfwriter
from dnaseq2seq.calling.vcf import Variant

@dataclass
class HaplotypeVars:
    hap0: Dict[Tuple[str, int, int], List[Variant]] = field(default_factory=dict)
    hap1: Dict[Tuple[str, int, int], List[Variant]] = field(default_factory=dict)


logger = logging.getLogger(__name__)

def load_model(model_path, device):
    """
    Create the VariantTransformer model using params / config settings from the given path
    Model is compiled with torch.compile and set to eval mode
    :returns: VariantTransformer model with parameters loaded
    """
    model_info = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in model_info:
        statedict = model_info['model_state_dict']
    else:
        statedict = model_info['model']

    modelconf = model_info['conf']
    new_state_dict = {}
    for key in statedict.keys():
      new_key = key.replace('_orig_mod.', '')
      new_state_dict[new_key] = statedict[key]
    statedict = new_state_dict

    logger.info(f"Loading model configuration: {modelconf}")
    model = VarTransformer(read_depth=modelconf['max_read_depth'],
                           feature_count=modelconf['feats_per_read'],
                           kmer_dim=util.FEATURE_DIM,  # Number of possible kmers
                           n_encoder_layers=modelconf['encoder_layers'],
                           n_decoder_layers=modelconf['decoder_layers'],
                           embed_dim_factor=modelconf['embed_dim_factor'],
                           decoder_embed_dim=modelconf['decoder_embed_dim'],
                           encoder_attention_heads=modelconf['encoder_attention_heads'],
                           decoder_attention_heads=modelconf['decoder_attention_heads'],
                           d_ff=modelconf['dim_feedforward'],
                           device=device)

    model.load_state_dict(statedict, strict=False)
    model.eval()
    model.to(device)
    
    return model, modelconf



class VarHapCaller:

    def __init__(self, model_path: str, ref_path: str, max_batch_size: int, device: torch.device):
        self.model_path = model_path
        self.ref_path = ref_path
        self.max_batch_size = max_batch_size
        self.model = None
        self.classifier = None
        self.data_buffer = []
        self.windows_buffered = 0
        self.bp_processed = 0
        self.call_time = 0
        self.resolve_haplotypes_time = 0
        self.device = device
    
    def _init_model(self):
        self.reference = pysam.FastaFile(self.ref_path)
        self.model, _ = load_model(self.model_path, self.device)
        self.model.eval()
        torch.set_num_threads(8)

    def set_device(self, device: torch.device):
        self.device = device
        if self.model is not None:
            self.model.to(device)

    def worker_init(self, worker_index: int):
        self.set_device(torch.device(f"cuda:{worker_index}"))

    def __call__(self, data: Tensor):
        data['encoded_pileup'] = data['encoded_pileup'].to(self.device)
        self.data_buffer.append(data)
        self.windows_buffered += data['encoded_pileup'].shape[0]
        logger.debug(f"New data with {data['encoded_pileup'].shape[0]} winodws, buffer length: {len(self.data_buffer)}, total windows buffered: {self.windows_buffered}")
        # We accumulate until we have slightly less than the real max batch size, this is because
        # later code will split the data into max-batch-size chunks, so if we have more than max-batch-size it'll
        # make 2 different batches, one big and one small, which is super inefficient. If we just trigger on slightly
        # less than the real max batch size, we'll make sure we always have one big batch
        if self.windows_buffered < (self.max_batch_size - 10):
            return stage.SkipResult()
        
        if self.model is None:
            self._init_model()
            
        datas = self.data_buffer #sorted(self.data_buffer, key=priority_func) # Sorting data chunks here helps ensure sorted output

        bp = sum(d['region'][2] - d['region'][1] for d in datas)
        self.bp_processed += bp

        logger.debug(
            f"Calling variants up to {datas[len(datas) // 2]['region'][0]}:{datas[-1]['region'][1]}-{datas[-1]['region'][2]}, total bp processed: {round(self.bp_processed / 1e6, 3)}MB"
        )
        logger.debug(f"Calling variants with {self.windows_buffered} windows")
        hapvars, call_time, resolve_haplotypes_time = call_multi_paths(datas, self.model, self.reference, max_batch_size=self.max_batch_size, device=self.device)
        logger.info(f"Called variants in {call_time :.3f} seconds")
        self.call_time += call_time
        self.resolve_haplotypes_time += resolve_haplotypes_time
        self.data_buffer = []
        self.windows_buffered = 0
        return hapvars


    def flush(self):
        if self.data_buffer:
            hapvars, call_time, resolve_haplotypes_time = call_multi_paths(self.data_buffer, self.model, self.reference, max_batch_size=self.max_batch_size, device=self.device)
            self.call_time += call_time
            self.resolve_haplotypes_time += resolve_haplotypes_time
            logger.info(f"Flushing last hap vars, datavars length: {len(self.data_buffer)}, windows buffered: {self.windows_buffered}")
            logger.info(f"Time to call: {self.call_time :.3f} seconds")
            logger.info(f"Time to resolve haplotypes: {self.resolve_haplotypes_time :.3f} seconds")
            return hapvars
        
        logger.info("Flushing 0 remaining records, exiting")
        return stage.SkipResult()


@torch.inference_mode()
def call_multi_paths(datas, model, reference, max_batch_size, device):
    """
    Concat a list of 'datas' objects, which contain encoded_pileups, batch offsets, and then call variants over all of them
    
    :return : List of VCFRecords for variants found
    """
    allencoded, batch_start_pos, batch_regions = merge_datas(datas)
    haplotype_vars, call_time, resolve_haplotypes_time = call_and_merge(allencoded, batch_start_pos, batch_regions, model, reference, max_batch_size, device)

    return haplotype_vars, call_time, resolve_haplotypes_time

def call_and_merge(batch, batch_offsets, regions, model, reference, max_batch_size, device):
    """
    Generate haplotypes for the batch, identify variants in each, and then 'merge genotypes' across the overlapping
    windows with the ad-hoc algo in the resolve_haplotypes function. This also filters out any variants not found
    in the 'regions' tuple

    Note that this function contains additional, unused logic to divide batch up into smaller regions and send those
    through the model individually. This might be useful if different regions require very different numbers of predicted
    tokens, for instance. However, since the time spent in forward passes of the model are almost constant in batch size
    this doesn't offer much speedup in a naive implementation (but since we don't do KV caching during decoding the cost
    for generating additional tokens increases linearly in the total token count... so maybe it makes sense to do this?)

    :param batch: Tensor of encoded regions
    :param batch_offsets: List of start positions, must have same length as batch.shape[0]
    :param regions: List of genomic regions, must have length equal to batch_offsets
    :param model: Model for haplotype prediction
    :param reference: pysam.FastaFile with reference genome
    :param max_batch_size: Maximum number of regions to call in one go
    :returns Dict[(chrom, start, end)] -> List[proto vars] for the variants found in each region
    """
    logger.debug(f"Predicting batch of size {batch.shape[0]} for chrom {regions[0][0]}:{regions[0][1]}-{regions[-1][2]}")
    dists = np.array([r[2] - bo for r, bo in zip(regions, batch_offsets)])
    byregion = defaultdict(list)
    n_output_toks = min(150 // util.TGT_KMER_SIZE - 1, max(dists) // util.TGT_KMER_SIZE + 1)
    start_time = time.perf_counter()
    batchvars = call_batch(batch, batch_offsets, regions, model, reference, n_output_toks, max_batch_size=max_batch_size, device=device)
    for region, bvars in zip(regions, batchvars):
        byregion[region].append(bvars)
    
    call_time = time.perf_counter() - start_time
    hap0 = defaultdict(list)
    hap1 = defaultdict(list)
    start_time = time.perf_counter()
    for region, rvars in byregion.items():
        chrom, start, end = region
        h0, h1 = resolve_haplotypes(rvars)
        for k, v in h0.items():
            if start <= v[0].pos < end:
                hap0[k].extend(v)
        for k, v in h1.items():
            if start <= v[0].pos < end:
                hap1[k].extend(v)
    end_time = time.perf_counter()
    resolve_haplotypes_time = end_time - start_time
    return HaplotypeVars(hap0=hap0, hap1=hap1), call_time, resolve_haplotypes_time

 
def resolve_haplotypes(genos):
    """
    Rearrange variants across haplotypes with a heuristic algorithm to minimize the number of conflicting
    predictions.
    Genos is a list of two-tuples of Variant objects representing the outputs of calling from multiple overlapping windows
    like this:
    [ (hap0 variants from window 1, hap1 variants from window 1),
      (hap0 variants from window 2, hap1 variants from window 2),
      ...
    ]
    The goal is to rearrange variants across haplotypes to minimize conflicts
    :returns : Two-tuple of dicts of variants, each representing one haplotype
    """
    # All unique variant keys, sorted by pos
    allvars = sorted(list(v for g in genos for v in g[0]) + list(v for g in genos for v in g[1]), key=lambda v: v.pos)
    allkeys = set()
    varsnodups = []
    for v in allvars:
        if v.key not in allkeys:
            allkeys.add(v.key)
            varsnodups.append(v)

    results = [[], []]

    prev_het = None
    prev_het_index = None

    # Loop over every unique variant and decide which haplotype to put it on
    for p in varsnodups:
        homcount = 0
        hetcount = 0
        for g in genos:
            a = p.key in [v.key for v in g[0]]
            b = p.key in [v.key for v in g[1]]
            if a and b:
                homcount += 1
            elif a or b:
                hetcount += 1
        if homcount > hetcount:
            results[0].append(p)
            results[1].append(p)
        elif prev_het is None:
            results[0].append(p) # No previous hets, so just add it to hap0
            prev_het = p
            prev_het_index = 0
        else:
            # There was a previous het variant, so figure out where this new one should go
            # determine if p should be in cis or trans with prev_het
            cis = 0
            trans = 0
            for g in genos:
                g0keys = [v.key for v in g[0]]
                g1keys = [v.key for v in g[1]]
                if (p.key in g0keys and prev_het.key in g0keys) or (p.key in g1keys and prev_het.key in g1keys):
                    cis += 1
                elif (p.key in g0keys and prev_het.key in g1keys) or (p.key in g1keys and prev_het.key in g0keys):
                    trans += 1
            if trans >= cis: # If there's a tie, assume trans. This covers the case where cis==0 and trans==0, because trans is safer
                results[1 - prev_het_index].append(p)
                prev_het = p
                prev_het_index = 1 - prev_het_index
            else:
                results[prev_het_index].append(p)
                prev_het = p
                prev_het_index = prev_het_index

    # Build dictionaries with correct haplotypes...
    allvars0 = dict()
    allvars1 = dict()
    for v in results[0]:
        allvars0[v.key] = [t for t in allvars if t.key == v.key]
    for v in results[1]:
        allvars1[v.key] = [t for t in allvars if t.key == v.key]
    return allvars0, allvars1


def _call_safe(encoded_reads, model, n_output_toks, max_batch_size, device, enable_amp=True):
    """
    Predict the sequence for the encoded reads, but dont submit more than 'max_batch_size' samples
    at once
    """
    seq_preds = None
    cls_preds = None
    probs = None
    start = 0
    
    while start < encoded_reads.shape[0]:
        end = min(encoded_reads.shape[0]+1, start + max_batch_size)
        logger.debug(f"Calling batch of size {end - start}")
        with torch.amp.autocast(device_type='cuda', enabled=enable_amp):
            preds, prbs, clspred, hap0_ref_pred, hap1_ref_pred, hap0_hap1_pred = util.predict_sequence(encoded_reads[start:end, :, :, :].to(device).float(), model,
                                            n_output_toks=n_output_toks, device=device)
        
        clspred = clspred.squeeze(-1)
        assert len(clspred) == preds.shape[0]
        
        if seq_preds is None:
            seq_preds = preds
            cls_preds = clspred
        else:
            seq_preds = torch.concat((seq_preds, preds), dim=0)
            cls_preds = torch.concat((cls_preds, clspred), dim=0)
            
        if probs is None:
            probs = prbs.detach().cpu().numpy()
        else:
            probs = np.concatenate((probs, prbs.detach().cpu().numpy()), axis=0)
        start += max_batch_size
    
    return seq_preds, probs, cls_preds.cpu().numpy()


def call_batch(encoded_reads, offsets, regions, model, reference, n_output_toks, max_batch_size, device):
    """
    Call variants in a batch (list) of regions, by running a forward pass of the model and
    then aligning the predicted sequences to the reference genome and picking out any
    mismatching parts
    :returns : List of variants called in both haplotypes for every item in the batch as a list of 2-tuples
    """
    assert encoded_reads.shape[0] == len(regions), f"Expected the same number of reads as regions, but got {encoded_reads.shape[0]} reads and {len(regions)}"
    assert len(offsets) == len(regions), f"Should be as many offsets as regions, but found {len(offsets)} and {len(regions)}"

    seq_preds, probs, clspred = _call_safe(encoded_reads, model, n_output_toks, max_batch_size, device)

    assert seq_preds.shape[0] == clspred.shape[0], f"BEFORE: Predictions and cls preds are not the same size! seq_preds shape: {seq_preds.shape} clspred.shape: {clspred.shape}, regions: {regions}"

    # convert clspred logits to probabilities
    clspred = np.exp(clspred)
    
    assert seq_preds.shape[0] == clspred.shape[0], f"AFTER: Predictions and cls preds are not the same size! seq_preds shape: {seq_preds.shape} clspred.shape: {clspred.shape}, regions: {regions}"

    calledvars = []
    for offset, (chrom, start, end), b in zip(offsets, regions, range(len(seq_preds))):
        hap0_t, hap1_t = seq_preds[b, 0, :, :], seq_preds[b, 1, :, :]
        hap0 = util.kmer_preds_to_seq(hap0_t, util.i2s)
        hap1 = util.kmer_preds_to_seq(hap1_t, util.i2s)
        probs0 = np.exp(util.expand_to_bases(probs[b, 0, :]))
        probs1 = np.exp(util.expand_to_bases(probs[b, 1, :]))

        try:
            tnpred = clspred[b].item() # This will need to change if clspred returns more than one value per region
        except Exception as ex:
            logger.error(f"Exception accessing clspred, index is: {b}, clspred shape: {clspred.shape}, seq_pred length: {len(seq_preds)}")
            raise ex

        refseq = reference.fetch(chrom, offset, offset + len(hap0))
        vars_hap0 = list(v for v in vcf.aln_to_vars(refseq, hap0, chrom, offset, probs=probs0) if start <= v.pos <= end)
        vars_hap1 = list(v for v in vcf.aln_to_vars(refseq, hap1, chrom, offset, probs=probs1) if start <= v.pos <= end)
        for v in vars_hap0 + vars_hap1:
            v.tnpred = tnpred
        #print(f"Offset: {offset}\twindow {start}-{end} frame: {start % 4} hap0: {vars_hap0}\n       hap1: {vars_hap1}")
        #calledvars.append((vars_hap0, vars_hap1))
        calledvars.append((vars_hap0[0:5], vars_hap1[0:5]))

    return calledvars


def merge_datas(datas):
    """
    Concat a list of datas dictionaries into a single tensor and lists of regions, start positions
    """
    batch_encoded = []
    batch_start_pos = []
    batch_regions = []

    for data in datas:
        # Load the data, parsing location + encoded data from file
        chrom, start, end = data['region']
        batch_encoded.append(data['encoded_pileup'])
        batch_start_pos.extend(data['start_positions'])
        batch_regions.extend((chrom, start, end) for _ in range(len(data['start_positions'])))
    
    if len(batch_encoded) > 1:
        allencoded = torch.concat(batch_encoded, dim=0)
    else:
        allencoded = batch_encoded[0]

    return allencoded, batch_start_pos, batch_regions



