
import logging
import sys
from collections import defaultdict
import time

import yaml
from datetime import datetime
import os
from pygit2 import Repository

import torch
from torch import nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
import torch.cuda.amp as amp

from torch.nn.parallel import DistributedDataParallel as DDP


from dnaseq2seq import vcf
from dnaseq2seq import loader
from dnaseq2seq import util
from dnaseq2seq.model import VarTransformer

LOG_FORMAT  ='[%(asctime)s] %(process)d  %(name)s  %(levelname)s  %(message)s'
formatter = logging.Formatter(LOG_FORMAT)
handler = logging.FileHandler("jovian_train.log")
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(handler)


USE_DDP = int(os.environ.get('RANK', -1)) >= 0 and os.environ.get('WORLD_SIZE') is not None
MASTER_PROCESS = (not USE_DDP) or os.environ.get('RANK') == '0'
DEVICE = None # This is set in the 'train' method


if os.getenv("ENABLE_COMET") and MASTER_PROCESS:
    logger.info("Enabling Comet.ai logging")
    from comet_ml import Experiment

    experiment = Experiment(
      api_key=os.getenv('COMET_API_KEY'),
      project_name="variant-transformer",
      workspace="brendan"
    )
else:
    experiment = None



class TrainLogger:
    """ Simple utility for writing various items to a log file CSV """

    def __init__(self, output, headers):
        self.headers = list(headers)
        if type(output) == str:
            self.output = open(output, "a")
        else:
            self.output = output
        self._write_header()


    def _write_header(self):
        self.output.write(",".join(self.headers) + "\n")
        self._flush_and_fsync()

    def _flush_and_fsync(self):
        try:
            self.output.flush()
            os.fsync()
        except:
            pass

    def log(self, items):
        assert len(items) == len(self.headers), f"Expected {len(self.headers)} items to log, but got {len(items)}"
        self.output.write(
            ",".join(str(items[k]) for k in self.headers) + "\n"
        )
        self._flush_and_fsync()



def compute_twohap_loss(preds, tgt, criterion):
    """
    Iterate over every item in the batch, and compute the loss in both configurations (under torch.no_grad())
    then swap haplotypes (dimension index 1) in the predictions if that leads to a lower loss
    Finally, re-compute loss with the new configuration for all samples and return it, storing gradients this time
    """
    # Compute losses in both configurations, and use the best
    with torch.no_grad():
        swaps = 0
        for b in range(preds.shape[0]):
            loss1 = criterion(preds[b, :, :, :].flatten(start_dim=0, end_dim=1),
                              tgt[b, :, :].flatten())
            loss2 = criterion(preds[b, :, :, :].flatten(start_dim=0, end_dim=1),
                              tgt[b, torch.tensor([1, 0]), :].flatten())

            if loss2.mean() < loss1.mean():
                preds[b, :, :, :] = preds[b, torch.tensor([1, 0]), :]
                swaps += 1

    return criterion(preds.flatten(start_dim=0, end_dim=2), tgt.flatten()), swaps


def train_n_samples(model, optimizer, criterion, tn_loss_criterion, loader_iter, num_samples, tn_loss_weight, lr_schedule=None, enable_amp=False):
    """
    Train until we've seen more than 'num_samples' from the loader, then return the loss
    """
    samples_seen = 0
    loss_sum = 0
    tn_loss_sum = 0
    model.train()
    scaler = GradScaler(enabled=enable_amp)
    start = time.perf_counter()
    samples_perf = 0
    for batch, (src, tgt_kmers, tntgt, *_) in enumerate(loader_iter):
        logger.debug("Got batch from loader...")
        tgt_kmer_idx = torch.argmax(tgt_kmers, dim=-1)
        tgt_kmers_input = tgt_kmers[:, :, :-1]
        tgt_expected = tgt_kmer_idx[:, :, 1:]
        tntgt = tntgt.float()
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_kmers_input.shape[-2]).to(DEVICE)

        optimizer.zero_grad()
        logger.debug("Forward pass...")

        with amp.autocast(enabled=enable_amp): # dtype is bfloat16 by default
            seq_preds, tn_preds = model(src, tgt_kmers_input, tgt_mask)

            logger.debug(f"Computing loss...")
            hap_loss, swaps = compute_twohap_loss(seq_preds, tgt_expected, criterion)
            tn_loss = tn_loss_criterion(tn_preds.squeeze(), tntgt)
            tn_loss_sum += tn_loss.item()
            loss = loss + tn_loss_weight * tn_loss

        scaler.scale(loss).backward()
        loss_sum += loss.item()
        #torch.nn.utils.clip_grad_norm_(model.parameters(),  1.0)
        
        # May not play nice with AMP? Dont clip gradients if we're using AMP
        if not enable_amp:
            torch.nn.utils.clip_grad_norm_(model.parameters(),  1.0)
        
        logger.debug("Stepping optimizer...")
        scaler.step(optimizer)
        scaler.update()
        lr_schedule.add_iters(src.shape[0])
        samples_perf += src.shape[0]
        if batch % 10 == 0:
            elapsed = time.perf_counter() - start
            samples_per_sec = samples_perf / elapsed
            logger.info(f"Batch {batch}  samples: {samples_seen}   loss: {loss.item():.3f}   swaps: {swaps}   samples/sec: {samples_per_sec :.2f}")
            start = time.perf_counter()
            samples_perf = 0

        if lr_schedule and batch % 10 == 0:
            lr = lr_schedule.get_lr()
            logger.info(f"LR samples seen: {lr_schedule.iters}, learning rate: {lr_schedule.get_last_lr() :.6f}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        samples_seen += src.shape[0]
        if samples_seen > num_samples:
            return loss_sum


def iter_indefinitely(loader, batch_size):
    iterations = 0
    while True:
        iterations += 1
        for items in loader.iter_once(batch_size):
            yield items
        logger.info(f"Completed iteration {iterations} of all training data")


def add_result_dicts(src, add):
    for key, subdict in add.items():
        for subkey, value in subdict.items():
            src[key][subkey] += value
    return src


def _calc_hap_accuracy(src, seq_preds, tgt, result_totals):
    # Compute val accuracy
    match = (torch.argmax(seq_preds[:, :, :].flatten(start_dim=0, end_dim=1),
                             dim=1) == tgt[:, :].flatten()
                ).float().mean()

    var_count = 0

    for b in range(src.shape[0]):
        predstr = util.kmer_preds_to_seq(seq_preds[b, :, 0:util.KMER_COUNT], util.i2s)
        tgtstr = util.kmer_idx_to_str(tgt[b, :], util.i2s)
        vc = len(list(vcf.aln_to_vars(tgtstr, predstr, chrom='X'))) # chrom arbitrary, required for building variant objs
        var_count += vc

        # Get TP, FN and FN based on reference, alt and predicted sequence.
        vartype_count = eval_prediction(util.readstr(src[b, :, 0, :]), tgtstr, seq_preds[b, :, 0:util.KMER_COUNT], counts=init_count_dict())
        result_totals = add_result_dicts(result_totals, vartype_count)

    return match, var_count, result_totals


def eval_prediction(refseqstr, altseq, predictions, counts):
    """
    Given a target sequence and two predicted sequences, attempt to determine if the correct *Variants* are
    detected from the target. This uses the vcf.align_seqs(seq1, seq2) method to convert string sequences to Variant
    objects, then compares variant objects
    :param tgt:
    :param predictions:
    :param midwidth:
    :return: Sets of TP, FP, and FN vars
    """
    known_vars = []
    for v in vcf.aln_to_vars(refseqstr, altseq, chrom='X'):
        known_vars.append(v)

    pred_vars = []
    predstr = util.kmer_preds_to_seq(predictions[:, 0:util.KMER_COUNT], util.i2s)
    for v in vcf.aln_to_vars(refseqstr, predstr, chrom='X'):
        pred_vars.append(v)

    for true_var in known_vars:
        true_var_type = util.var_type(true_var)
        if true_var in pred_vars:
            counts[true_var_type]['tp'] += 1
        else:
            counts[true_var_type]['fn'] += 1

    for detected_var in pred_vars:
        if detected_var not in known_vars:
            vartype = util.var_type(detected_var)
            counts[vartype]['fp'] += 1

    return counts



def calc_val_accuracy(loader, model, criterion, tn_criterion):
    """
    Compute accuracy (fraction of predicted bases that match actual bases),
    calculates mean number of variant counts between tgt and predicted sequence,
    and also calculates TP, FP and FN count based on reference, alt and predicted sequence.
    across all samples in the loader, using the given model, and return it
    :returns : Average model accuracy across all validation sets, vaf MSE 
    """
    model.eval()
    with torch.no_grad():
        match_sum0 = 0
        match_sum1 = 0
        result_totals0 = init_count_dict()
        result_totals1 = init_count_dict()

        var_counts_sum0 = 0
        var_counts_sum1 = 0
        tot_samples = 0
        total_batches = 0
        loss_tot = 0
        tn_loss = 0
        tn_acc = 0

        swap_tot = 0
        for src, tgt_kmers, tntgt, *_ in loader.iter_once(64):
            total_batches += 1
            tot_samples += src.shape[0]
            seq_preds, probs, tn_logits = util.predict_sequence(src, model, n_output_toks=37, device=DEVICE) # 150 // 4 = 37, this will need to be changed if we ever want to change the output length
            tn_loss += tn_criterion(tn_logits.squeeze(), tntgt.float()).item()
            tn_acc += (torch.round(torch.sigmoid(tn_logits.squeeze())) == tntgt).float().mean().item()

            #tgt_kmers = util.tgt_to_kmers(tgt[:, :, 0:truncate_seq_len]).float().to(DEVICE)
            tgt_kmer_idx = torch.argmax(tgt_kmers, dim=-1)[:, :, 1:]
            j = tgt_kmer_idx.shape[-1]
            seq_preds = seq_preds[:, :, 0:j, :] # tgt_kmer_idx might be a bit shorter if the sequence is truncated

            loss, swaps = compute_twohap_loss(seq_preds, tgt_kmer_idx, criterion)
            loss_tot += loss
            swap_tot += swaps

            midmatch0, varcount0, results_totals0 = _calc_hap_accuracy(src, seq_preds[:, 0, :, :], tgt_kmer_idx[:, 0, :], result_totals0)
            midmatch1, varcount1, results_totals1 = _calc_hap_accuracy(src, seq_preds[:, 1, :, :], tgt_kmer_idx[:, 1, :], result_totals1)
            match_sum0 += midmatch0
            match_sum1 += midmatch1

            var_counts_sum0 += varcount0
            var_counts_sum1 += varcount1
                
    return (match_sum0 / total_batches,
            match_sum1 / total_batches,
            var_counts_sum0 / tot_samples,
            var_counts_sum1 / tot_samples,
            result_totals0, result_totals1,
            loss_tot,
            swap_tot,
            tn_loss,
            tn_acc / tot_samples)


def safe_compute_ppav(results0, results1, key):
    try:
        ppa = (results0[key]['tp'] + results1[key]['tp']) / (
                results0[key]['tp'] + results1[key]['tp'] + results0[key]['fn'] + results1[key]['fn'])
    except ZeroDivisionError:
        ppa = 0
    try:
        ppv = (results0[key]['tp'] + results1[key]['tp']) / (
                results0[key]['tp'] + results1[key]['tp'] + results0[key]['fp'] + results1[key]['fp'])
    except ZeroDivisionError:
        ppv = 0

    return ppa, ppv

def load_model(modelconf, ckpt):
    statedict = None
    if ckpt is not None:
        if 'model' in ckpt:
            statedict = ckpt['model']
            new_state_dict = {}
            for key in statedict.keys():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = statedict[key]
            statedict = new_state_dict
        else:
            statedict = ckpt

        if 'conf' in ckpt:
            logger.warning(f"Found model conf AND a checkpoint with model conf - using the model params from checkpoint")
            modelconf = ckpt['conf']

    logger.info(f"Model conf: {modelconf}")
    model = VarTransformer(read_depth=modelconf['max_read_depth'],
                           feature_count=modelconf['feats_per_read'],
                           kmer_dim=util.FEATURE_DIM,  # Number of possible kmers
                           n_encoder_layers=modelconf['encoder_layers'],
                           n_decoder_layers=modelconf['decoder_layers'],
                           embed_dim_factor=modelconf['embed_dim_factor'],
                           encoder_attention_heads=modelconf['encoder_attention_heads'],
                           decoder_attention_heads=modelconf['decoder_attention_heads'],
                           d_ff=modelconf['dim_feedforward'],
                           device=DEVICE)

    
    model_tot_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_tot_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    decoder_tot_params = 2 * sum(p.numel() for p in model.decoder0.parameters() if p.requires_grad)
    
    logger.info(f"Decoder0: {model.decoder0}")
    logger.info(f"Creating model with {model_tot_params} trainable params")
    logger.info(f"Encoder tot params: {encoder_tot_params} ")
    logger.info(f"Decoder tot params: {decoder_tot_params} ")

    if statedict is not None:
        logger.info(f"Initializing model weights from state dict")
        model.load_state_dict(statedict)
    
    #logger.info("Turning OFF gradient computation for fc1 and fc2 embedding layers")
    #model.fc1.requires_grad_(False)
    #model.fc2.requires_grad_(False)
    
    logger.info("Compiling model...")
    model = torch.compile(model)
    
    if USE_DDP:
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        logger.info(f"Creating DDP model with rank {rank} and device_id: {device_id}")
        model = model.to(device_id)
        model = DDP(model, device_ids=[device_id])
    else:
        model = model.to(DEVICE)

    model.train()
    return model

def train_epochs(model,
                 optimizer,
                 epochs,
                 dataloader,
                 scheduler,
                 checkpoint_freq=0,
                 model_dest=None,
                 val_dir=None,
                 batch_size=64,
                 xtra_checkpoint_items={},
                 samples_per_epoch=10000,
):


    criterion = nn.NLLLoss()
    tn_loss_criterion = torch.nn.BCEWithLogitsLoss()
    tn_loss_weight = 0.5 # ? Put this on a schedule?

    trainlogpath = str(model_dest).replace(".model", "").replace(".pt", "") + "_train.log"
    logger.info(f"Training log data will be saved at {trainlogpath}")

    swaps = 0
    trainlogger = TrainLogger(trainlogpath, [
            "epoch", "trainingloss", "val_accuracy",
            "mean_var_count", "ppa_dels", "ppa_ins", "ppa_snv",
            "ppv_dels", "ppv_ins", "ppv_snv", "learning_rate", "tn_loss", "epochtime",
    ])


    if val_dir:
        logger.info(f"Using validation data in {val_dir}")
        val_loader = loader.PregenLoader(device=DEVICE, datadir=val_dir, max_decomped_batches=4, threads=8, tgt_prefix="tgkmers")
    else:
        logger.info(f"No val. dir. provided retaining a few training samples for validation")
        valpaths = dataloader.retain_val_samples(fraction=0.05)
        val_loader = loader.PregenLoader(device=DEVICE, datadir=None, pathpairs=valpaths, threads=4, tgt_prefix="tgkmers")
        logger.info(f"Pulled {len(valpaths)} samples to use for validation")

    try:
        sample_iter = iter_indefinitely(dataloader, batch_size)
        for epoch in range(epochs):
            starttime = datetime.now()
            assert samples_per_epoch > 0, "Must have positive number of samples per epoch"
            loss = train_n_samples(model,
                              optimizer,
                              criterion,
                              tn_loss_criterion,
                              sample_iter,
                              samples_per_epoch,
                              tn_loss_weight,
                              scheduler)

            elapsed = datetime.now() - starttime

            if MASTER_PROCESS:
                acc0, acc1, var_count0, var_count1, results0, results1, val_loss, swaps, tn_loss, tn_acc = calc_val_accuracy(val_loader, model, criterion)

                ppa_dels, ppv_dels = safe_compute_ppav(results0, results1, 'del')
                ppa_ins, ppv_ins = safe_compute_ppav(results0, results1, 'ins')
                ppa_snv, ppv_snv = safe_compute_ppav(results0, results1, 'snv')

                logger.info(f"Epoch {epoch} Secs: {elapsed.total_seconds():.2f} lr: {scheduler.get_last_lr():.5f} loss: {loss:.4f} val acc: {acc0:.3f} / {acc1:.3f}  ppa: {ppa_snv:.3f} / {ppa_ins:.3f} / {ppa_dels:.3f}  ppv: {ppv_snv:.3f} / {ppv_ins:.3f} / {ppv_dels:.3f} swaps: {swaps}")
                trainlogger.log({
                    "epoch": epoch,
                    "trainingloss": loss,
                    "val_accuracy": acc0.item() if isinstance(acc0, torch.Tensor) else acc0,
                    "mean_var_count": var_count0,
                    "ppa_snv": ppa_snv,
                    "ppa_ins": ppa_ins,
                    "ppa_dels": ppa_dels,
                    "ppv_ins": ppv_ins,
                    "ppv_snv": ppv_snv,
                    "ppv_dels": ppv_dels,
                    "learning_rate": scheduler.get_last_lr(),
                    "tn_loss": tn_loss,
                    "epochtime": elapsed.total_seconds(),
                })
            
            if experiment:
                experiment.log_metrics({
                    "epoch": epoch,
                    "trainingloss": loss,
                    "validation_loss": val_loss,
                    "accuracy/val_acc_hap0": acc0,
                    "accuracy/val_acc_hap1": acc1,
                    "accuracy/var_count0": var_count0,
                    "accuracy/var_count1": var_count1,
                    "accuracy/ppa dels": ppa_dels,
                    "accuracy/ppa ins": ppa_ins,
                    "accuracy/ppa snv": ppa_snv,
                    "accuracy/ppv dels": ppv_dels,
                    "accuracy/ppv ins": ppv_ins,
                    "accuracy/ppv snv": ppv_snv,
                    "accuracy/tn": tn_acc,
                    "tn_loss": tn_loss,
                    "learning_rate": scheduler.get_last_lr(),
                    "hap_swaps": swaps,
                    "epochtime": elapsed.total_seconds(),
                }, step=epoch)

            if MASTER_PROCESS and epoch > -1 and checkpoint_freq > 0 and (epoch % checkpoint_freq == 0):
                modelparts = str(model_dest).rsplit(".", maxsplit=1)
                checkpoint_name = modelparts[0] + f"_epoch{epoch}." + modelparts[1]
                logger.info(f"Saving model state dict to {checkpoint_name}")
                m = model.module if (isinstance(model, nn.DataParallel) or isinstance(model, DDP)) else model
                ckpt_data = {
                    'model': m.state_dict(),
                    'conf': xtra_checkpoint_items,
                    'opt': optimizer.state_dict(),
                }
                torch.save(ckpt_data, checkpoint_name)

        logger.info(f"Training completed after {epoch} epochs")
    except KeyboardInterrupt:
        pass

    if model_dest is not None:
        logger.info(f"Saving model state dict to {model_dest}")
        m = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(m.to('cpu').state_dict(), model_dest)


def load_train_conf(confyaml):
    logger.info(f"Loading configuration from {confyaml}")
    conf = yaml.safe_load(open(confyaml).read())
    assert 'reference' in conf, "Expected 'reference' entry in training configuration"
    # assert 'data' in conf, "Expected 'data' entry in training configuration"
    return conf


def init_count_dict():
    return {
        'del': defaultdict(int),
        'ins': defaultdict(int),
        'snv': defaultdict(int),
        'mnv': defaultdict(int),
    }

def set_comet_conf(model_tot_params, **kwargs):
    """ Set various config params for logging in WandB / Comet"""
    # get git branch info for logging
    git_repo = Repository(os.path.abspath(__file__))
    # what to log in wandb
    run_config_params = dict(
        learning_rate=kwargs.get('init_learning_rate'),
        embed_dim_factor=kwargs.get('embed_dim_factor'),
        feats_per_read=kwargs.get('feats_per_read'),
        batch_size=kwargs.get('batch_size'),
        read_depth=kwargs.get('max_read_depth'),
        encoder_attn_heads=kwargs.get('encoder_attention_heads'),
        decoder_attn_heads=kwargs.get('decoder_attention_heads'),
        transformer_dim=kwargs.get('dim_feedforward'),
        encoder_layers=kwargs.get('encoder_layers'),
        decoder_layers=kwargs.get('decoder_layers'),
        git_branch=git_repo.head.name,
        git_target=git_repo.head.target,
        model_param_count=model_tot_params,
        git_last_commit=next(git_repo.walk(git_repo.head.target)).message,
        samples_per_epoch=kwargs.get('samples_per_epoch'),
        commandline=' '.join(sys.argv),
    )

    # change working dir so wandb finds git repo info
    current_working_dir = os.getcwd()
    git_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(git_dir)

    experiment.log_parameters({
        "config": run_config_params,
        "dir": current_working_dir,
    })
    experiment.set_name(kwargs.get('run_name'))

    # back to correct working dir
    os.chdir(current_working_dir)


def load_conf(conf_file, **kwargs):
    with open(conf_file) as fh:
        conf = yaml.safe_load(fh)
    conf.update((k,v) for k,v in kwargs.items() if v is not None)
    return conf


def train(output_model, **kwargs):
    """
    Conduct a training run and save the trained parameters (statedict) to output_model
    :param config: Path to config yaml
    :param output_model: Path to save trained params to
    :param input_model: Start training with params from input_model
    :param epochs: How many passes over training data to conduct
    """

    kwargs = load_conf(kwargs.get('config'), **kwargs)
    run_name = kwargs.get("run_name", "training_run")
    dest = f"{run_name}_training_conf.yaml"
    with open(dest, "w") as fh:
        fh.write(yaml.dump(kwargs) + "\n")

    global DEVICE

    if USE_DDP:
        logger.info(f"Using DDP: Master addr: {os.environ['MASTER_ADDR']}, port: {os.environ['MASTER_PORT']}, global rank: {os.environ['RANK']}, world size: {os.environ['WORLD_SIZE']}") 
        if MASTER_PROCESS:
            logger.info(f"Master process is {os.getpid()}")
        else:
            logger.info(f"Process {os.getpid()} is NOT the master")
        logger.info(f"Number of available CUDA devices: {torch.cuda.device_count()}")
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        DEVICE = f"cuda:{device_id}"
        logger.info(f"Setting cuda device to {DEVICE}")
        torch.cuda.set_device(DEVICE)
        logger.info(f"DDP [{os.getpid()}] CUDA device {DEVICE} name: {torch.cuda.get_device_name()}")
    else:
        logger.info(f"Configuring for non-DDP: torch device: {DEVICE}")
        if 'cuda' in str(DEVICE):
            for idev in range(torch.cuda.device_count()):
                logger.info(f"CUDA device {idev} name: {torch.cuda.get_device_name({idev})}")
        DEVICE = torch.device("cuda") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")
    
    logger.info(f"Using pregenerated training data from {kwargs.get('datadir')}")

    dataloader = loader.PregenLoader(DEVICE,
                                     kwargs.get("datadir"),
                                     threads=kwargs.get('threads'),
                                     max_decomped_batches=kwargs.get('max_decomp_batches'),
                                     tgt_prefix="tgkmers")

    if kwargs.get('input_model'):
        ckpt = torch.load(kwargs.get("input_model"), map_location=DEVICE)
    else:
        ckpt = None
    model = load_model(kwargs['model'], ckpt)

    model_tot_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model total parameter count: {model_tot_params}")
    if experiment:
        set_comet_conf(model_tot_params, **kwargs)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=kwargs.get('learning_rate', 0.001),
        betas=(0.9, 0.999)
    )
    if ckpt is not None and ckpt.get('opt') is not None:
        logger.info("Loading optimizer state dict from checkpoint")
        optimizer.load_state_dict(ckpt.get('opt'))

    init_learning_rate = kwargs.get('learning_rate', 0.0001)
    scheduler = util.WarmupCosineLRScheduler(
        max_lr=init_learning_rate,
        min_lr=kwargs.get('min_learning_rate', init_learning_rate / 5.0),
        warmup_iters=kwargs.get('lr_warmup_iters', 1e6),
        lr_decay_iters=kwargs.get('lr_decay_iters', 20e6),
    )

    train_epochs(model,
                 optimizer,
                 kwargs.get('epochs'),
                 dataloader,
                 scheduler=scheduler,
                 model_dest=output_model,
                 checkpoint_freq=kwargs.get('checkpoint_freq', 10),
                 val_dir=kwargs.get('val_dir'),
                 batch_size=kwargs.get("batch_size"),
                 samples_per_epoch=kwargs.get('samples_per_epoch'),
                 xtra_checkpoint_items=kwargs['model'],
                 )

