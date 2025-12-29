
import logging
import sys
import time

import yaml
from datetime import datetime
import os
from pathlib import Path
from pygit2 import Repository

import torch
from torch import nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP

from dnaseq2seq.calling import vcf
from dnaseq2seq.training import loader
from dnaseq2seq import util
from dnaseq2seq.model import VarTransformer
from dnaseq2seq.training import loggers
from dnaseq2seq.training.modelcheckpointer import CheckpointManager
from dnaseq2seq.training.evalpreds import calc_val_accuracy, safe_compute_ppav, compute_twohap_loss

LOG_FORMAT  ='[%(asctime)s] %(process)d  %(name)s  %(levelname)s %(funcName)s: l.%(lineno)d  %(message)s'
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


def train_n_samples(model, optimizer, criterion, loader_iter, num_samples, lr_schedule=None, enable_amp=False):
    """
    Train until we've seen more than 'num_samples' from the loader, then return the loss
    """
    samples_seen = 0
    loss_sum = 0
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=enable_amp)
    start = time.perf_counter()
    samples_perf = 0
    tn_criterion = nn.BCEWithLogitsLoss()
    hap0_ref_criterion = nn.BCEWithLogitsLoss()
    hap1_ref_criterion = nn.BCEWithLogitsLoss()
    hap0_hap1_criterion = nn.BCEWithLogitsLoss()
    tn_loss_weight = 0.1
    for batch, data in enumerate(loader_iter):
        src = data["read"].float().to(DEVICE)
        tgt_kmers = data["tgkmers"].long().to(DEVICE)
        tgt_cls = data["tntgt"].float().to(DEVICE)
        hap0_ref_match = data["hap0_ref_match"].float().to(DEVICE)
        hap1_ref_match = data["hap1_ref_match"].float().to(DEVICE)
        hap0_hap1_match = data["hap0_hap1_match"].float().to(DEVICE)
        logger.debug("Got batch from loader...")
        tgt_kmer_idx = torch.argmax(tgt_kmers, dim=-1)
        tgt_kmers_input = tgt_kmers[:, :, :-1]
        tgt_expected = tgt_kmer_idx[:, :, 1:]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_kmers_input.shape[-2]).to(DEVICE)

        optimizer.zero_grad()
        logger.debug("Forward pass...")

        with torch.amp.autocast(device_type='cuda', enabled=enable_amp): # dtype is bfloat16 by default
            seq_preds, cls_pred, hap0_ref_pred, hap1_ref_pred, hap0_hap1_pred = model(src, tgt_kmers_input, tgt_mask)

            logger.debug(f"Computing loss...")
            loss, swaps = compute_twohap_loss(seq_preds, tgt_expected, criterion)

            tnloss = tn_criterion(cls_pred.squeeze(1), tgt_cls)
            hap0_ref_loss = hap0_ref_criterion(hap0_ref_pred.squeeze(1), hap0_ref_match)
            hap1_ref_loss = hap1_ref_criterion(hap1_ref_pred.squeeze(1), hap1_ref_match)
            hap0_hap1_loss = hap0_hap1_criterion(hap0_hap1_pred.squeeze(1), hap0_hap1_match)
            loss = loss + tn_loss_weight * (tnloss +  hap0_ref_loss +  hap1_ref_loss +  hap0_hap1_loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        loss_sum += loss.item()

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


def iter_indefinitely(loader):
    iterations = 0
    while True:
        iterations += 1
        for items in loader:
            yield items
        logger.info(f"Completed iteration {iterations} of all training data")


def load_fix_encoder(model, ckpt):
    ckpt = torch.load(ckpt, map_location=DEVICE)
    statedict = ckpt['model']
    new_state_dict = {}
    for key in statedict.keys():
        new_key = key.replace('_orig_mod.', '')
        new_state_dict[new_key] = statedict[key]
    statedict = new_state_dict

    encoder_state_dict = {k: v for k, v in statedict.items() if k.startswith("encoder.")}
    fc1_state_dict = {k: v for k, v in statedict.items() if k.startswith("fc1.")}
    fc2_state_dict = {k: v for k, v in statedict.items() if k.startswith("fc2.")}

    model.encoder.load_state_dict(encoder_state_dict, strict=False)
    model.fc1.load_state_dict(fc1_state_dict, strict=False)
    model.fc2.load_state_dict(fc2_state_dict, strict=False)

    for param in model.encoder.parameters():
        param.requires_grad = False

    for param in model.fc1.parameters():
        param.requires_grad = False

    for param in model.fc2.parameters():
        param.requires_grad = False

    return model


def load_model(modelconf, ckpt):
    statedict = None
    if ckpt is not None:
        if 'model_state_dict' in ckpt:
            statedict = ckpt['model_state_dict']
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
    model = VarTransformer(read_depth=modelconf.get('max_read_depth', 150),
                           feature_count=modelconf['feats_per_read'],
                           kmer_dim=util.FEATURE_DIM,  # Number of possible kmers
                           n_encoder_layers=modelconf['encoder_layers'],
                           n_decoder_layers=modelconf['decoder_layers'],
                           embed_dim_factor=modelconf['embed_dim_factor'],
                           encoder_attention_heads=modelconf['encoder_attention_heads'],
                           decoder_attention_heads=modelconf['decoder_attention_heads'],
                           decoder_embed_dim=modelconf['decoder_embed_dim'],
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
    
    #logger.info("Compiling model...")
    #model = torch.compile(model)
    
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
                 val_loader,
                 scheduler,
                 checkpoint_freq=0,
                 model_dest=None,
                 xtra_checkpoint_items={},
                 samples_per_epoch=10000,
):


    criterion = nn.NLLLoss()

    trainlogpath = str(model_dest).replace(".model", "").replace(".pt", "") + "_train.log"
    logger.info(f"Training log data will be saved at {trainlogpath}")

    swaps = 0
    trainlogger = loggers.TrainLogger(trainlogpath, [
            "epoch", "trainingloss", "val_accuracy",
            "mean_var_count", "ppa_dels", "ppa_ins", "ppa_snv",
            "ppv_dels", "ppv_ins", "ppv_snv", "learning_rate", "epochtime",
    ])

    model_save_dir = Path(model_dest).parent
    model_save_prefix = Path(model_dest).stem
    checkpointer = CheckpointManager(model=unwrap_model(model),
                                save_prefix=model_save_prefix,
                                save_dir=model_save_dir,
                                minimize=True,
                                max_checkpoints=5)

    try:
        sample_iter = iter_indefinitely(dataloader)
        for epoch in range(epochs):
            starttime = datetime.now()
            assert samples_per_epoch > 0, "Must have positive number of samples per epoch"
            loss = train_n_samples(model,
                              optimizer,
                              criterion,
                              sample_iter,
                              samples_per_epoch,
                              scheduler,
                              enable_amp=True)

            elapsed = datetime.now() - starttime

            dist.barrier()

            # This runs on every process, to avoid communication timeouts when there are lots of validation samples
            val_metrics = calc_val_accuracy(val_loader, model, criterion, DEVICE)
            acc0, acc1 = val_metrics["acc_hap0"], val_metrics["acc_hap1"]
            var_count0, var_count1 = val_metrics["var_count_hap0"], val_metrics["var_count_hap1"]
            results0, results1 = val_metrics["results_hap0"], val_metrics["results_hap1"]
            val_loss, swaps = val_metrics["val_loss"], val_metrics["swap_count"]
            tn_prec, tn_recall, tn_f1 = val_metrics["tn_precision"], val_metrics["tn_recall"], val_metrics["tn_f1"]

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
                "epochtime": elapsed.total_seconds(),
            })
            
            if MASTER_PROCESS and experiment:
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
                    "learning_rate": scheduler.get_last_lr(),
                    "hap_swaps": swaps,
                    "epochtime": elapsed.total_seconds(),
                    "tn_stats/tn_precision": tn_prec,
                    "tn_stats/tn_recall": tn_recall,
                    "tn_stats/tn_f1": tn_f1,
                    "tn_stats/hap0_ref_f1": val_metrics["hap0_ref_f1"],
                    "tn_stats/hap1_ref_f1": val_metrics["hap1_ref_f1"],
                    "tn_stats/hap0_hap1_f1": val_metrics["hap0_hap1_f1"],
                }, step=epoch)


            if MASTER_PROCESS and epoch > -1:
                checkpointer.step(value=val_loss, step=epoch, conf=xtra_checkpoint_items, opt=optimizer.state_dict())
                
            dist.barrier()
        logger.info(f"Training completed after {epoch} epochs")
    except KeyboardInterrupt:
        checkpointer.step(value=val_loss, step=epoch, conf=xtra_checkpoint_items, opt=optimizer.state_dict())



def load_train_conf(confyaml):
    logger.info(f"Loading configuration from {confyaml}")
    conf = yaml.safe_load(open(confyaml).read())
    assert 'reference' in conf, "Expected 'reference' entry in training configuration"
    # assert 'data' in conf, "Expected 'data' entry in training configuration"
    return conf


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


def unwrap_model(module):
    if isinstance(module, DDP):
        return module.module
    return module


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

    # dataloader = loader.PregenLoader(DEVICE,
    #                                  kwargs.get("datadir"),
    #                                  threads=kwargs.get('threads'),
    #                                  max_decomped_batches=kwargs.get('max_decomp_batches'),
    #                                  tgt_prefix="tgkmers")

    # val_loader = loader.PregenLoader(DEVICE,
    #                                  kwargs.get("val_dir"),
    #                                  threads=kwargs.get('threads'),
    #                                  max_decomped_batches=kwargs.get('max_decomp_batches'),
    #                                  tgt_prefix="tgkmers")

    if kwargs.get('input_model'):
        ckpt = torch.load(kwargs.get("input_model"), map_location=DEVICE, weights_only=False)
        model = load_model(ckpt['conf'], ckpt)
    else:
        ckpt = None
        model = load_model(kwargs['model'], ckpt)

    model_unwrapped = unwrap_model(model)

    logger.info(f"Truncating max read depth to {model_unwrapped.read_depth}")
    # dataloader = loader.TruncateDepthLoader(dataloader, model_unwrapped.read_depth)
    dataloader = loader.make_loader(kwargs.get('datadir'), 
                                    batch_size=kwargs.get('batch_size'),
                                    num_workers=kwargs.get('threads'),
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True,
                                    max_read_depth=model_unwrapped.read_depth)
 
    # val_loader = loader.TruncateDepthLoader(val_loader, model_unwrapped.read_depth)
    val_loader = loader.make_loader(kwargs.get('val_dir'), 
                                    num_workers=kwargs.get('threads'),
                                    batch_size=kwargs.get('batch_size'),
                                    shuffle=False,
                                    pin_memory=True,
                                    drop_last=False,
                                    max_read_depth=model_unwrapped.read_depth)


    if kwargs.get('model_encoder_fix'):
        logger.info(f"Loading and freezing encoder from {kwargs['model_encoder_fix']}")
        if hasattr(model, 'module'):
            model_unwrapped = model.module
        else:
            model_unwrapped = model
        model = load_fix_encoder(model_unwrapped, kwargs['model_encoder_fix'])

    model_tot_params = sum(p.numel() for p in model.parameters())
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model total parameter count: {model_tot_params}, trainable params: {model_trainable_params}")
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
                 val_loader,
                 scheduler=scheduler,
                 model_dest=output_model,
                 checkpoint_freq=kwargs.get('checkpoint_freq', 10),
                 samples_per_epoch=kwargs.get('samples_per_epoch'),
                 xtra_checkpoint_items=kwargs['model'],
                 )

