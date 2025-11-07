
import logging
import torch
import torch.nn as nn

from dnaseq2seq.model import VarTransformer
from dnaseq2seq import loader
from dnaseq2seq import util
from dnaseq2seq.evalpreds import calc_val_accuracy, safe_compute_ppav

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")


def load_model_for_eval(model_path):
    """
    Load a model from a checkpoint file and set it to evaluation mode.
    
    :param model_path: Path to the model checkpoint file
    :returns: Tuple of (model, model_config)
    """
    logger.info(f"Loading model from {model_path}")
    model_info = torch.load(model_path, map_location=DEVICE, weights_only=False)
    statedict = model_info['model_state_dict']
    modelconf = model_info['conf']
    
    # Remove '_orig_mod.' prefix from state dict keys if present (from torch.compile)
    new_state_dict = {}
    for key in statedict.keys():
        new_key = key.replace('_orig_mod.', '')
        new_state_dict[new_key] = statedict[key]
    statedict = new_state_dict

    logger.info(f"Loading model configuration: {modelconf}")
    model = VarTransformer(
        read_depth=modelconf['max_read_depth'],
        feature_count=modelconf['feats_per_read'],
        kmer_dim=util.FEATURE_DIM,
        n_encoder_layers=modelconf['encoder_layers'],
        n_decoder_layers=modelconf['decoder_layers'],
        embed_dim_factor=modelconf['embed_dim_factor'],
        encoder_attention_heads=modelconf['encoder_attention_heads'],
        decoder_attention_heads=modelconf['decoder_attention_heads'],
        decoder_embed_dim=modelconf['decoder_embed_dim'],
        d_ff=modelconf['dim_feedforward'],
        device=DEVICE
    )

    model.load_state_dict(statedict, strict=False)
    model.eval()
    model.to(DEVICE)
    
    return model, modelconf


def evaluate_model(model_path, dataset_path, batch_size=64, threads=1, device=DEVICE):
    """
    Evaluate a model on a dataset and print all evaluation metrics to standard output.
    
    This function loads a model from a checkpoint, creates a data loader from the dataset,
    computes all validation accuracy statistics, and prints them in a formatted way.
    
    :param model_path: Path to the model checkpoint file
    :param dataset_path: Path to the dataset directory (LMDB or pre-generated)
    :param batch_size: Batch size for evaluation (default: 64)
    :param threads: Number of worker processes for data loading (default: 1)
    """
    # Load model
    model, modelconf = load_model_for_eval(model_path)
    
    # Create data loader
    logger.info(f"Creating data loader from {dataset_path}")
    data_loader = loader.make_loader(
        dataset_path,
        batch_size=batch_size,
        num_workers=threads,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        max_read_depth=modelconf['max_read_depth']
    )
    
    # Create loss criterion
    criterion = nn.NLLLoss()
    
    # Compute validation accuracy metrics
    logger.info("Computing evaluation metrics...")
    (acc0, acc1, var_count0, var_count1, results0, results1, 
     val_loss, swap_tot, tn_prec, tn_recall, tn_f1) = calc_val_accuracy(
        data_loader, model, criterion, DEVICE
    )
    
    # Compute PPA and PPV for each variant type
    ppa_dels, ppv_dels = safe_compute_ppav(results0, results1, 'del')
    ppa_ins, ppv_ins = safe_compute_ppav(results0, results1, 'ins')
    ppa_snv, ppv_snv = safe_compute_ppav(results0, results1, 'snv')
    ppa_mnv, ppv_mnv = safe_compute_ppav(results0, results1, 'mnv')
    
    # Print all metrics to standard output
    print("=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    print(f"\nModel: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"\n{'Metric':<40} {'Value':<20}")
    print("-" * 80)
    
    # Accuracy metrics
    print(f"{'Haplotype 0 Accuracy':<40} {acc0.item() if isinstance(acc0, torch.Tensor) else acc0:.6f}")
    print(f"{'Haplotype 1 Accuracy':<40} {acc1.item() if isinstance(acc1, torch.Tensor) else acc1:.6f}")
    print(f"{'Mean Variant Count (Hap 0)':<40} {var_count0:.6f}")
    print(f"{'Mean Variant Count (Hap 1)':<40} {var_count1:.6f}")
    print(f"{'Validation Loss':<40} {val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss:.6f}")
    print(f"{'Total Haplotype Swaps':<40} {swap_tot}")
    
    # Binary classification metrics
    print(f"\n{'Binary Classification Metrics':<40}")
    print("-" * 80)
    print(f"{'TN Precision':<40} {tn_prec:.6f}")
    print(f"{'TN Recall':<40} {tn_recall:.6f}")
    print(f"{'TN F1 Score':<40} {tn_f1:.6f}")
    
    # Variant detection metrics by type
    print(f"\n{'Variant Detection Metrics':<40}")
    print("-" * 80)
    
    # Deletions
    print(f"\n{'Deletions':<40}")
    print(f"{'  PPA (Positive Predictive Accuracy)':<40} {ppa_dels:.6f}")
    print(f"{'  PPV (Positive Predictive Value)':<40} {ppv_dels:.6f}")
    tp_dels = results0['del']['tp'] + results1['del']['tp']
    fp_dels = results0['del']['fp'] + results1['del']['fp']
    fn_dels = results0['del']['fn'] + results1['del']['fn']
    print(f"{'  True Positives':<40} {tp_dels}")
    print(f"{'  False Positives':<40} {fp_dels}")
    print(f"{'  False Negatives':<40} {fn_dels}")
    
    # Insertions
    print(f"\n{'Insertions':<40}")
    print(f"{'  PPA (Positive Predictive Accuracy)':<40} {ppa_ins:.6f}")
    print(f"{'  PPV (Positive Predictive Value)':<40} {ppv_ins:.6f}")
    tp_ins = results0['ins']['tp'] + results1['ins']['tp']
    fp_ins = results0['ins']['fp'] + results1['ins']['fp']
    fn_ins = results0['ins']['fn'] + results1['ins']['fn']
    print(f"{'  True Positives':<40} {tp_ins}")
    print(f"{'  False Positives':<40} {fp_ins}")
    print(f"{'  False Negatives':<40} {fn_ins}")
    
    # SNVs
    print(f"\n{'SNVs':<40}")
    print(f"{'  PPA (Positive Predictive Accuracy)':<40} {ppa_snv:.6f}")
    print(f"{'  PPV (Positive Predictive Value)':<40} {ppv_snv:.6f}")
    tp_snv = results0['snv']['tp'] + results1['snv']['tp']
    fp_snv = results0['snv']['fp'] + results1['snv']['fp']
    fn_snv = results0['snv']['fn'] + results1['snv']['fn']
    print(f"{'  True Positives':<40} {tp_snv}")
    print(f"{'  False Positives':<40} {fp_snv}")
    print(f"{'  False Negatives':<40} {fn_snv}")
    
    # MNVs
    print(f"\n{'MNVs':<40}")
    print(f"{'  PPA (Positive Predictive Accuracy)':<40} {ppa_mnv:.6f}")
    print(f"{'  PPV (Positive Predictive Value)':<40} {ppv_mnv:.6f}")
    tp_mnv = results0['mnv']['tp'] + results1['mnv']['tp']
    fp_mnv = results0['mnv']['fp'] + results1['mnv']['fp']
    fn_mnv = results0['mnv']['fn'] + results1['mnv']['fn']
    print(f"{'  True Positives':<40} {tp_mnv}")
    print(f"{'  False Positives':<40} {fp_mnv}")
    print(f"{'  False Negatives':<40} {fn_mnv}")
    
    print("\n" + "=" * 80)

