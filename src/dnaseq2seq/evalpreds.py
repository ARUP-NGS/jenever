
from collections import defaultdict
import torch
from dnaseq2seq import util
from dnaseq2seq import vcf
from sklearn import metrics

def init_count_dict():
    return {
        'del': defaultdict(int),
        'ins': defaultdict(int),
        'snv': defaultdict(int),
        'mnv': defaultdict(int),
    }

def add_result_dicts(src, add):
    for key, subdict in add.items():
        for subkey, value in subdict.items():
            src[key][subkey] += value
    return src



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


def calc_val_accuracy(loader, model, criterion, device):
    """
    Compute accuracy (fraction of predicted bases that match actual bases),
    calculates mean number of variant counts between tgt and predicted sequence,
    and also calculates TP, FP and FN count based on reference, alt and predicted sequence.
    across all samples in valpaths, using the given model, and return it
    :param valpaths: List of paths to (src, tgt) saved tensors
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
        tot_precision = 0
        tot_recall = 0
        tot_f1 = 0

        swap_tot = 0
        for i, data in enumerate(loader):
            src = data["read"].float().to(device)
            tgt_kmers = data["tgkmers"].long().to(device)
            tgt_cls = data["tntgt"].float().to(device)
            total_batches += 1
            tot_samples += src.shape[0]
            seq_preds, probs, tn_logits = util.predict_sequence(src, model, n_output_toks=37, device=device) # 150 // 4 = 37, this will need to be changed if we ever want to change the output length

            #tgt_kmers = util.tgt_to_kmers(tgt[:, :, 0:truncate_seq_len]).float().to(DEVICE)
            tgt_kmer_idx = torch.argmax(tgt_kmers, dim=-1)[:, :, 1:]
            j = tgt_kmer_idx.shape[-1]
            seq_preds = seq_preds[:, :, 0:j, :] # tgt_kmer_idx might be a bit shorter if the sequence is truncated

            loss, swaps = compute_twohap_loss(seq_preds, tgt_kmer_idx, criterion)
            loss_tot += loss
            swap_tot += swaps
            # Compute binary classification accuracy metrics
            tnpreds = torch.sigmoid(tn_logits) > 0.5
            prec, recall, f1, _ = metrics.precision_recall_fscore_support(tgt_cls.detach().cpu().numpy(), tnpreds.detach().cpu().numpy(), average='binary')
            tot_precision += prec
            tot_recall += recall
            tot_f1 += f1

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
            tot_precision / total_batches,
            tot_recall / total_batches,
            tot_f1 / total_batches)



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