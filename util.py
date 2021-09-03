import torch
import numpy as np


INDEX_TO_BASE = [
    'A', 'C', 'G', 'T'
]

def readstr(t):
    t = t.detach().cpu().numpy()
    assert len(t.shape) == 2, "Need two dimensional input"
    output = []
    for pos in range(t.shape[0]):
        if t[pos, 0:4].sum() == 0:
            output.append(".")
        else:
            output.append(INDEX_TO_BASE[np.argmax(t[pos, 0:4])])
            if t.shape[-1] > 4 and t[pos, -1] == 1: # Clip flag is set, which means emit lowercase
                output[-1] = output[-1].lower()

    return "".join(output)


def to_pileup(data, altmask=None):
    pileup = []
    for i in range(data.shape[1]):
        alt = "" if altmask is None else altmask[i].item()
        pileup.append(readstr(data[:, i, :]) + f"\t{alt}")
    return "\n".join(pileup)


def predprobs(t):
    t = t.detach().cpu().numpy()
    output = []
    for pos in range(t.shape[0]):
        if t[pos, :].sum() == 0:
            output.append(".")
        else:
            output.append(f"{t[pos, np.argmax(t[pos, :])]:.1}f")
    return "".join(output)


def tgt_str(seq):
    return "".join(INDEX_TO_BASE[b] for b in seq)


def correctstr(seq, predseq):
    seq = "".join(INDEX_TO_BASE[b] for b in seq)
    output = "".join('*' if a==b else 'x' for a,b in zip(seq, predseq))
    return output


def writeseqtensor(t):
    assert len(t.shape) == 2, f"Expected 2-dimensional input"
    for pos in range(t.shape[0]):
        clip = 1 if t[pos, 7] else 0
        print(clip, end="")
    print()
    for pos in range(t.shape[0]):
        refmatch = 1 if t[pos, 6] else 0
        print(refmatch, end="")
    print()
    for pos in range(t.shape[0]):
        if torch.sum(t[pos, 0:4]) == 0:
            base = '.'
        else:
            base = INDEX_TO_BASE[torch.argmax(t[pos, 0:4])]
        print(f"{base}", end="")
    print()
