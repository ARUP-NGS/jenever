import numpy as np
from dnaseq2seq.vcf import _cigtups, Cigar
from typing import List, Tuple
from collections import Counter
from skbio.alignment import StripedSmithWaterman


class RefSeqMap:
    """ 
    Refmap allows access from a single reference base into the zero or more query bases that align to it 
    Insertions of query bases are stacked together into a single string that maps to one reference base
    Deletions are represented with a '-' (dash) character
    There's also a 'skip' operator that maps to None, which is used when the beginning of the target sequence alignment
    doesn't start at base 0 of the query (reference) sequence
    """
    def __init__(self, aln, probs=None):
        self.aln = aln
        if probs is not None:
            assert len(probs) == len(self.aln.target_sequence)
            self.probs = probs
        else:
            self.probs = np.ones(len(self.aln.target_sequence))
        self.cigtups = list(_cigtups(self.aln.cigar))
        
        if self.aln.target_begin > 0:
            self.cigtups[0].len -= 1
            if len(self.cigtups) == 0:
                self.cigtups.pop(0)
            self.cigtups.insert(0, Cigar(op="S", len=1))
            self.cigtups.insert(1, Cigar(op="D", len=self.aln.target_begin))
            
        if self.aln.query_begin > 0:
            self.cigtups.insert(0, Cigar(op="S", len=self.aln.query_begin))
        
        if len(self.aln.query_sequence) > self.aln.query_end:
            self.cigtups.append(Cigar(op="S", len=len(self.aln.query_sequence) - self.aln.query_end - 1))

        # print(self.cigtups)
        self.refmap = self._build_map()
    
    def _build_map(self):
        refmap = []
        q_offset = 0
        t_offset = 0
        for cig in self.cigtups:
            # print(f"cig: {cig.op} {cig.len}, q_offset: {q_offset}, t_offset: {t_offset}")
            if cig.op == "M":
                for i in range(cig.len):
                    refmap.append({
                        "ref": self.aln.query_sequence[q_offset],
                        "target": self.aln.target_sequence[t_offset],
                        "prob": self.probs[t_offset],
                        })
                    # print(f"q offset: {q_offset} rm: {refmap[-1]}")
                    q_offset += 1
                    t_offset += 1
                    
            elif cig.op == "I":
                for _ in range(cig.len):
                    refmap.append({
                        "ref": self.aln.query_sequence[q_offset],
                        "target": "-",
                        "prob": -1.0,
                        })
                    # print(f"q offset: {q_offset} rm: {refmap[-1]}")
                    q_offset += 1

            elif cig.op == "D":
                if refmap[-1]['target'] is not None:
                    refmap[-1]['target'] = refmap[-1]['target'] + self.aln.target_sequence[t_offset:t_offset+cig.len]
                    refmap[-1]['prob'] = np.mean(self.probs[t_offset:t_offset+cig.len])
                else:
                    refmap[-1]['target'] = self.aln.target_sequence[t_offset:t_offset+cig.len]
                    refmap[-1]['prob'] = np.mean(self.probs[t_offset:t_offset+cig.len])
                # print(f"q offset: {q_offset} rm: {refmap[-1]}")
                t_offset += cig.len

            elif cig.op == "S":
                for i in range(cig.len):
                    refmap.append({
                        "ref": self.aln.query_sequence[q_offset],
                        "target": None,
                        "prob": -1.0,
                        })
                    # print(f"q offset: {q_offset} rm: {refmap[-1]}")
                    q_offset += 1
            else:
                raise ValueError(f"Unknown cigar op {cig.op}")
        return refmap
    
    def __getitem__(self, idx):
        return self.refmap[idx]
    
    def __len__(self):
        return len(self.refmap)
    
    def ref_bases(self):
        return "".join([r['ref'] for r in self.refmap])
    

class MultiRefMap:
    def __init__(self, refmaps: List[RefSeqMap]):
        self.refmaps = refmaps
        for rm in refmaps:
            if rm.ref_bases() != refmaps[0].ref_bases():
                m = []
                for i, (a,b) in enumerate(zip(rm.ref_bases(), refmaps[0].ref_bases())):
                    if a != b:
                        m.append(f"{i}: {a} != {b}")
                    else:
                        m.append(f"{i}: {a} == {b}")
                print("\n".join(m))
                raise Exception(f"Refmaps do not have idential ref bases at positions {m}")

    def __getitem__(self, idx):
        return [rm[idx] for rm in self.refmaps]
    
    def __len__(self):
        return len(self.refmaps[0])
    
    def ref_bases(self):
        return self.refmaps[0].ref_bases()
    

def merge_refmaps(maps: MultiRefMap):
    """
    Merge the aligned haplotypes represented by the MultiRefMap into a single sequence
    This is accomplished by examining all bases / sequences that align to a each reference position and 
    selecting the one with the highest sum of probabilities across all aligned sequences

    : returns: the merged haplotype sequence as a string
    """
    merged_haplotype = []
    for i in range(len(maps)):
        counts = Counter()
        bases = maps[i]
        for b in bases:
            counts[b['target']] += b['prob']
        merged_haplotype.append(counts.most_common(1)[0][0])
    return "".join(filter(lambda x: x is not None, merged_haplotype))

def fmt(s):
    if s is None:
        return "N".ljust(5)
    else:
        return s.ljust(5)

def align_and_merge_haplotypes(haplotypes: List[Tuple[str, np.array]], ref_seq: str, pos_offset: int = 0):
    """
    Merge the overlapping haplotypes into a single sequence
    The algorithm here is to align each haplotype to the reference sequence and then merge the aligned haplotypes
    """
    ssw = StripedSmithWaterman(ref_seq,
                               gap_open_penalty=5,
                               gap_extend_penalty=0.1,
                               match_score=1,
                               mismatch_score=-1)
    refmaps = []
    for i, (hapseq, probs) in enumerate(haplotypes):
        aln = ssw(hapseq)
        refmap = RefSeqMap(aln, probs)
        refmaps.append(refmap)
        # if i > 1:
        #     if refmaps[-2].ref_bases() != refmaps[-1].ref_bases():
        #         raise Exception(f"Refmaps {i} and {i-1}do not have idential ref bases")
    

    # for i in range(len(refmaps[0])):
    #     print(f"{i}\t{refmaps[0][i]['ref']}\t{refmaps[0][i]['target']}\t{refmaps[0][i]['prob'] :.4f}")

    refmaps = MultiRefMap(refmaps)
    refbase = refmaps.ref_bases()

    # for i in range(len(refmaps)):
    #     d = " ".join(fmt(r['target']) for r in refmaps[i])
    #     print(f"{i :5}\t{refbase[i]}\t{d}")

    merged = merge_refmaps(refmaps)
    return merged
