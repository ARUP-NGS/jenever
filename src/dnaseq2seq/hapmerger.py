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
    def __init__(self, aln, ref_offset: int, probs=None):
        self.aln = aln
        self.ref_offset = ref_offset
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
                        "target": "",
                        "prob": 1.0, # No idea what to put here - but if you want the deletion bases to be included in the merged haplotype then this should be ~1.0
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
                        "prob": 0.0,
                        })
                    # print(f"q offset: {q_offset} rm: {refmap[-1]}")
                    q_offset += 1
            else:
                raise ValueError(f"Unknown cigar op {cig.op}")
        return refmap
    
    def __getitem__(self, ref_pos):
        idx = ref_pos - self.ref_offset
        if idx < 0 or idx >= len(self.refmap):
            return None
        else:
            return self.refmap[idx]
    
    def __len__(self):
        return len(self.refmap)
    
    def ref_bases(self):
        return "".join([r['ref'] for r in self.refmap])
    

class MultiRefMap:
    def __init__(self, refmaps: List[RefSeqMap]):
        self.refmaps = refmaps
        self.ref_min = min([rm.ref_offset for rm in refmaps])
        self.ref_max = max([rm.ref_offset + len(rm) for rm in refmaps])
        self.refbases = []
        for i in range(self.ref_min, self.ref_max):
            bset = set()
            for rm in refmaps:
                b = rm[i]
                if b is not None:
                    bset.add(b['ref'])
            if len(bset) == 1:
                self.refbases.append(bset.pop())
            else:
                raise Exception(f"Multiple bases at position {i}: {bset}")
        

    def __getitem__(self, idx):
        return [rm[idx] for rm in self.refmaps]
    
    def __len__(self):
        return len(self.refbases)
    
    def ref_bases(self):
        return self.refbases
    

def merge_refmaps(maps: MultiRefMap):
    """
    Merge the aligned haplotypes represented by the MultiRefMap into a single sequence
    This is accomplished by examining all bases / sequences that align to a each reference position and 
    selecting the one with the highest sum of probabilities across all aligned sequences

    Note that this has oddly different behavior for SNPs and insertions than it does for deletions.
    For SNVs and insertions, the entire modified unit is represneted as a single entry, but for
    deletions each deleted base is represented as a separate entry.
    This means that when iterating over each aligned sequence,SNVs and insertions are accepted or rejected basically by
    voting across all the aligned sequences, but for deletions each base is voted on individually.

    : returns: the merged haplotype sequence as a string
    """
    merged_haplotype = []
    meta = []
    for i in range(maps.ref_min, maps.ref_max):
        counts = Counter()
        bases = maps[i]
        for b in bases:
            if b is None:
                continue
            counts[b['target']] += b['prob']
        
        most_common_base, most_common_prob_sum = counts.most_common(1)[0]
        merged_haplotype.append(most_common_base)
        meta.append({
            "overlapping_windows": len([b for b in bases if b is not None]),
            "total_bases": len(bases),
            "total_prob": sum(b['prob'] for b in bases if b is not None),
            "supporting_prob_sum": most_common_prob_sum,
        })
    
    merged_hap_seq = []
    merged_meta = []
    for hap, meta in zip(merged_haplotype, meta):
        if hap is not None and hap != "":
            merged_hap_seq.append(hap)
            for _ in range(len(hap)): # We require a meta entry for each base in the merged haplotype, and insertions have more than one base
                merged_meta.append(meta)

    final_hap_seq = "".join(merged_hap_seq)
    if len(final_hap_seq) != len(merged_meta):
        print("aargh")
    assert len(final_hap_seq) == len(merged_meta), f"Final haplotype sequence length {len(final_hap_seq)} does not match meta length {len(merged_meta)}"
    return final_hap_seq, merged_meta


def fmt(s):
    if s is None:
        return "N".ljust(5)
    else:
        return s.ljust(5)


def align_and_merge_haplotypes(haplotypes: List[Tuple[str, np.array, int]], ref_seq: str, ref_start: int):
    """
    Merge the overlapping haplotypes into a single sequence
    The algorithm here is to align each haplotype to the reference sequence and then merge the aligned haplotypes
    """

    refmaps = []
    for i, (hapseq, probs, ref_offset) in enumerate(haplotypes):
        seq_offset = max(0, ref_offset - ref_start - 0)
        ssw = StripedSmithWaterman(ref_seq[seq_offset:seq_offset+len(hapseq) + 10],
                            gap_open_penalty=5,
                            gap_extend_penalty=0.1,
                            match_score=1,
                            mismatch_score=-1)
        aln = ssw(hapseq)
        refmap = RefSeqMap(aln, seq_offset, probs)
        refmaps.append(refmap)

    refmaps = MultiRefMap(refmaps)

    # refbase = refmaps.ref_bases()
    # for i in range(refmaps.ref_min, refmaps.ref_max):
    #     d = " ".join(fmt(r['target']) if r else "  N   " for r in refmaps[i])
    #     print(f"{i}\t{i+ref_start :5}\t{refbase[i - refmaps.ref_min]}\t{d}")

    merged, merge_info = merge_refmaps(refmaps)
    return merged, merge_info
