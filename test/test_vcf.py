import pysam
import pytest
from pathlib import Path
import vcf

TEST_DATA=Path(__file__).parent / "test_data"

class MockReference:

    def fetch(self, start, end, **kwargs):
        return "A" * (end - start)

@pytest.fixture
def tinybam():
    return TEST_DATA / "tiny.bam"


def test_mismatches_to_vars():
    refseq =  "AAAAAAAA"
    altseq = "TAACGAAC"
    mm = list(vcf._mismatches_to_vars(refseq, altseq, cig_offset=17, probs=[1] * len(refseq), chrom='X', window_offset=0))
    assert len(mm) == 3
    assert mm[0].ref == "A"
    assert mm[0].alt == "T"
    assert mm[0].pos == 17
    assert mm[1].ref == 'AA'
    assert mm[1].alt == 'CG'
    assert mm[1].pos == 20
    assert mm[2].ref == 'A'
    assert mm[2].alt == 'C'
    assert mm[2].pos == 24


def test_aln_to_vars_ignore_delstart():
    refseq = "ACTGACTGACTG"
    altseq =   "TGACTGACTG"
    v = list(vcf.aln_to_vars(refseq, altseq, 'X', strip_leading_indels=False))
    assert len(v) == 1
    assert v[0].ref == 'AC'
    assert v[0].alt == ''
    assert v[0].pos == 0

    v = list(vcf.aln_to_vars(refseq, altseq, 'X', strip_leading_indels=True))
    assert len(v) == 0


def test_aln_to_varsinternal_del():
    refseq = "ACTGACTGACTG"
    altseq = "ACTGA--GACTG".replace("-", "")
    v = list(vcf.aln_to_vars(refseq, altseq, 'X'))
    assert len(v) == 1
    v = v[0]
    assert v.ref == 'CT'
    assert v.alt == ''
    assert v.pos == 5


def test_aln_to_varsinternal_delins():
    refseq = "CCCCACTGA-CTG--ACTGAAAA".replace("-", "")
    altseq = "CCCCACTGA-GGGG-ACTGAAAA".replace("-", "")
    v = list(vcf.aln_to_vars(refseq, altseq, 'X'))
    assert len(v) == 2
    assert v[0].ref == 'CT'
    assert v[0].alt == 'GG'
    assert v[0].pos == 9 
    assert v[1].ref == ''
    assert v[1].alt == 'G'
    assert v[1].pos == 11
    


def test_aln_to_vars_offset_del():
    refseq = "ACTGACTGACTGACGACGT"
    altseq = "---GACTG-CTGACGACGT".replace("-", "")
    v = list(vcf.aln_to_vars(refseq, altseq, 'X'))
    assert len(v) == 1
    assert v[0].ref == 'A'
    assert v[0].alt == ''
    assert v[0].pos == 8

def test_leftalign_indel():
    refseq = "ACTGACACACACACTTCGGTG"
    altseq = "ACTGACA--CACACTTCGGTG".replace("-", "")
    v = list(vcf.aln_to_vars(refseq, altseq, 'X'))
    assert len(v) == 1
    assert v[0].ref == 'AC'
    assert v[0].alt == ''
    assert v[0].pos == 4

    # It doesn't matter if we delete an AC or a CA, the result haplotype will be the same
    refseq = "ACTGACACACACACTTCGGTG"
    altseq = "ACTGAC--ACACACTTCGGTG".replace("-", "")
    v = list(vcf.aln_to_vars(refseq, altseq, 'X'))
    assert len(v) == 1
    assert v[0].ref == 'AC'
    assert v[0].alt == ''
    assert v[0].pos == 4

        # It doesn't matter if we delete an AC or a CA, the result haplotype will be the same
    refseq = "ACTGACACACACACTTCGGTG"
    altseq = "ACTGA--CACACACTTCGGTG".replace("-", "")
    v = list(vcf.aln_to_vars(refseq, altseq, 'X'))
    assert len(v) == 1
    assert v[0].ref == 'AC'
    assert v[0].alt == ''
    assert v[0].pos == 4

def test_leftalign_across_snv():
    """
    In this case the left-alignment must stop at the SNV upstream of it
    """
    refseq = "ACTGACACACACACACTTCGGTG"
    altseq = "ACTGACTCA--CACACTTCGGTG".replace("-", "")
    v = list(vcf.aln_to_vars(refseq, altseq, 'X'))
    assert len(v) == 2
    assert v[0].ref == 'A'
    assert v[0].alt == 'T'
    assert v[0].pos == 6   
    assert v[1].ref == 'CA'
    assert v[1].alt == ''
    assert v[1].pos == 7   

def test_multi_snv_ins():
    refseq = "GGTGACTGATAAC----TGACTGACACTG".replace("-", "")
    altseq = "GGTGAC--ATAACAGTTTTACTCACACTG".replace("-", "")
    v = list(vcf.aln_to_vars(refseq, altseq, 'X', offset=10))
    assert len(v) == 4
    assert v[0].ref == 'TG'
    assert v[0].alt == ''
    assert v[0].pos == 16
    assert v[0].window_offset == 6

    assert v[1].ref == 'T'
    assert v[1].alt == 'A'
    assert v[1].pos == 23
    assert v[1].window_offset == 13

    assert v[2].ref == ''
    assert v[2].alt == 'TTTT'
    assert v[2].pos == 25
    assert v[2].window_offset == 15

    assert v[3].ref == 'G'
    assert v[3].alt == 'C'
    assert v[3].pos == 28
    assert v[3].window_offset == 18

def test_leftalign():
    refseq = "ACTGACACACACACTTCGGTG"
    # Deletion
    v = vcf.Variant(chrom='X', pos=8, ref='AC', alt='', qual=1.0, window_offset=0, var_index=0)
    v = vcf.left_align(refseq, v)
    assert v.pos == 4
    assert v.ref == 'AC'

    # Insertion
    v = vcf.Variant(chrom='X', pos=10, ref='', alt='AC', qual=1.0, window_offset=0, var_index=0)
    v = vcf.left_align(refseq, v)
    assert v.pos == 4
    assert v.alt == 'AC'

    # Homopolymer
    refseq = "AAAAAAACACACACTTCGGTG"
    v = vcf.Variant(chrom='X', pos=6, ref='A', alt='', qual=1.0, window_offset=0, var_index=0)
    v = vcf.left_align(refseq, v)
    assert v.pos == 0
    assert v.ref == 'A'

    v = vcf.Variant(chrom='X', pos=6, ref='', alt='A', qual=1.0, window_offset=0, var_index=0)
    v = vcf.left_align(refseq, v)
    assert v.pos == 0
    assert v.alt == 'A'

    refseq = "TAATTTCTTTTGTTTAAGCTTCAGTTGTTCTTTCGTTCTCTACTTTCTCAAGGAAGAAGCTTTAGTTACTGATTTTTTACCTTCTTTTCTTATATATGAACTTAATGTTACACATTTCTTCTAAGCATTGCT"
    altseq = "AATTTCTTTTGTTTAAGCTTCAATTGTTCTTTCGTTCTCTACTTTCTCAAGGAAGAAGCTTTAGTTACTGATTTTTTACCTTCTTTTCTTATATATGAACTTAATGTTACACATTTTCTTCTAAGCATTGCT"
    v = vcf.Variant(chrom='X', pos=25000494, ref='', alt='T', qual=1.0, window_offset=0, var_index=0)
    v = vcf.left_align(refseq, v, var_offset=25000377)
    assert v.pos == 25000491
    assert v.alt == 'T'



def test_display_aln():
    refseq = "GGTGACTGATAAC----TGACTGACACTG".replace("-", "")
    altseq = "GGTGAC--ATAACAGTTTTACTCACACTG".replace("-", "")
    result = vcf.align_sequences(refseq, altseq)
    vcf._display_aln(refseq, altseq, result.paths[0])
    print(result.paths[0].to_cigar())


def test_agg_variants(tinybam):
    vars0 = {
        ('1', 167085919, 'G', 'A'): [
            vcf.Variant(chrom='1', pos=167085919, ref='G', alt='A', qual=1.0, step=1, window_offset=5, var_count=2),
            vcf.Variant(chrom='1', pos=167085919, ref='G', alt='A', qual=0.5, step=1, window_offset=10, var_count=2),
        ],
        ('1', 167086038, 'G', 'T'): [
            vcf.Variant(chrom='1', pos=167086038, ref='G', alt='T', qual=0.6, step=1, window_offset=5, var_count=2),
        ],
    }
    vars1 = {
        ('1', 167086038, 'G', 'T'): [
            vcf.Variant(chrom='1', pos=167086038, ref='G', alt='T', qual=0.5, step=1, window_offset=7, var_count=1),
        ],
    }
    aln = pysam.AlignmentFile(tinybam)
    vcfvars = list(vcf.construct_vcfvars(vars0, vars1, aln, MockReference()))
    assert len(vcfvars) == 2
    first = vcfvars[0]
    assert first.chrom == '1'
    assert first.pos == 167085920
    assert first.ref == 'G'
    assert first.alt == 'A'
    assert first.quals == [1.0, 0.5]
    assert first.window_offset == [5, 10]
    assert first.het == True
    assert first.genotype == (0, 1)
    assert first.step_count == 2
    assert first.call_count == 2

    second = vcfvars[1]
    assert second.chrom == '1'
    assert second.pos == 167086039
    assert second.ref == 'G'
    assert second.alt == 'T'
    assert second.quals == [0.6, 0.5]
    assert second.step_count == 1
    assert second.call_count == 2
