
import pytest
from dataclasses import dataclass
from dnaseq2seq import call, vcf, util
from typing import List



def test_merge_and_pad_vars():
    target = vcf.Variant(chrom='X', pos=10, ref='ATCGCTA', alt='', qual=1)
    overlaps = [vcf.Variant(chrom='X', pos=12, ref='C', alt='', qual=1),
                vcf.Variant(chrom='X', pos=15, ref='TA', alt='CT', qual=1),
                ]
    new_var = call.merge_and_pad_vars(target, overlaps)
    assert new_var.pos == target.pos
    assert new_var.ref == 'ATCGCTA'
    assert new_var.alt == 'ATGCCT'
