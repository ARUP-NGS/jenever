#!/usr/bin/env python

import random
import string
import sys
import yaml
import subprocess
import pysam
from pathlib import Path


BEDTOOLS_PATH="/home/22319/miniforge3/envs/py3/bin/bedtools"

RUN_HAPPY="/home/22319/src/jenever/scripts/run_happy.sh"
#RUN_HAPPY="/uufs/chpc.utah.edu/common/home/u0379426/src/jovian/scripts/run_happy.sh"


GIABROOT="/mnt/ri_share/Data/variant-transformer/GIAB_NISTv4.2.1_2023-10-26"
#GIABROOT="/uufs/chpc.utah.edu/common/home/u0379426/GIAB_NISTv4.2.1_2023-10-26"

GIAB_DATA={
        "na12878_vcf": f"{GIABROOT}/vcf/HG001_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "na24631_vcf": f"{GIABROOT}/vcf/HG005_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "na24385_vcf": f"{GIABROOT}/vcf/HG002_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "hg003_vcf": f"{GIABROOT}/vcf/HG003_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "hg004_vcf": f"{GIABROOT}/vcf/HG004_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "hg005_vcf": f"{GIABROOT}/vcf/HG005_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "hg006_vcf": f"{GIABROOT}/vcf/HG006_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "hg007_vcf": f"{GIABROOT}/vcf/HG007_GRCh37_1_22_v4.2.1_benchmark.vcf.gz",
        "na12878_bed": f"{GIABROOT}/bed/HG001_GRCh37_1_22_v4.2.1_benchmark.bed",
        "na24385_bed": f"{GIABROOT}/bed/HG002_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed",
        "hg003_bed": f"{GIABROOT}/bed/HG003_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed",
        "hg004_bed": f"{GIABROOT}/bed/HG004_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed",
        "na24631_bed": f"{GIABROOT}/bed/HG005_GRCh37_1_22_v4.2.1_benchmark.bed",
        "hg005_bed": f"{GIABROOT}/bed/HG005_GRCh37_1_22_v4.2.1_benchmark.bed",
        "hg006_bed": f"{GIABROOT}/bed/HG006_GRCh37_1_22_v4.2.1_benchmark.bed",
        "hg007_bed": f"{GIABROOT}/bed/HG007_GRCh37_1_22_v4.2.1_benchmark.bed",
        }


def find_giab(name):
    name = name.upper()
    if ("NA12878" in name) or ("HG001" in name):
        return GIAB_DATA['na12878_bed'], GIAB_DATA['na12878_vcf']
    elif "GM24631" in name:
        return GIAB_DATA['na24631_bed'], GIAB_DATA['na24631_vcf']
    elif ("GM24385" in name) or ("HG002" in name) or ("NA24385" in name) or ("NIST-002" in name):
        return GIAB_DATA['na24385_bed'], GIAB_DATA['na24385_vcf']
    elif "HG003" in name or ("NIST-003" in name):
        return GIAB_DATA['hg003_bed'], GIAB_DATA['hg003_vcf']
    elif "HG004" in name or ("NIST-004" in name):
        return GIAB_DATA['hg004_bed'], GIAB_DATA['hg004_vcf']
    elif "HG005" in name:
        return GIAB_DATA['hg005_bed'], GIAB_DATA['hg005_vcf']
    elif "HG006" in name or "NA24694" in name:
        return GIAB_DATA['hg006_bed'], GIAB_DATA['hg006_vcf']
    elif "HG007" in name or "NA24695":
        return GIAB_DATA['hg007_bed'], GIAB_DATA['hg007_vcf']
    raise ValueError(f"Couldn't find a match for input file {name}")


def runhappy(truthvcf, query_vcf, bed):
    subprocess.run(f"{RUN_HAPPY} {truthvcf} {query_vcf} {bed}", shell=True)


def bed_intersect(b1, b2):
    assert Path(b1).exists()
    assert Path(b2).exists()
    assert b1.endswith('.bed')
    assert b2.endswith('.bed')
    outbed = "." + Path(b1).name.replace(".bed", "")[0:12] + "_" + Path(b2).name.replace(".bed", "")[0:12] + f"_{''.join(random.sample(string.ascii_letters, k=6))}.bed"
    cmd = f"{BEDTOOLS_PATH} intersect -a {b1} -b {b2} > {outbed}"
    sys.stderr.write(f"Running {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)
    return outbed

def sort_vcf(vcf):
    dest = Path(vcf).name.replace(".vcf", "") + "_sorted.vcf"
    cmd = f"{BEDTOOLS_PATH} sort -header -i {vcf} > {dest}"
    sys.stderr.write(f"Running {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)
    return dest


def main(bed, vcfs):
    for vcf in vcfs:
        giab_bed, truth_vcf = find_giab(vcf)
        region_giab_bed = bed_intersect(giab_bed, bed)
        sorted_vcf = sort_vcf(vcf)
        runhappy(truth_vcf, sorted_vcf, region_giab_bed)
        # collate results somehow?

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2:])
