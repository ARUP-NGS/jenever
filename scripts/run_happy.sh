# script to run hap.py from the pkrusche/hap.py docker image

# repo:  https://github.com/Illumina/hap.py
# manual:  https://github.com/Illumina/hap.py/blob/master/doc/happy.md
# --help output below
# docker image:  docker pull pkrusche/hap.py
# all file paths absolute! (simplified docker volume mounting)


# output files
#OUT_PREFIX="Nextera_GM24385_500ng"

# vcfs and bed to compare
TRUTH_VCF=$(readlink -f $1)
QUERY_VCF=$(readlink -f $2)
TARGET_BED=$(readlink -f $3)

OUT_PREFIX=$(basename $QUERY_VCF | sed -e "s/.vcf//g")

#OUT_DIR="/mnt/ri_share/Data/variant-transformer/analysis/happy_results/${OUT_PREFIX}/"
OUT_DIR="$(pwd)/${OUT_PREFIX}/"
mkdir -p $OUT_DIR

# stratification description tsv in dir of stratification beds
STRAT_TSV="/mnt/ri_share/Data/variant-transformer/v3.0-stratifications-GRCh37/v3.0-GRCh37-v4.2.1-stratifications.tsv"
# STRAT_TSV="/mnt/rd_share/RD_RW/jacob/wgs/happy/v3.0-stratifications-GRCh37/ARUP_and_v3.0_top_level_stratifications.tsv"

# reference files
REF_DIR="/mnt/ri_share/Data/variant-transformer/ref/" 
REF="human_g1k_v37_decoy_phiXAdaptr.fasta" 
VCFEVAL_REF="/mnt/ri_share/Data/variant-transformer/ref/human_g1k_v37_decoy_phiXAdaptr.sdf"


sudo docker run -it \
        -v ${TRUTH_VCF%/*}:${TRUTH_VCF%/*} \
        -v ${QUERY_VCF%/*}:${QUERY_VCF%/*} \
        -v ${TARGET_BED%/*}:${TARGET_BED%/*} \
        -v ${STRAT_TSV%/*}:${STRAT_TSV%/*} \
        -v $OUT_DIR:$OUT_DIR \
        -v $REF_DIR:$REF_DIR \
         mgibio/hap.py:v0.3.15 \
        /opt/hap.py/bin/hap.py \
        $TRUTH_VCF \
        $QUERY_VCF \
        -o $OUT_DIR/$OUT_PREFIX \
        -r $REF_DIR/$REF \
        --stratification $STRAT_TSV \
        --target-regions $TARGET_BED \
        --engine vcfeval \
        --engine-vcfeval-template $REF_DIR/$VCFEVAL_REF


# usage: Haplotype Comparison [-h] [-v] [-r REF] [-o REPORTS_PREFIX]
#                             [--scratch-prefix SCRATCH_PREFIX] [--keep-scratch]
#                             [-t {xcmp,ga4gh}] [-f FP_BEDFILE]
#                             [--stratification STRAT_TSV]
#                             [--stratification-region STRAT_REGIONS]
#                             [--stratification-fixchr] [-V] [-X]
#                             [--no-write-counts] [--output-vtc]
#                             [--preserve-info] [--roc ROC] [--no-roc]
#                             [--roc-regions ROC_REGIONS]
#                             [--roc-filter ROC_FILTER] [--roc-delta ROC_DELTA]
#                             [--ci-alpha CI_ALPHA] [--no-json]
#                             [--location LOCATIONS] [--pass-only]
#                             [--filters-only FILTERS_ONLY] [-R REGIONS_BEDFILE]
#                             [-T TARGETS_BEDFILE] [-L] [--no-leftshift]
#                             [--decompose] [-D] [--bcftools-norm] [--fixchr]
#                             [--no-fixchr] [--bcf] [--somatic]
#                             [--set-gt {half,hemi,het,hom,first}]
#                             [--gender {male,female,auto,none}]
#                             [--preprocess-truth] [--usefiltered-truth]
#                             [--preprocessing-window-size PREPROCESS_WINDOW]
#                             [--adjust-conf-regions] [--no-adjust-conf-regions]
#                             [--unhappy] [-w WINDOW]
#                             [--xcmp-enumeration-threshold MAX_ENUM]
#                             [--xcmp-expand-hapblocks HB_EXPAND]
#                             [--threads THREADS]
#                             [--engine {xcmp,vcfeval,scmp-somatic,scmp-distance}]
#                             [--engine-vcfeval-path ENGINE_VCFEVAL]
#                             [--engine-vcfeval-template ENGINE_VCFEVAL_TEMPLATE]
#                             [--scmp-distance ENGINE_SCMP_DISTANCE]
#                             [--logfile LOGFILE] [--verbose | --quiet]
#                             [_vcfs [_vcfs ...]]
# 
# positional arguments:
#   _vcfs                 Two VCF files.
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   -v, --version         Show version number and exit.
#   -r REF, --reference REF
#                         Specify a reference file.
#   -o REPORTS_PREFIX, --report-prefix REPORTS_PREFIX
#                         Filename prefix for report output.
#   --scratch-prefix SCRATCH_PREFIX
#                         Directory for scratch files.
#   --keep-scratch        Filename prefix for scratch report output.
#   -t {xcmp,ga4gh}, --type {xcmp,ga4gh}
#                         Annotation format in input VCF file.
#   -f FP_BEDFILE, --false-positives FP_BEDFILE
#                         False positive / confident call regions (.bed or
#                         .bed.gz). Calls outside these regions will be labelled
#                         as UNK.
#   --stratification STRAT_TSV
#                         Stratification file list (TSV format -- first column
#                         is region name, second column is file name).
#   --stratification-region STRAT_REGIONS
#                         Add single stratification region, e.g.
#                         --stratification-region TEST:test.bed
#   --stratification-fixchr
#                         Add chr prefix to stratification files if necessary
#   -V, --write-vcf       Write an annotated VCF.
#   -X, --write-counts    Write advanced counts and metrics.
#   --no-write-counts     Do not write advanced counts and metrics.
#   --output-vtc          Write VTC field in the final VCF which gives the
#                         counts each position has contributed to.
#   --preserve-info       When using XCMP, preserve and merge the INFO fields in
#                         truth and query. Useful for ROC computation.
#   --roc ROC             Select a feature to produce a ROC on (INFO feature,
#                         QUAL, GQX, ...).
#   --no-roc              Disable ROC computation and only output summary
#                         statistics for more concise output.
#   --roc-regions ROC_REGIONS
#                         Select a list of regions to compute ROCs in. By
#                         default, only the '*' region will produce ROC output
#                         (aggregate variant counts).
#   --roc-filter ROC_FILTER
#                         Select a filter to ignore when making ROCs.
#   --roc-delta ROC_DELTA
#                         Minimum spacing between ROC QQ levels.
#   --ci-alpha CI_ALPHA   Confidence level for Jeffrey's CI for recall,
#                         precision and fraction of non-assessed calls.
#   --no-json             Disable JSON file output.
#   --location LOCATIONS, -l LOCATIONS
#                         Comma-separated list of locations [use naming after
#                         preprocessing], when not specified will use whole VCF.
#   --pass-only           Keep only PASS variants.
#   --filters-only FILTERS_ONLY
#                         Specify a comma-separated list of filters to apply (by
#                         default all filters are ignored / passed on.
#   -R REGIONS_BEDFILE, --restrict-regions REGIONS_BEDFILE
#                         Restrict analysis to given (sparse) regions (using -R
#                         in bcftools).
#   -T TARGETS_BEDFILE, --target-regions TARGETS_BEDFILE
#                         Restrict analysis to given (dense) regions (using -T
#                         in bcftools).
#   -L, --leftshift       Left-shift variants safely.
#   --no-leftshift        Do not left-shift variants safely.
#   --decompose           Decompose variants into primitives. This results in
#                         more granular counts.
#   -D, --no-decompose    Do not decompose variants into primitives.
#   --bcftools-norm       Enable preprocessing through bcftools norm -c x -D
#                         (requires external preprocessing to be switched on).
#   --fixchr              Add chr prefix to VCF records where necessary
#                         (default: auto, attempt to match reference).
#   --no-fixchr           Do not add chr prefix to VCF records (default: auto,
#                         attempt to match reference).
#   --bcf                 Use BCF internally. This is the default when the input
#                         file is in BCF format already. Using BCF can speed up
#                         temp file access, but may fail for VCF files that have
#                         broken headers or records that don't comply with the
#                         header.
#   --somatic             Assume the input file is a somatic call file and
#                         squash all columns into one, putting all FORMATs into
#                         INFO + use half genotypes (see also --set-gt). This
#                         will replace all sample columns and replace them with
#                         a single one.
#   --set-gt {half,hemi,het,hom,first}
#                         This is used to treat Strelka somatic files Possible
#                         values for this parameter: half / hemi / het / hom /
#                         half to assign one of the following genotypes to the
#                         resulting sample: 1 | 0/1 | 1/1 | ./1. This will
#                         replace all sample columns and replace them with a
#                         single one.
#   --gender {male,female,auto,none}
#                         Specify gender. This determines how haploid calls on
#                         chrX get treated: for male samples, all non-ref calls
#                         (in the truthset only when running through hap.py) are
#                         given a 1/1 genotype.
#   --preprocess-truth    Preprocess truth file with same settings as query
#                         (default is to accept truth in original format).
#   --usefiltered-truth   Use filtered variant calls in truth file (by default,
#                         only PASS calls in the truth file are used)
#   --preprocessing-window-size PREPROCESS_WINDOW
#                         Preprocessing window size (variants further apart than
#                         that size are not expected to interfere).
#   --adjust-conf-regions
#                         Adjust confident regions to include variant locations.
#                         Note this will only include variants that are included
#                         in the CONF regions already when viewing with
#                         bcftools; this option only makes sure insertions are
#                         padded correctly in the CONF regions (to capture
#                         these, both the base before and after must be
#                         contained in the bed file).
#   --no-adjust-conf-regions
#                         Do not adjust confident regions for insertions.
#   --unhappy, --no-haplotype-comparison
#                         Disable haplotype comparison (only count direct GT
#                         matches as TP).
#   -w WINDOW, --window-size WINDOW
#                         Minimum distance between variants such that they fall
#                         into the same superlocus.
#   --xcmp-enumeration-threshold MAX_ENUM
#                         Enumeration threshold / maximum number of sequences to
#                         enumerate per block.
#   --xcmp-expand-hapblocks HB_EXPAND
#                         Expand haplotype blocks by this many basepairs left
#                         and right.
#   --threads THREADS     Number of threads to use.
#   --engine {xcmp,vcfeval,scmp-somatic,scmp-distance}
#                         Comparison engine to use.
#   --engine-vcfeval-path ENGINE_VCFEVAL
#                         This parameter should give the path to the "rtg"
#                         executable. The default is
#                         /opt/hap.py/lib/python27/Haplo/../../../libexec/rtg-
#                         tools-install/rtg
#   --engine-vcfeval-template ENGINE_VCFEVAL_TEMPLATE
#                         Vcfeval needs the reference sequence formatted in its
#                         own file format (SDF -- run rtg format -o ref.SDF
#                         ref.fa). You can specify this here to save time when
#                         running hap.py with vcfeval. If no SDF folder is
#                         specified, hap.py will create a temporary one.
#   --scmp-distance ENGINE_SCMP_DISTANCE
#                         For distance-based matching, this is the istance
#                         between variants to use.
#   --logfile LOGFILE     Write logging information into file rather than to
#                         stderr
#   --verbose             Raise logging level from warning to info.
#   --quiet               Set logging level to output errors only.
