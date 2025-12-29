
from dnaseq2seq import util
from dnaseq2seq.training.lmdbdataset import LMDBDataset
from torch.utils.data import Dataset


class VarsInfoDataWrapper:
    def __init__(self, dataset: Dataset):
        assert isinstance(dataset, Dataset), "dataset must be a Dataset"
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        assert "tgkmers" in item
        assert "tntgt" in item
        assert "read" in item

        ref = util.readstr(item['read'][:, 0, :])
        tgkmers = item['tgkmers']
        tntgt = item['tntgt']

        tgt_hap0 = util.kmer_preds_to_seq(tgkmers[0, 1:], util.i2s)
        tgt_hap1 = util.kmer_preds_to_seq(tgkmers[1, 1:], util.i2s)

        min_len = min(len(ref), len(tgt_hap0), len(tgt_hap1))
        hap1_match = tgt_hap1[:min_len] == ref[:min_len]
        hap0_match = tgt_hap0[:min_len] == ref[:min_len]
        hap0_hap1_match = tgt_hap0[:min_len] == tgt_hap1[:min_len]

        return {
            **item,
            "hap1_ref_match": hap1_match,
            "hap0_ref_match": hap0_match,
            "hap0_hap1_match": hap0_hap1_match,
        }
        

if __name__ == "__main__":
    dataset = LMDBDataset("/mnt/ri_share/Data/variant-transformer/pregen/good44_part01_lmdb")
    wrapper = VarsInfoDataWrapper(dataset)
    
    x = wrapper[7]
    print(x)