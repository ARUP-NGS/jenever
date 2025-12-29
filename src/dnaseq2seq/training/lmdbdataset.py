import lmdb
import json
import msgpack
import numpy as np
from typing import Dict, Any, List
from torch.utils.data import Dataset
import threading
import os

import torch
import lz4.frame

class LMDBDataset(Dataset):
    """
    PyTorch Dataset for accessing LMDB database.
    
    This dataset provides access to items stored in an LMDB database where:
    - The length is determined by querying the metadata 'total_items' value
    - Each item is accessed by key "item:{i}" where i is the index
    - Data is automatically decompressed and deserialized
    """
    
    def __init__(self, db_path: str, device: str = "cpu", max_read_depth: int = -1, max_readers: int = 126):
        """
        Initialize the LMDB Dataset.
        
        Args:
            db_path: Path to the LMDB database
            device: Device to load tensors on (default: "cpu")
            max_read_depth: Maximum read depth to load (default: -1)
            max_readers: Maximum number of LMDB readers (default: 126)
        """
        self.db_path = db_path
        self.device = device
        self.env = None
        self._length = None
        self.max_read_depth = max_read_depth
        self.max_readers = max_readers
        self._local = None
        

    def _create_env(self):
        return lmdb.open(
                self.db_path, 
                readonly=True,
                max_readers=self.max_readers,
                lock=False,  # Disable locking for readonly access
                readahead=False,  # Disable readahead to reduce memory usage
                meminit=False,  # Don't initialize memory
                writemap=False,  # Don't use writemap for readonly
                subdir=True,  # Allow subdirectories
                create=False  # Don't create if doesn't exist
            )

    def _get_or_create_local_env(self):
        """
        Get or create LMDB environment and store it in thread-local storage
        Note that these cannot be pickled, so we need to create a new one for each worker
        """
        # Use thread-local storage to ensure each worker has its own environment
        if self._local is None:
            self._local = threading.local()
        if not hasattr(self._local, 'env') or self._local.env is None:
            # Configure LMDB for multi-worker usage
            self._local.env = self._create_env()
        return self._local.env
    
    def __len__(self) -> int:
        """Return the length of the dataset by querying metadata 'total_items'."""
        if self._length is None:
            lmdb_env = self._create_env()
            with lmdb_env.begin() as txn:
                metadata_value = txn.get(b"metadata")
                if metadata_value:
                    metadata = json.loads(metadata_value.decode())
                    self._length = metadata.get("total_batch_items", -1)
                else:
                    raise ValueError("Metadata not found in database")
        return self._length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item at index idx.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing the item data
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of range for dataset of length {len(self)}")
        
        # Use a more efficient transaction handling
    
        with self._get_or_create_local_env().begin() as txn:
            key = f"item:{idx}".encode()
            value = txn.get(key)
            
            if value is None:
                raise KeyError(f"Item with key 'item:{idx}' not found in database")
            
            # Decompress LZ4 data and deserialize with pickle
            decompressed_data = lz4.frame.decompress(value)
            raw_item_data = msgpack.unpackb(decompressed_data, raw=False)
            item_data = self._hydrate_items(raw_item_data)
            return item_data

    def _hydrate_items(self, item_data: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
        """Helper method to serialize tensors in an item."""
        
        read_info = item_data["read"]
        # Convert bytes back to tensor
        tensor_data = read_info["data"]
        shape = tuple(read_info["shape"])
        dtype = read_info["dtype"]
        order = read_info.get("order", "C")
        
        # Convert bytes to numpy array then to tensor
        np_array = np.frombuffer(tensor_data, dtype=dtype).reshape(shape)
        if order == "F":  # Fortran order
            np_array = np_array.T
        item_data["read"] = torch.from_numpy(np_array).to(device)
        if self.max_read_depth != -1:
            item_data["read"] = item_data["read"][:, 0:self.max_read_depth, :]
        

        tgkmers_info = item_data["tgkmers"]
        
        # Convert bytes back to tensor
        tensor_data = tgkmers_info["data"]
        shape = tuple(tgkmers_info["shape"])
        dtype = tgkmers_info["dtype"]
        order = tgkmers_info.get("order", "C")
        
        # Convert bytes to numpy array then to tensor
        np_array = np.frombuffer(tensor_data, dtype=dtype).reshape(shape)
        if order == "F":  # Fortran order
            np_array = np_array.T
        item_data["tgkmers"] = torch.from_numpy(np_array).to(device)

        return item_data

    def close(self):
        """Close the database connection."""
        # Close thread-local environment
        if hasattr(self._local, 'env') and self._local.env is not None:
            self._local.env.close()
            self._local.env = None
        
        # Also close the main environment if it exists
        if self.env:
            self.env.close()
            self.env = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()

def create_multi_lmdb_dataset(db_paths: List[str]) -> Dataset:
    """
    Create a multi-LMDB dataset from a list of LMDB database paths.
    
    Args:
        db_paths: List of LMDB database paths
    """
    return torch.utils.data.ConcatDataset([LMDBDataset(db_path) for db_path in db_paths])


if __name__ == "__main__":
    dataset = LMDBDataset("/data2/brendan/testlmdb")
    print(f"Dataset length: {len(dataset)}")
    item = dataset[17]  # Get first item
    print(f"Item keys: {list(item.keys())}")
    print(f"Reads tensor shape: {item['read'].shape}")
    print(f"TGKMers tensor shape: {item['tgkmers'].shape}")
    print(f"TNtgt : {item['tntgt']}")