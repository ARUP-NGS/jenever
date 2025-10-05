#!/usr/bin/env python3
"""
LMDB Example Script

This script demonstrates how to use LMDB (Lightning Memory-Mapped Database) for
storing and querying tensor data. LMDB is a fast, compact key-value database that's
particularly useful for machine learning applications and data processing pipelines.

Features demonstrated:
- Creating and opening LMDB databases
- Loading real tensor data from files (supports .pt and .lz4 formats)
- Using find_files function to match src/tgkmers/tntgt file tuples
- Querying for specific items by name
- Querying by tensor shape and path patterns
- Batch operations
- Error handling
- Database statistics

Usage:
    python scripts/lmdb_example.py --datadir /path/to/data
    python scripts/lmdb_example.py --help
"""

import lmdb
import json
import pickle
import random
import string
from typing import Dict, List, Any, Optional
import os
import sys
from pathlib import Path
import io
import lz4
import torch

def find_files(datadir, src_prefix='src', tgt_prefix='tgkmers', tn_prefix='tntgt'):
    """
    Examine files in datadir and match up all src / tgkmers / tntgt files and store them as tuples in a list
    :returns : List of (src, tgt, vaftgt) tuples of matched files
    """
    datadir = Path(datadir)
    allsrc = list(datadir.glob(src_prefix + "*"))
    for src in allsrc:
        suffix = src.name.split("_")[-1]
        yield (
            src, 
            f"{datadir}/{tgt_prefix}_{suffix}",
            f"{datadir}/{tn_prefix}_{suffix}",
            )
    


def tensor_from_lz4(path, device):
    with io.BytesIO(lz4.frame.decompress(path)) as bfh:
        return torch.load(bfh, map_location=device)

class LMDBExample:
    """Example class demonstrating LMDB usage with genomic data."""
    
    def __init__(self, db_path: str = "example_lmdb"):
        """Initialize the LMDB example with a database path."""
        self.db_path = db_path
        self.env = None
        
    def create_database(self, map_size: int = 1024**3) -> None:
        """Create and open an LMDB database."""
        print(f"Creating LMDB database at: {self.db_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Open LMDB environment
        self.env = lmdb.open(
            self.db_path,
            map_size=map_size,  # 1GB default
            max_dbs=10,  # Allow multiple named databases
            writemap=True  # Enable write mapping for better performance
        )
        print("✓ Database created successfully")
    
    def load_tensor_data(self, file_path: str, device: str = "cpu", store_compressed: bool = True):
        """Load tensor data from a file. For .lz4 files, can return compressed bytes or decompressed tensor."""
        file_path = Path(file_path)
        
        if file_path.suffix == '.lz4':
            if store_compressed:
                # Return compressed bytes for storage
                with open(file_path, 'rb') as f:
                    return f.read()
            else:
                # Decompress and return tensor
                return tensor_from_lz4(file_path, device)
        elif file_path.suffix == '.pt':
            # Load regular tensor
            return torch.load(file_path, map_location=device)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def extract_item_name(self, file_path: str) -> str:
        """Extract item name from file path by removing .pt/.lz4 suffixes and taking last underscore-separated element."""
        file_path = Path(file_path)
        # Remove .pt or .lz4 suffix
        name = file_path.name
        if name.endswith('.pt'):
            name = name[:-3]
        elif name.endswith('.lz4'):
            name = name[:-4]
        
        # Get last underscore-separated element
        return name.split('_')[-1]
    
    def decompress_tensor(self, data, device: str = "cpu") -> torch.Tensor:
        """Decompress tensor data if it's compressed bytes, otherwise return as-is."""
        if isinstance(data, bytes):
            return tensor_from_lz4(data, device)
        else:
            return data
    
    def add_data_from_directory(self, datadir: str, device: str = "cpu") -> None:
        """Add data from directory using find_files function."""
        print(f"\nLoading data from directory: {datadir}")
        
        if not os.path.exists(datadir):
            raise ValueError(f"Directory does not exist: {datadir}")
        
        file_tuples = list(find_files(datadir))
        print(f"Found {len(file_tuples)} file tuples")
        
        if not file_tuples:
            print("No matching files found in directory")
            return
        
        with self.env.begin(write=True) as txn:
            for i, (src_path, tgt_path, tntgt_path) in enumerate(file_tuples):
                try:
                    # Extract item name from src file
                    item_name = self.extract_item_name(str(src_path)).rstrip(".pt")
                    
                    # Load data (compressed bytes for .lz4, tensors for .pt)
                    reads_data = self.load_tensor_data(str(src_path), device, store_compressed=True)
                    tgkmers_data = self.load_tensor_data(str(tgt_path), device, store_compressed=True)
                    tntgt_data = self.load_tensor_data(str(tntgt_path), device, store_compressed=True)
                    
                    # Get shapes for metadata (need to decompress temporarily for .lz4 files)
                    def get_shape(data, file_path):
                        if isinstance(data, bytes):  # Compressed data
                            # Decompress temporarily to get shape
                            temp_tensor = tensor_from_lz4(data, device)
                            return list(temp_tensor.shape)
                        else:  # Regular tensor
                            return list(data.shape)
                    
                    reads_shape = get_shape(reads_data, str(src_path))
                    tgkmers_shape = get_shape(tgkmers_data, str(tgt_path))
                    tntgt_shape = get_shape(tntgt_data, str(tntgt_path))
                    
                    # Create data structure
                    item_data = {
                        "item_name": item_name,
                        "reads": reads_data,
                        "tgkmers": tgkmers_data,
                        "tntgt": tntgt_data,
                        "metadata": {
                            "src_path": str(src_path),
                            "tgt_path": str(tgt_path),
                            "tntgt_path": str(tntgt_path),
                            "reads_shape": reads_shape,
                            "tgkmers_shape": tgkmers_shape,
                            "tntgt_shape": tntgt_shape,
                            "reads_compressed": str(src_path).endswith('.lz4'),
                            "tgkmers_compressed": str(tgt_path).endswith('.lz4'),
                            "tntgt_compressed": str(tntgt_path).endswith('.lz4')
                        }
                    }
                    
                    # Store as pickle
                    key = f"item:{item_name}".encode()
                    value = pickle.dumps(item_data)
                    txn.put(key, value)
                    
                    print(f"  Loaded item {i+1}/{len(file_tuples)}: {item_name}")
                    
                except Exception as e:
                    print(f"  Error loading item from {src_path}: {e}")
                    continue
            
            # Add metadata
            metadata = {
                "total_items": len(file_tuples),
                "database_version": "1.0",
                "created_by": "lmdb_example.py",
                "datadir": datadir,
                "device": device
            }
            txn.put(b"metadata", json.dumps(metadata).encode())
        
        print("✓ Data loaded successfully")
    
    def query_item(self, item_name: str, decompress: bool = False, device: str = "cpu") -> Optional[Dict[str, Any]]:
        """Query for a specific item by name."""
        with self.env.begin() as txn:
            key = f"item:{item_name}".encode()
            value = txn.get(key)
            
            if value:
                item_data = pickle.loads(value)
                if decompress:
                    # Decompress tensors if requested
                    item_data = self._decompress_item_tensors(item_data, device)
                return item_data
            return None
    
    def _decompress_item_tensors(self, item_data: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
        """Helper method to decompress tensors in an item."""
        decompressed_data = item_data.copy()
        
        # Decompress each tensor if it's compressed
        for tensor_name in ["reads", "tgkmers", "tntgt"]:
            if tensor_name in item_data:
                decompressed_data[tensor_name] = self.decompress_tensor(item_data[tensor_name], device)
        
        return decompressed_data
    
    def query_all_items(self, decompress: bool = False, device: str = "cpu") -> List[Dict[str, Any]]:
        """Query all items in the database."""
        items = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if key.startswith(b"item:"):
                    item_data = pickle.loads(value)
                    if decompress:
                        item_data = self._decompress_item_tensors(item_data, device)
                    items.append(item_data)
        return items
    
    def get_decompressed_tensor(self, item_name: str, tensor_name: str, device: str = "cpu") -> Optional[torch.Tensor]:
        """Get a specific decompressed tensor from an item."""
        item = self.query_item(item_name, decompress=False)
        if item and tensor_name in item:
            return self.decompress_tensor(item[tensor_name], device)
        return None
    
    def query_items_by_shape(self, tensor_type: str, target_shape: List[int]) -> List[Dict[str, Any]]:
        """Query items by tensor shape for a specific tensor type (reads, tgkmers, tntgt)."""
        matching_items = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if key.startswith(b"item:"):
                    item_data = pickle.loads(value)
                    if tensor_type in item_data["metadata"]:
                        actual_shape = item_data["metadata"][f"{tensor_type}_shape"]
                        if actual_shape == target_shape:
                            matching_items.append(item_data)
        return matching_items
    
    def query_items_by_path_pattern(self, pattern: str) -> List[Dict[str, Any]]:
        """Query items by path pattern in metadata."""
        matching_items = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if key.startswith(b"item:"):
                    item_data = pickle.loads(value)
                    metadata = item_data["metadata"]
                    if (pattern in metadata.get("src_path", "") or 
                        pattern in metadata.get("tgt_path", "") or 
                        pattern in metadata.get("tntgt_path", "")):
                        matching_items.append(item_data)
        return matching_items
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        with self.env.begin() as txn:
            # Get metadata
            metadata_value = txn.get(b"metadata")
            if metadata_value:
                stats["metadata"] = json.loads(metadata_value.decode())
            
            # Count entries and calculate sizes
            item_count = 0
            total_tensor_elements = 0
            compressed_items = 0
            total_compressed_size = 0
            
            cursor = txn.cursor()
            for key, value in cursor:
                if key.startswith(b"item:"):
                    item_count += 1
                    item_data = pickle.loads(value)
                    metadata = item_data.get("metadata", {})
                    
                    # Check if any tensors are compressed
                    has_compressed = any([
                        metadata.get("reads_compressed", False),
                        metadata.get("tgkmers_compressed", False),
                        metadata.get("tntgt_compressed", False)
                    ])
                    
                    if has_compressed:
                        compressed_items += 1
                    
                    # Calculate tensor elements from metadata (no need to decompress)
                    for tensor_name in ["reads", "tgkmers", "tntgt"]:
                        shape_key = f"{tensor_name}_shape"
                        if shape_key in metadata:
                            shape = metadata[shape_key]
                            elements = 1
                            for dim in shape:
                                elements *= dim
                            total_tensor_elements += elements
                    
                    # Calculate compressed size for compressed tensors
                    for tensor_name in ["reads", "tgkmers", "tntgt"]:
                        if tensor_name in item_data:
                            tensor_data = item_data[tensor_name]
                            if isinstance(tensor_data, bytes):  # Compressed
                                total_compressed_size += len(tensor_data)
            
            stats["item_count"] = item_count
            stats["total_tensor_elements"] = total_tensor_elements
            stats["compressed_items"] = compressed_items
            stats["total_compressed_bytes"] = total_compressed_size
            stats["total_entries"] = item_count
        
        return stats
    
    def batch_query_items(self, item_names: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Batch query multiple items."""
        results = {}
        with self.env.begin() as txn:
            for item_name in item_names:
                key = f"item:{item_name}".encode()
                value = txn.get(key)
                if value:
                    results[item_name] = pickle.loads(value)
                else:
                    results[item_name] = None
        return results
    
    
    def close(self):
        """Close the database connection."""
        if self.env:
            self.env.close()
            print("✓ Database connection closed")


def main():
    """Main function to run the LMDB example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LMDB Example Script")
    parser.add_argument("--datadir", "-d", type=str, help="Directory containing data files to load")
    parser.add_argument("--device", type=str, default="cpu", help="Device for tensor loading (cpu/cuda)")
    args = parser.parse_args()
    
    try:
        # Create and run the example
        example = LMDBExample("example_lmdb")
        example.create_database()
        
        if args.datadir:
            example.add_data_from_directory(args.datadir, device=args.device)
            
            # Show database statistics
            print("\n" + "=" * 40)
            print("Database Statistics")
            print("=" * 40)
            stats = example.get_database_stats()
            for key, value in stats.items():
                if key == "metadata" and isinstance(value, dict):
                    print(f"{key}:")
                    for meta_key, meta_value in value.items():
                        print(f"  {meta_key}: {meta_value}")
                else:
                    print(f"{key}: {value}")
            
            # Show some items
            print("\n" + "=" * 40)
            print("Sample Items (compressed storage)")
            print("=" * 40)
            all_items = example.query_all_items(decompress=False)
            print(f"Found {len(all_items)} items")
            
            if all_items:
                for i, item in enumerate(all_items[:3]):
                    print(f"\nItem {i+1}: {item['item_name']}")
                    metadata = item['metadata']
                    print(f"  Reads: {metadata.get('reads_shape', 'N/A')} {'(compressed)' if metadata.get('reads_compressed') else '(uncompressed)'}")
                    print(f"  Tgkmers: {metadata.get('tgkmers_shape', 'N/A')} {'(compressed)' if metadata.get('tgkmers_compressed') else '(uncompressed)'}")
                    print(f"  Tntgt: {metadata.get('tntgt_shape', 'N/A')} {'(compressed)' if metadata.get('tntgt_compressed') else '(uncompressed)'}")
            
            # Demonstrate decompression
            if all_items:
                print("\n" + "=" * 40)
                print("Decompression Example")
                print("=" * 40)
                first_item = all_items[0]
                item_name = first_item['item_name']
                
                # Get decompressed tensor
                reads_tensor = example.get_decompressed_tensor(item_name, "reads", device=args.device)
                if reads_tensor is not None:
                    print(f"Decompressed reads tensor for {item_name}:")
                    print(f"  Shape: {reads_tensor.shape}")
                    print(f"  Data type: {reads_tensor.dtype}")
                    print(f"  Device: {reads_tensor.device}")
        else:
            print("No data directory provided. Database will be empty.")
        
        example.close()
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("Database files are stored in: example_lmdb/")
        if args.datadir:
            print(f"Data was loaded from: {args.datadir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running LMDB example: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
