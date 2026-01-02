#!/usr/bin/env python3


from codecs import raw_unicode_escape_encode
import lmdb
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
import os
import sys
from pathlib import Path
import io
import lz4.frame
from sympy.logic.boolalg import true
import torch
import torch.multiprocessing as mp
import msgpack
import numpy as np
from multiprocessing import Pool, Queue, Process
from humanfriendly import parse_size, InvalidSize
from typing import Generator
import logging



logging.basicConfig(format='[%(asctime)s] %(process)d  %(name)s  %(levelname)s  %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


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
    

def tensor_from_lz4(path):
    with open(path, 'rb') as f:
        raw = f.read()
    with io.BytesIO(lz4.frame.decompress(raw)) as bfh:
        return torch.load(bfh, map_location='cpu')


def process_item_worker(src_path: str, tgt_path: str, tntgt_path: str, item_name: str, queue: Queue, max_queue_size: int = 1024):
    """
    Worker function to process a single item and put results in queue one at a time.
    
    Args:
        src_path, tgt_path, tntgt_path: File paths
        item_name: Name of the item
        device: Device to load tensors on
        queue: Multiprocessing queue to put results in
        max_queue_size: Maximum queue size (for backpressure)
    """
    try:
        # Load and decompress data
        reads_data = tensor_from_lz4(src_path)
        tgkmers_data = tensor_from_lz4(tgt_path) 
        tntgt_data = tensor_from_lz4(tntgt_path)
        
        # Process each batch item one at a time
        for j, (read, tgkmers, tntgt) in enumerate(zip(reads_data, tgkmers_data, tntgt_data)):
            # print(f"Read shape {read.shape}, dtype {read.dtype}")        
            np_read = read.numpy()    
            np_tgkmers = tgkmers.numpy()
            item_data = {
                "source_name": item_name,
                "batch_item": j,
                "read":  {
                    "data": np_read.tobytes(order='C'),
                    "shape": np_read.shape,
                    "dtype": str(np_read.dtype),    
                    "order": 'C',
                },
                "tgkmers": {
                    "data": np_tgkmers.tobytes(order='C'),
                    "shape": np_tgkmers.shape,
                    "dtype": str(np_tgkmers.dtype),
                    "order": 'C',
                },
                "tntgt":  tntgt.item(),
            }
            
            # Serialize and compress
            key = f"item:{item_name}-{j}".encode()
            
            serialized_data = msgpack.packb(item_data, use_bin_type=True)
            compressed_data = lz4.frame.compress(serialized_data)
       
            queue.put((key, compressed_data))
        
        # Signal completion for this item
        queue.put(("COMPLETED", item_name))
        
    except Exception as e:
        # Put error in queue
        queue.put(("ERROR", f"Error processing {item_name}: {str(e)}"))
        logger.error(f"Error in worker processing {item_name}: {e}")


def process_items(items: List[Dict[str, str]],  queue: Queue):
    for item in items:
        process_item_worker(item["src_path"], item["tgt_path"], item["tntgt_path"], item["item_name"], queue)


class LMDB:
    """Example class demonstrating LMDB usage with genomic data."""
    
    def __init__(self, db_path: str = "example_lmdb"):
        """Initialize the LMDB example with a database path."""
        self.db_path = db_path
        self.env = None
        
    def init_database(self, map_size: int = 1024**3) -> None:
        """Create and open an LMDB database."""
        logger.info(f"Creating LMDB database at: {self.db_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Open LMDB environment
        self.env = lmdb.open(
            self.db_path,
            map_size=map_size,  # 1GB default
            max_dbs=10,  # Allow multiple named databases
            writemap=True  # Enable write mapping for better performance
        )
        logger.info("✓ Database created successfully")
    
    def load_tensor_data(self, file_path: str, device: str = "cpu", decompress: bool = True):
        """Load tensor data from a file. For .lz4 files, can return compressed bytes or decompressed tensor."""
        file_path = Path(file_path)
        
        if file_path.suffix == '.lz4':
            if decompress:
                return tensor_from_lz4(file_path, device)
            else:
                with open(file_path, 'rb') as f:
                    return f.read()
                
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
    
    
    def add_data_from_directory(self, datadir: str, device: str = "cpu", num_processes: int = None, max_queue_size: int = 1024, filter_file: str = None) -> None:
        """Add data from directory using find_files function with multiprocessing and bounded queue."""
        logger.info(f"Loading data from directory: {datadir}")
        if filter_file:
            logger.info(f"Reading filter file: {filter_file}")
            with open(filter_file, 'r') as f:
                filters = [line.strip() for line in f.readlines()]
            logger.info(f"Found {len(filters)} filters")
        else:
            filters = None
        
        if not os.path.exists(datadir):
            raise ValueError(f"Directory does not exist: {datadir}")
        
        file_tuples = list(find_files(datadir))
        if not file_tuples:
            logger.warning("No matching files found in directory")
            return

        logger.info(f"Found {len(file_tuples)} file tuples before filters")
        chunks_to_process = [[] for _ in range(num_processes)]
        for i, (src_path, tgt_path, tntgt_path) in enumerate(file_tuples):
            item_name = self.extract_item_name(str(src_path)).rstrip(".pt")
            if filters and not any(filter in item_name for filter in filters):
                logger.debug(f"Skipping item {item_name} because it is not in the filter file")
                continue

            chunks_to_process[i % num_processes].append({
                "src_path": src_path,
                "tgt_path": tgt_path,
                "tntgt_path": tntgt_path,
                "item_name": item_name
            })
        tot_files = sum(len(chunk) for chunk in chunks_to_process)
        logger.info(f"Found {tot_files} file tuples after filtering")

                
        logger.info(f"Using {num_processes} processes for data loading with queue size limit: {max_queue_size}")
        
        # Create shared queue
        queue = Queue(maxsize=max_queue_size)
        
        # Start worker processes
        processes = []
        for chunk in chunks_to_process:
            p = Process(
                target=process_items,
                args=(chunk, queue)
            )
            p.start()
            processes.append(p)
        
        # Consumer: read from queue and insert into database
        logger.info("Processing items and inserting into database...")
        total_items = 0
        completed_items = 0
        errors = []
        
        with self.env.begin(write=True) as txn:
            while completed_items < tot_files:
                # Get item from queue with timeout
                key, data = queue.get(timeout=30)  # 30 second timeout
                
                if key == "COMPLETED":
                    completed_items += 1
                    logger.info(f"Completed processing item {completed_items}/{tot_files}: {data}")
                elif key == "ERROR":
                    errors.append(data)
                    logger.error(f"Error: {data}")
                else:
                    # Insert into database
                    txn.put(f"item:{total_items}".encode(), data)
                    total_items += 1
                    if total_items % 5000 == 0:  # Progress update every 100 items
                        logger.info(f"  Inserted {total_items} items so far...")
        
        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=5)  # 5 second timeout
            if p.is_alive():
                logger.warning(f"Process {p.pid} did not terminate cleanly")
                p.terminate()
                p.join()
        
        # Add metadata
        with self.env.begin(write=True) as txn:
            metadata = {
                "total_files": tot_files,
                "total_batch_items": total_items,
                "datadir": datadir,
                "errors": errors
            }
            txn.put(b"metadata", json.dumps(metadata).encode())
        
        if errors:
            logger.warning(f"Completed with {len(errors)} errors:")
            for error in errors:
                logger.warning(f"  - {error}")
        
        logger.info(f"✓ Data loaded successfully: {total_items} batch items from {len(file_tuples)} files")
    
    def query_item(self, item_name: str, decompress: bool = true) -> Optional[Dict[str, Any]]:
        """Query for a specific item by name."""
        with self.env.begin() as txn:
            key = item_name.encode()
            value = txn.get(key)
            
            if value:
                # Decompress LZ4 data and deserialize with msgpack
                decompressed_data = lz4.frame.decompress(value)
                raw_item_data = msgpack.unpackb(decompressed_data, raw=False)
                if decompress:
                    # Decompress tensors if requested
                    item_data = self._hydrate_items(raw_item_data)
                return item_data
            return None
    
    def _hydrate_items(self, item_data: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
        """Helper method to decompress tensors in an item."""
        
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
    
    def query_all_items(self, decompress: bool = False, device: str = "cpu") -> List[Dict[str, Any]]:
        """Query all items in the database."""
        items = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if key.startswith(b"item:"):
                    # Decompress LZ4 data and deserialize with msgpack
                    decompressed_data = lz4.frame.decompress(value)
                    item_data = msgpack.unpackb(decompressed_data, raw=False)
                    if decompress:
                        item_data = self.hydrate_items(item_data, device)
                    items.append(item_data)
        return items

    
    def iterate_keys(self) -> Generator[str, None, None]:
        """Iterate over all keys in the database."""
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                yield key.decode()
    
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
                    # Decompress LZ4 data and deserialize with msgpack
                    decompressed_data = lz4.frame.decompress(value)
                    item_data = msgpack.unpackb(decompressed_data, raw=False)
                    
                    # Calculate tensor elements from tensor data
                    for tensor_name in ["read", "tgkmers"]:
                        if tensor_name in item_data and isinstance(item_data[tensor_name], dict):
                            tensor_info = item_data[tensor_name]
                            if "shape" in tensor_info:
                                shape = tensor_info["shape"]
                                elements = 1
                                for dim in shape:
                                    elements *= dim
                                total_tensor_elements += elements
                    
                    # tntgt is a scalar, count as 1 element
                    if "tntgt" in item_data:
                        total_tensor_elements += 1
                    
                    # Calculate compressed size for tensor data
                    for tensor_name in ["read", "tgkmers"]:
                        if tensor_name in item_data and isinstance(item_data[tensor_name], dict):
                            tensor_info = item_data[tensor_name]
                            if "data" in tensor_info and isinstance(tensor_info["data"], bytes):
                                total_compressed_size += len(tensor_info["data"])
            
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
                    # Decompress LZ4 data and deserialize with msgpack
                    decompressed_data = lz4.frame.decompress(value)
                    results[item_name] = msgpack.unpackb(decompressed_data, raw=False)
                else:
                    results[item_name] = None
        return results
    
    
    def close(self):
        """Close the database connection."""
        if self.env:
            self.env.close()
            logger.info("✓ Database connection closed")



def main():
    """Main function to run the LMDB example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LMDB Example Script")
    parser.add_argument("--db-path", "-p", type=str, help="Path to the database", required=True)
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    create_parser = subparsers.add_parser('create', help='Create database and load data')
    create_parser.add_argument("--datadir", "-d", type=str, required=True, help="Directory containing data files to load")
    create_parser.add_argument("--map-size", "-m", type=str, default="1G", 
                              help="Map size for the database (e.g., '1G', '100M', '1GiB', '500MB', '1GB')")
    create_parser.add_argument("--num-processes", "-n", type=int, default=4,
                              help="Number of processes to use for data loading (default: 4")
    create_parser.add_argument("--max-queue-size", "-q", type=int, default=1024,
                              help="Maximum queue size for producer-consumer pattern (default: 1024)")
    create_parser.add_argument("--filter-file", "-f", type=str, help="File to read filter patterns from")
    
    stats_parser = subparsers.add_parser('stats', help='Display database statistics')
    
    keys_parser = subparsers.add_parser('keys', help='List all keys in the database')
    keys_parser.add_argument("--filter", "-f", type=str, help="Filter keys by prefix (e.g., 'item:')")
    keys_parser.add_argument("--limit", "-l", type=int, help="Limit number of keys to display")
    
    args = parser.parse_args()
    
    if args.command == 'create':
        # Parse human-readable map size
        try:
            map_size_bytes = parse_size(args.map_size)
        except InvalidSize as e:
            logger.error(f"Error parsing map size '{args.map_size}': {e}")
            sys.exit(1)
        
        # Set up multiprocessing for PyTorch tensors
        mp.set_start_method('spawn', force=True)
        
        # Create and run the example
        example = LMDB(args.db_path)
        logger.info(f"Creating database with map size: {args.map_size} ({map_size_bytes:,} bytes)")
        example.init_database(map_size=map_size_bytes)
        
        example.add_data_from_directory(args.datadir, 
                                        device='cpu', 
                                        num_processes=args.num_processes, 
                                        max_queue_size=args.max_queue_size,
                                        filter_file=args.filter_file)
        
        example.close()
        
    elif args.command == 'stats':
        # Display database statistics
        example = LMDB(args.db_path)
        try:
            example.init_database()  # Open existing database
            stats = example.get_database_stats()
            
            logger.info("=" * 60)
            logger.info("DATABASE STATISTICS")
            logger.info("=" * 60)
            
            if "metadata" in stats:
                metadata = stats["metadata"]
                logger.info(f"Database Version: {metadata.get('database_version', 'Unknown')}")
                logger.info(f"Created By: {metadata.get('created_by', 'Unknown')}")
                logger.info(f"Data Directory: {metadata.get('datadir', 'Unknown')}")
                logger.info(f"Device: {metadata.get('device', 'Unknown')}")
                logger.info(f"Number of Processes Used: {metadata.get('num_processes', 'Unknown')}")
                logger.info(f"Max Queue Size: {metadata.get('max_queue_size', 'Unknown')}")
                logger.info("")
            
            logger.info(f"Total File Items: {stats.get('total_items', 0)}")
            logger.info(f"Total Batch Items: {stats.get('total_batch_items', 0)}")
            logger.info(f"Total Entries: {stats.get('total_entries', 0)}")
            logger.info(f"Total Tensor Elements: {stats.get('total_tensor_elements', 0):,}")
            logger.info(f"Compressed Items: {stats.get('compressed_items', 0)}")
            logger.info(f"Total Compressed Size: {stats.get('total_compressed_bytes', 0):,} bytes")
            
            if stats.get('total_compressed_bytes', 0) > 0:
                size_mb = stats['total_compressed_bytes'] / (1024 * 1024)
                logger.info(f"Total Compressed Size: {size_mb:.2f} MB")
            
            if "errors" in stats and stats["errors"]:
                logger.warning(f"Errors: {len(stats['errors'])}")
                for error in stats["errors"]:
                    logger.warning(f"  - {error}")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error reading database: {e}")
            sys.exit(1)
        finally:
            example.close()
            
    elif args.command == 'keys':
        # Display database keys
        example = LMDB(args.db_path)
        try:
            example.init_database()  # Open existing database
            for key in example.iterate_keys():
                logger.info(key)
            
        except Exception as e:
            logger.error(f"Error reading database: {e}")
            sys.exit(1)
        finally:
            example.close()
            
    else:
        parser.print_help()
        


if __name__ == "__main__":
    main()
