import os
import torch
import heapq
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CheckpointRecord:
    """Data class to store checkpoint information"""
    value: float  # The metric value (e.g., validation loss)
    filepath: str  # Path to the checkpoint file
    step: int     # Training step when checkpoint was saved

    def __lt__(self, other):
        """Define comparison for heap operations (lower values are better)"""
        return self.value < other.value

class CheckpointManager:
    """
    Manages model checkpoints during training, keeping track of the best N checkpoints
    based on a specified metric (e.g., validation loss).
    
    Args:
        model: PyTorch model to checkpoint
        save_dir (str): Directory to save checkpoints
        max_checkpoints (int): Maximum number of checkpoints to retain
        minimize (bool, optional): If True, lower values are better. Default: True
    """
    def __init__(
        self,
        model: torch.nn.Module,
        save_dir: str,
        max_checkpoints: int,
        minimize: bool = True,
        save_prefix: str = "checkpoint"
    ):
        self.model = model
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.minimize = minimize
        self.save_prefix = save_prefix
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize heap for tracking best checkpoints
        # Use negative values if maximizing
        self.sign = 1 if minimize else -1
        self.checkpoints = []  # Will be used as a heap
        
        # Keep track of training step
        self.current_step = 0
    
    def step(
        self,
        value: float,
        step: Optional[int] = None,
        **kwargs: Dict[str, Any]
    ) -> bool:
        """
        Process a new checkpoint, saving if it's among the best N seen so far.
        
        Args:
            value (float): Metric value for this checkpoint (e.g., validation loss)
            step (int, optional): Current training step. If None, uses internal counter
            **kwargs: Additional items to save in checkpoint
            
        Returns:
            bool: True if checkpoint was saved, False otherwise
        """
        if step is None:
            step = self.current_step
            self.current_step += 1
            
        # Adjust value based on minimize/maximize
        heap_value = value * self.sign
        
        # Create checkpoint record
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_prefix}_step{step}_{timestamp}.pt"
        filepath = os.path.join(self.save_dir, filename)
        record = CheckpointRecord(heap_value, filepath, step)
        
        # If we haven't reached max checkpoints, always save
        if len(self.checkpoints) < self.max_checkpoints:
            self._save_checkpoint(record, kwargs)
            heapq.heappush(self.checkpoints, record)
            return True
            
        # Otherwise, only save if better than worst checkpoint
        # Find the worst checkpoint (max heap_value, since smaller is better after sign transform)
        worst = max(self.checkpoints, key=lambda x: x.value)
        if record < worst:  # Using __lt__ defined in CheckpointRecord
            # Remove old checkpoint file
            if os.path.exists(worst.filepath):
                os.remove(worst.filepath)
            
            # Save new checkpoint
            self._save_checkpoint(record, kwargs)
            
            # Replace worst checkpoint in heap
            # Find index of worst checkpoint
            worst_idx = self.checkpoints.index(worst)
            # Replace it with new record
            self.checkpoints[worst_idx] = record
            # Reheapify to maintain heap property
            heapq.heapify(self.checkpoints)
            return True
            
        return False
    
    def _save_checkpoint(self, record: CheckpointRecord, additional_items: Dict[str, Any]):
        """Save checkpoint with model state and additional items"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'step': record.step,
            'value': record.value * self.sign,  # Store original value
            **additional_items
        }
        torch.save(checkpoint, record.filepath)
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Returns path to the best checkpoint file"""
        if not self.checkpoints:
            return None
        # heap[0] is always the best (minimum heap_value after sign transform)
        return self.checkpoints[0].filepath
    
    def get_all_checkpoints(self) -> list[str]:
        """Returns list of all checkpoint files, sorted from worst to best"""
        return [record.filepath for record in sorted(self.checkpoints, reverse=True)]