
import os

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
import time

class TrainLogger:
    """ Simple utility for writing various items to a log file CSV """

    def __init__(self, output, headers):
        self.headers = list(headers)
        if type(output) == str:
            self.output = open(output, "a")
        else:
            self.output = output
        self._write_header()


    def _write_header(self):
        self.output.write(",".join(self.headers) + "\n")
        self._flush_and_fsync()

    def _flush_and_fsync(self):
        try:
            self.output.flush()
            os.fsync()
        except:
            pass

    def log(self, items):
        assert len(items) == len(self.headers), f"Expected {len(self.headers)} items to log, but got {len(items)}"
        self.output.write(
            ",".join(str(items[k]) for k in self.headers) + "\n"
        )
        self._flush_and_fsync()




class GradientMonitor:

    def __init__(self, model: torch.nn.Module, window_size: int = 100):
        """
        Initialize gradient monitoring system.

        Args:
            model: PyTorch model to monitor
            window_size: Number of batches to keep in moving average
        """
        self.model = model
        self.window_size = window_size
        self.grad_history = defaultdict(lambda: defaultdict(list))
        self.moving_averages = defaultdict(lambda: defaultdict(float))
        self.start_time = time.time()

        # Register hooks for all parameters
        self.handles = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(
                    lambda grad, name: self._grad_hook(grad, name)
                )
                self.handles.append(handle)

    def _grad_hook(self, grad: torch.Tensor, name: str) -> None:
        """Record gradient statistics for a parameter."""
        if grad is not None:
            batch_time = time.time() - self.start_time
            grad_norm = grad.norm().item()

            self.grad_history[name]['time'].append(batch_time)
            self.grad_history[name]['norm'].append(grad_norm)

            # Update moving average
            recent_norms = self.grad_history[name]['norm'][-self.window_size:]
            self.moving_averages[name]['norm'] = np.mean(recent_norms)

    def keys(self):
        """Get the names of all parameters being monitored."""
        return self.grad_history.keys()

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for each parameter's gradients."""
        stats = {}
        for name in self.grad_history:
            norms = self.grad_history[name]['norm']
            stats[name] = {
                'mean': np.mean(norms),
                'std': np.std(norms),
                'max': np.max(norms),
                'min': np.min(norms),
                'moving_avg': self.moving_averages[name]['norm']
            }
        return stats

    def clear_history(self):
        """Clear gradient history to free memory."""
        self.grad_history.clear()
        self.moving_averages.clear()

    def remove_hooks(self):
        """Remove gradient hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()