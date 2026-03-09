from torch.multiprocessing import Queue, Event, Process, Manager, Value
from typing import Callable, Dict, Any, Iterable, List
from enum import Enum
import time
import random
import logging
import ctypes
from queue import Empty

from pprint import pprint
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s %(levelname)s %(message)s')

class StageStatus(Enum):
    NOT_STARTED = 'not_started'
    RUNNING = 'running'
    STOPPED = 'stopped'

class StageStopSignal:
    pass

class WorkerEndSignal:
    pass

class SkipResult:
    pass

@dataclass
class MultiResult:
    results: List[Any]
    
def _worker_run(
        stage_name: str,
        worker_index: int,
        target_func: Callable, 
        input_queue: Queue,
        output_queue: Queue, 
        should_stop: Event, 
        coordination: Dict[str, Any],
        worker_stats_list: List,
        custom_item_counter: Callable = None,
        halt_on_exception: bool = True,
        stats_update_interval: int = 1):
    """
    Pull items from the input queue, process them, and put the results on the output queue, with some basic stats tracking.
    This function is used by Stage to actually execute the target function on each item from the input queue, with 
    appropriate synchronization, stats tracking, and error handling.

    It pulls items from the input queue until either the StageStopSignal is received, or the WorkerEndSignal is received from all other workers.
    
    """
    logger.info(f"Worker {worker_index} of stage {stage_name} starting")
    abort = False

    if hasattr(target_func, 'worker_init'):
        target_func.worker_init(worker_index)

    local_items_received = 0
    local_items_processed = 0
    local_total_wait_time = 0.0
    local_total_process_time = 0.0
    local_custom_counter = 0
    local_items_skipped = 0
    local_first_item_time = -1.0
    items_since_last_sync = 0

    while True:
        wait_start = time.time()
        try:
            item = input_queue.get(timeout=5)
            logger.debug(f"Worker {worker_index} of stage {stage_name} got item")
        except Empty:
            logger.debug(f"Worker {worker_index} of stage {stage_name} empty queue, continuing")
            if should_stop.is_set():
                break
            else:
                continue
        except KeyboardInterrupt:
            logger.info(f"Worker {worker_index} of stage {stage_name} received keyboard interrupt")
            should_stop.set()
            break

        wait_time = time.time() - wait_start
        local_total_wait_time += wait_time
        
        if isinstance(item, StageStopSignal):
            abort = True
            break

        if isinstance(item, WorkerEndSignal):
            with coordination['worker_end_lock']:
                coordination['worker_end_signals_received'].value += 1
                if coordination['worker_end_signals_received'].value == coordination['worker_end_signals_expected'].value:
                    logger.info(f"Worker {worker_index} of stage {stage_name} received all worker end signals")
                    should_stop.set()
            continue

        if local_first_item_time < 0:
            local_first_item_time = time.time() - coordination['start_time'].value

        local_items_received += 1
        if custom_item_counter is not None:
            local_custom_counter += custom_item_counter(item)
        
        process_start = time.time()
        try:
            output_items_put = 0
            result = target_func(item)
            process_time = time.time() - process_start
            local_total_process_time += process_time

            if isinstance(result, MultiResult):
                for r in result.results:
                    output_queue.put(r)
                    output_items_put += 1
            elif isinstance(result, SkipResult):
                local_items_skipped += 1
            else:
                output_queue.put(result)
                output_items_put += 1

            local_items_processed += output_items_put
            items_since_last_sync += 1

            if items_since_last_sync >= stats_update_interval:
                worker_stats_list[worker_index] = {
                    'items_received': local_items_received,
                    'items_processed': local_items_processed,
                    'total_wait_time': local_total_wait_time,
                    'total_process_time': local_total_process_time,
                    'items_skipped': local_items_skipped,
                    'custom_counter': local_custom_counter,
                    'first_item_time': local_first_item_time,
                }
                items_since_last_sync = 0

        except Exception as e:
            process_time = time.time() - process_start
            local_total_process_time += process_time
            
            output_queue.put(WorkerEndSignal())
            if halt_on_exception:
                should_stop.set()
            raise e
    
    flush_items_produced = 0
    if not abort and hasattr(target_func, 'flush'):
        flush_start = time.time()
        result = target_func.flush()
        flush_time = time.time() - flush_start
        local_total_process_time += flush_time
        
        if isinstance(result, MultiResult):
            for r in result.results:
                output_queue.put(r)
                flush_items_produced += 1
        elif isinstance(result, SkipResult):
            local_items_skipped += 1
        else:
            output_queue.put(result)
            flush_items_produced += 1
        
        local_items_processed += flush_items_produced

    worker_stats_list[worker_index] = {
        'items_received': local_items_received,
        'items_processed': local_items_processed,
        'total_wait_time': local_total_wait_time,
        'total_process_time': local_total_process_time,
        'items_skipped': local_items_skipped,
        'custom_counter': local_custom_counter,
        'first_item_time': local_first_item_time,
    }

    logger.debug(f"Worker {worker_index} of stage {stage_name} putting worker end signal")
    output_queue.put(WorkerEndSignal())
    logger.info(f"Worker {worker_index} of stage {stage_name} finished")

    coordination['release_workers'].wait()
    logger.info(f"Worker {worker_index} of stage {stage_name} terminating")


class Stage:
    """
    A processing stage that runs a target function across one or more worker
    processes. Each worker maintains its own local counters; aggregate stats
    are computed on-demand by iterating worker slots when callers read ``stats``
    or ``get_stats()``.

    Only coordination primitives (end-signal tracking and worker release) are
    shared across workers.
    """

    def __init__(self, name: str, 
            target_func: Callable, 
            n_workers: int = 1, 
            input_queue_maxsize: int = 1000, 
            output_queue_maxsize: int = 1000,
            custom_item_counter: Callable = None,
            stats_update_interval: int = 10):
        self.name = name
        self.target_func = target_func
        self.n_workers = n_workers
        self.input_queue = Queue(maxsize=input_queue_maxsize)
        self.output_queue = Queue(maxsize=output_queue_maxsize)
        self.should_stop = Event()
        self.downstream_stage = None
        self.upstream_stage = None
        self.custom_item_counter = custom_item_counter
        self.stats_update_interval = stats_update_interval
        
        self.manager = Manager()
        
        self._coordination = {
            'start_time': Value(ctypes.c_double, 0.0),
            'worker_end_signals_received': Value(ctypes.c_int, 0),
            'worker_end_signals_expected': Value(ctypes.c_int, -1),
            'worker_end_lock': self.manager.Lock(),
            'release_workers': self.manager.Event(),
        }
        
        self.worker_stats_list = self.manager.list([{} for _ in range(n_workers)])
        
        self._status = StageStatus.NOT_STARTED
        self.workers = None

    def _aggregate_worker_stats(self) -> Dict[str, Any]:
        """Iterate all worker slots and sum counters into an aggregate dict."""
        totals = {
            'items_received': 0,
            'items_processed': 0,
            'custom_counter': 0,
            'items_skipped': 0,
            'total_wait_time': 0.0,
            'total_process_time': 0.0,
            'first_item_time': -1.0,
        }
        per_worker: Dict[str, Any] = {}

        for i, ws in enumerate(self.worker_stats_list):
            if not ws:
                continue
            totals['items_received'] += ws.get('items_received', 0)
            totals['items_processed'] += ws.get('items_processed', 0)
            totals['custom_counter'] += ws.get('custom_counter', 0)
            totals['items_skipped'] += ws.get('items_skipped', 0)
            totals['total_wait_time'] += ws.get('total_wait_time', 0.0)
            totals['total_process_time'] += ws.get('total_process_time', 0.0)

            wfit = ws.get('first_item_time', -1.0)
            if wfit >= 0:
                if totals['first_item_time'] < 0 or wfit < totals['first_item_time']:
                    totals['first_item_time'] = wfit

            per_worker[f'worker_{i}_items_received'] = ws.get('items_received', 0)
            per_worker[f'worker_{i}_items_processed'] = ws.get('items_processed', 0)
            per_worker[f'worker_{i}_wait_time'] = ws.get('total_wait_time', 0.0)
            per_worker[f'worker_{i}_process_time'] = ws.get('total_process_time', 0.0)
            per_worker[f'worker_{i}_items_skipped'] = ws.get('items_skipped', 0)

        return {**totals, **per_worker}

    def connect(self, downstream_stage: 'Stage'):
        if self._status != StageStatus.NOT_STARTED:
            raise Exception(f"Stage {self.name} is not in the not started state")
        self.downstream_stage = downstream_stage
        self.downstream_stage.input_queue = self.output_queue
        self.downstream_stage.upstream_stage = self
        self.downstream_stage.set_upstream_worker_count(self.n_workers)

    def set_upstream_worker_count(self, total_workers: int):
        self._coordination['worker_end_signals_expected'].value = total_workers

    def _init_workers(self):
        self.workers = [
            Process(
                target=_worker_run, 
                args=(
                    self.name, i, self.target_func, 
                    self.input_queue, self.output_queue, 
                    self.should_stop, self._coordination,
                    self.worker_stats_list,
                    self.custom_item_counter,
                    True,  # halt_on_exception
                    self.stats_update_interval,
                )
            )
            for i in range(self.n_workers)
        ]

    def run(self):
        self._init_workers()
        self._status = StageStatus.RUNNING
        self._coordination['start_time'].value = time.time()
        for i, worker in enumerate(self.workers):
            logger.info(f"Stage {self.name} starting worker {i}")
            worker.start()

    @property
    def status(self) -> StageStatus:
        return self._status

    @property
    def items_processed(self) -> int:
        return self._aggregate_worker_stats()['items_processed']

    def put(self, item: Any):
        if self._status == StageStatus.STOPPED:
            raise Exception(f"Stage {self.name} is stopped")
        self.input_queue.put(item)

    def get(self, timeout: float = None) -> Any:
        return self.output_queue.get(timeout=timeout)

    @property
    def stats(self) -> Dict[str, Any]:
        """
        Aggregate stats across all workers. Per-worker breakdowns are included
        as ``worker_<i>_*`` keys for rich display.
        """
        agg = self._aggregate_worker_stats()
        agg['total_workers'] = self.n_workers
        return agg

    def drain(self):
        """
        Drain results from this stage's *output* queue until all of this stage's workers have
        emitted their terminal `WorkerEndSignal()`.
        """
        items_drained = 0
        end_signals_seen = 0
        expected_end_signals = self.n_workers

        while end_signals_seen < expected_end_signals:
            try:
                result = self.get(timeout=1)
            except Empty:
                continue

            if isinstance(result, WorkerEndSignal):
                end_signals_seen += 1
                logger.info(
                    f"Stage {self.name} received worker end signal "
                    f"{end_signals_seen}/{expected_end_signals}"
                )
                continue

            items_drained += 1
            yield result

    def abort(self):
        logger.info(f"Stage {self.name} aborting")
        self.should_stop.set()
        for _ in range(self.n_workers):
            self.put(StageStopSignal())
        self._status = StageStatus.STOPPED

    def join(self, propogate_downstream: bool = False):
        self._coordination['release_workers'].set()
        for i, worker in enumerate(self.workers):
            logger.info(f"Stage {self.name} joining worker {i}")
            if worker.is_alive():
                logger.info(f"Stage {self.name} worker {i} is alive, joining...")
                worker.join()
                logger.info(f"Stage {self.name} joined worker {i}")
            else:
                logger.info(f"Stage {self.name} worker {i} is not alive, not joining")
        self._status = StageStatus.STOPPED
        if propogate_downstream and self.downstream_stage is not None:
            self.downstream_stage.join(propogate_downstream=propogate_downstream)

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregated statistics with derived averages."""
        agg = self._aggregate_worker_stats()
        items_received = agg['items_received']
        items_processed = agg['items_processed']

        agg['status'] = self._status.value
        agg['total_workers'] = self.n_workers
        agg['avg_wait_time'] = agg['total_wait_time'] / items_received if items_received > 0 else 0
        agg['avg_process_time'] = agg['total_process_time'] / items_processed if items_processed > 0 else 0
        return agg



def _initial_worker_run(
        stage_name: str,
        target_func: Callable, 
        output_queue: Queue, 
        should_stop: Event, 
        stats: Dict[str, Any],
        release_workers: Event,
        iterator_factory: Callable = None,
        iterator_kwargs: Dict[str, Any] = None,
        custom_item_counter: Callable = None):
    """
    Worker function for InitialStage that iterates over an iterable and puts items into output_queue.
    
    Since InitialStage always has exactly 1 worker, no synchronization is needed.
    We use a simple Manager dict for stats - direct writes with no batching or locks.
    """
    logger.info(f"Worker of stage {stage_name} starting")
    if iterator_factory is not None:
        iterator = iterator_factory(**iterator_kwargs)
    else:
        iterator = target_func
    
    begin_time = time.time()
    for item in iterator:
        end_time = time.time()
        
        # Direct updates - no locks needed since there's only 1 worker
        stats['total_process_time'] += end_time - begin_time
        stats['items_processed'] += 1
        if custom_item_counter is not None:
            stats['custom_counter'] += custom_item_counter(item)
        
        output_queue.put(item)
        
        if should_stop.is_set():
            logger.info(f"Worker of stage {stage_name} found should_stop signal, stopping")
            break
        begin_time = end_time
    
    logger.info(f"Worker of stage {stage_name} putting worker end signal")
    output_queue.put(WorkerEndSignal())
    release_workers.wait()  # Block here until release_workers event is set
    logger.info(f"Worker of stage {stage_name} released")


class InitialStage(Stage):
    """
    Initial stage that iterates over an iterable (or callable that returns an iterable)
    and puts items into the output queue. This stage has no input queue and exactly 1 worker.
    Since there's only 1 worker, we use a simple Manager dict
    for stats with no synchronization overhead - no atomic Values, no locks, no batching.
    """

    def __init__(self, name: str, target_func: Iterable[Any], output_queue_maxsize: int = 1000, 
                 iterator_factory: Callable = None, iterator_kwargs: Dict[str, Any] = None, 
                 custom_item_counter: Callable = None):
        # Don't call super().__init__ - we need simpler initialization for single-worker stage
        self.name = name
        self.target_func = target_func
        self.n_workers = 1
        self.input_queue = None  # InitialStage has no input queue
        self.output_queue = Queue(maxsize=output_queue_maxsize)
        self.should_stop = Event()
        self.downstream_stage = None
        self.upstream_stage = None
        self.iterator_factory = iterator_factory
        self.iterator_kwargs = iterator_kwargs
        self.custom_item_counter = custom_item_counter
        
        # Create Manager for shared state
        self.manager = Manager()
        
        # Simple dict for stats - no locks needed since there's only 1 worker
        self._stats = self.manager.dict({
            'items_processed': 0,
            'total_process_time': 0.0,
            'custom_counter': 0,
        })
        self._release_workers = self.manager.Event()
        
        self._status = StageStatus.NOT_STARTED
        self.workers = None

    def put(self, item: Any):
        raise Exception("InitialStage does not support put")

    def _init_workers(self):
        self.workers = [
            Process(
                target=_initial_worker_run, 
                args=(
                    self.name, 
                    self.target_func, 
                    self.output_queue, 
                    self.should_stop, 
                    self._stats,
                    self._release_workers,
                    self.iterator_factory, 
                    self.iterator_kwargs, 
                    self.custom_item_counter
                )
            )
        ]
    
    def run(self):
        """Start the initial stage worker."""
        self._init_workers()
        self._status = StageStatus.RUNNING
        logger.info(f"Stage {self.name} starting worker")
        self.workers[0].start()
    
    def abort(self):
        """
        Abort this stage in the middle of processing. This will cause the worker to stop right away.
        This does NOT terminate or join the worker, which will continue to wait until join() is called.
        """
        logger.info(f"Stage {self.name} aborting")
        self.should_stop.set()
        self._release_workers.set()
        self._status = StageStatus.STOPPED

    def join(self, propogate_downstream: bool = False):
        """Wait for the worker process to finish and join it."""
        self._release_workers.set()
        if self.workers[0].is_alive():
            self.workers[0].join()
        self._status = StageStatus.STOPPED
        if propogate_downstream and self.downstream_stage is not None:
            self.downstream_stage.join(propogate_downstream=propogate_downstream)

    @property
    def stats(self) -> Dict[str, Any]:
        """Property for compatibility with existing code that accesses stage.stats"""
        return {
            'items_processed': self._stats['items_processed'],
            'custom_counter': self._stats['custom_counter'],
            'total_process_time': self._stats['total_process_time'],
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Return a dictionary with current processing statistics."""
        items_processed = self._stats['items_processed']
        total_process_time = self._stats['total_process_time']
        
        return {
            'status': self._status.value,
            'total_workers': 1,
            'items_processed': items_processed,
            'total_process_time': total_process_time,
            'custom_counter': self._stats['custom_counter'],
            'avg_process_time': total_process_time / items_processed if items_processed > 0 else 0,
        }


class StatefulFunc:
    def __init__(self, group_size: int):
        self.group_size = group_size
        self.group_index = 0
        self.group = []

    def __call__(self, item):
        self.group.append(item)
        if len(self.group) == self.group_size:
            group = self.group
            self.group = []
            return MultiResult(group)
        else:
            return SkipResult()
    
    def flush(self):
        if len(self.group) > 0:
            group = self.group
            self.group = []
            return MultiResult(group)
        else:
            return SkipResult()
            

def target_func(item):
    time.sleep(random.random() * 1)
    # logger.info(f"Stage 1Processing item: {item}")
    return item

def target_func2(item):
    time.sleep(random.random() * 5)
    # logger.info(f"Stage 2 Processing item: {item}")
    return item * -1


def target_func3(item):
    time.sleep(random.random() * 2)
    # logger.info(f"Stage 3 Processing item: {item}")
    return item * 10

if __name__ == "__main__":
    stage = InitialStage("Uno", range(60))
    stage2 = Stage("Dos", target_func2, n_workers=5)
    stage3 = Stage("Tres", StatefulFunc(group_size=3), n_workers=2)
    stage4 = Stage("Cuatro", target_func3, n_workers=3)
    stage.connect(stage2)
    stage2.connect(stage3)
    stage3.connect(stage4)
    stage.run()
    stage2.run()
    stage3.run()
    stage4.run()
    
    results = []
    for result in stage4.drain():
        results.append(result)

    print(f"Results: {sorted(results)}")
    print(f"Results length: {len(results)}")
    
    # Display stats using the new helper functions
    from dnaseq2seq.calling.stats_display import print_stats
    stats_list = [stage.get_stats(), stage2.get_stats(), stage3.get_stats(), stage4.get_stats()]
    stage_names = ["Uno", "Dos", "Tres", "Cuatro"]
    print_stats(stats_list, stage_names)