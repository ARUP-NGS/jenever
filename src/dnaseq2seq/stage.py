from multiprocessing import Queue, Event, Process, Manager
from typing import Callable, Dict, Any, Iterable, List
from enum import Enum
import time
import random
import logging
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
        stats: Dict[str, Any],
        halt_on_exception: bool = True):
    """Worker function that processes items from input_queue and puts results in output_queue."""
    logger.info(f"Worker {worker_index} of stage {stage_name} starting")
    abort = False
    while True:
        # Track time waiting for item from queue
        wait_start = time.time()
        try:
            item = input_queue.get(timeout=1)
            logger.debug(f"Worker {worker_index} of stage {stage_name} got item: {item}")
        except Empty:
            logger.info(f"Worker {worker_index} of stage {stage_name} empty queue, continuing")
            if should_stop.is_set():
                break
            else:
                continue

        wait_time = time.time() - wait_start
        
        if isinstance(item, StageStopSignal):
            abort = True
            break

        if isinstance(item, WorkerEndSignal):
            with stats['lock']:
                stats['worker_end_signals_received'] += 1
            if stats['worker_end_signals_received'] == stats['worker_end_signals_expected']:
                logger.info(f"Worker {worker_index} of stage {stage_name} received all {stats['worker_end_signals_expected']} worker end signals, stopping stage")
                should_stop.set()
            continue

        with stats['lock']:
            stats['total_wait_time'] += wait_time
            stats['items_received'] += 1
        
        # Track time processing the item
        process_start = time.time()
        try:
            output_items_put = 0
            result = target_func(item)
            process_time = time.time() - process_start

            logger.debug(f"Worker {worker_index} of stage {stage_name} put item")
            if isinstance(result, MultiResult):
                for result in result.results:
                    output_queue.put(result)
                    output_items_put += 1
            elif isinstance(result, SkipResult):
                logger.info(f"Worker {worker_index} of stage {stage_name} skipping result: {result}")
                with stats['lock']:
                    stats['items_skipped'] += 1
            else:
                output_queue.put(result)
                output_items_put += 1

            # Update processing stats - it ain't processed until after its put in the output queue
            with stats['lock']:
                stats['total_process_time'] += process_time
                stats['items_processed'] += output_items_put
                stats['output_items_put'] += output_items_put
        except Exception as e:
            process_time = time.time() - process_start
            
            # Update exception stats
            with stats['lock']:
                stats['total_process_time'] += process_time
                exception_info = {
                    'item': str(item),
                    'exception_type': type(e).__name__,
                    'exception_message': str(e),
                    'timestamp': time.time()
                }
                stats['exceptions'].append(exception_info)
        
            output_queue.put(WorkerEndSignal())    
            if halt_on_exception:
                should_stop.set()

            raise e
    
    # Some stateful functions may have a few more items to put into the output queue, even
    # after the last item was pulled from the input queue
    if not abort and hasattr(target_func, 'flush'):
        result = target_func.flush()
        if isinstance(result, MultiResult):
            for result in result.results:
                output_queue.put(result)
        elif isinstance(result, SkipResult):
            with stats['lock']:
                stats['items_skipped'] += 1
        else:
            output_queue.put(result)
            output_items_put += 1

    logger.info(f"Worker {worker_index} of stage {stage_name} putting worker end signal")
    output_queue.put(WorkerEndSignal())
    logger.info(f"Worker {worker_index} of stage {stage_name} finished and waiting for workers to be released")

    stats['release_workers'].wait() # Block here forever until the release_workers event is set
    logger.info(f"Worker {worker_index} of stage {stage_name} released workers")



class Stage:

    def __init__(self, name: str, target_func: Callable, n_workers: int = 1, input_queue_maxsize: int = 1000, output_queue_maxsize: int = 1000):
        self.name = name
        self.target_func = target_func
        self.n_workers = n_workers
        self.input_queue = Queue(maxsize=input_queue_maxsize)
        self.output_queue = Queue(maxsize=output_queue_maxsize)
        self.should_stop = Event()
        self.downstream_stage = None
        self.upstream_stage = None
        
        # Create Manager for shared state
        self.manager = Manager()
        self.stats = self.manager.dict({
            'total_wait_time': 0.0,
            'total_process_time': 0.0,
            'items_received': 0, # Counts items pulled from input queue
            'items_processed': 0, # Counts items put processsed by target function
            'items_skipped': 0, # Counts items skipped by target function
            'output_items_put': 0, # Counts items put into output queue
            'total_workers': n_workers,
            'worker_end_signals_received': 0,
            'worker_end_signals_expected': -1,
            'exceptions': self.manager.list(),
            'status': StageStatus.NOT_STARTED,
            'release_workers': self.manager.Event(),
        })
        self.stats['lock'] = self.manager.Lock()
        self.workers = None
        

    def connect(self, downstream_stage: 'Stage'):
        if self.status != StageStatus.NOT_STARTED:
            raise Exception(f"Stage {self.name} is not in the not started state")
        self.downstream_stage = downstream_stage
        self.downstream_stage.input_queue = self.output_queue
        self.downstream_stage.upstream_stage = self
        self.downstream_stage.set_upstream_worker_count(self.n_workers)

    def set_upstream_worker_count(self, total_workers: int):
        self.stats['worker_end_signals_expected'] = total_workers


    def _init_workers(self):
        self.workers = [
            Process(target=_worker_run, args=(self.name, i, self.target_func, self.input_queue, self.output_queue, self.should_stop, self.stats))
            for i in range(self.n_workers)
        ]

    @property
    def status(self) -> StageStatus:
        return self.stats['status']

    @property
    def items_processed(self) -> int:
        """
        Total numbers of items processed and put into output queue
        """
        with self.stats['lock']:
            return self.stats['items_processed']

    def put(self, item: Any):
        if self.stats['status'] == StageStatus.STOPPED:
            raise Exception(f"Stage {self.name} is stopped")
        self.input_queue.put(item) # This will block if the queue is full

    def get(self, timeout: float = None) -> Any:
        return self.output_queue.get(timeout=timeout)

    def drain(self):
        """
        Preferred method for obtaining all results
        """
        items_drained = 0
        if self.stats['worker_end_signals_expected'] == -1:
            raise Exception("Upstream worker count not set")
        
        # First, process until we receive all the worker end signals (one from each worker on the 'upstream' stage)
        # After this, we'll know that the upstream stage isn't producing any more items
        while self.stats['worker_end_signals_received'] < self.stats['worker_end_signals_expected']:
            logger.info(f"Stage {self.name} draining, worker end signal count: {self.stats['worker_end_signals_received']}, stage status: {self.status}")
            try:
                result = self.get(timeout=1)
                if not isinstance(result, WorkerEndSignal):
                    items_drained += 1
                    yield result
            except Empty:
                logger.info(f"Empty queue, worker end signal count: {self.stats['worker_end_signals_received']}, stage status: {self.status}")
                continue
        
        logger.info(f"Stage {self.name} workers finished, items processed: {self.stats['items_processed']}, items received: {self.stats['items_received']}")
        
        while items_drained < self.stats['output_items_put']:
            try:
                result = self.get(timeout=10)
                logger.info(f"Stage {self.name} draining last bits, got result: {result}")
                logger.info(f"Items drained: {items_drained}, items processed: {self.stats['output_items_put']}")
                if not isinstance(result, WorkerEndSignal):
                    items_drained += 1
                    yield result
            except Empty:
                logger.info(f"Empty queue, items drained: {items_drained}, items processed: {self.stats['output_items_put']}")
                continue
        
    def abort(self):
        """
        Abort this stage in the middle of processing. This will cause all workers to stop right away, even if there
        are more items to process in the input queue.
        This does NOT terminate or join the workers, and all workers will continue to wait until join() is called.
        """
        logger.info(f"Stage {self.name} aborting")
        self.should_stop.set()
        for _ in range(self.n_workers):
            self.put(StageStopSignal())
        self.stats['status'] = StageStatus.STOPPED

    def join(self, propogate_downstream: bool = False):
        """
        Wait for all worker processes to finish and join them. This sets the release_workers flag,
        meaning that workers will terminate and and clear associated resources (like tensors)

        If propogate_downstream is True, this will also signal the downstream stage to join, thus cascading
        across all stages in the pipeline.
        """
        self.stats['release_workers'].set()
        for i, worker in enumerate(self.workers):
            if worker.is_alive():
                logger.info(f"Joining worker {i} of stage {self.name}")
                worker.join()
            else:
                logger.info(f"Worker {i} of stage {self.name} is not alive, skipping, not joining it")
        if propogate_downstream and self.downstream_stage is not None:
            self.downstream_stage.join(propogate_downstream=propogate_downstream)
            
    def run(self):
        self._init_workers()
        self.stats['status'] = StageStatus.RUNNING
        for i, worker in enumerate(self.workers):
            logger.info(f"Stage {self.name} starting worker {i}")
            worker.start()
        

    def get_stats(self) -> Dict[str, Any]:
        """Return a dictionary with current processing statistics."""
        with self.stats['lock']:
            stats_dict = dict(self.stats)
            # Convert manager.list to regular list for exceptions
            stats_dict['exceptions'] = list(stats_dict['exceptions'])
            # Remove lock from returned dict (not serializable)
            stats_dict.pop('lock', None)
            
            # Calculate derived metrics
            if stats_dict['items_received'] > 0:
                stats_dict['avg_wait_time'] = 0.0 if stats_dict['items_received'] == 0 else stats_dict['total_wait_time'] / stats_dict['items_received']
                stats_dict['avg_process_time'] = 0.0 if stats_dict['items_processed'] == 0 else stats_dict['total_process_time'] / stats_dict['items_processed']
            
            return stats_dict


def _initial_worker_run(
        stage_name: str,
        worker_index: int,
        target_func: Callable, 
        output_queue: Queue, 
        should_stop: Event, 
        stats: Dict[str, Any],
        iterator_factory: Callable = None,
        iterator_kwargs: Dict[str, Any] = None):
    """Worker function that processes items from input_queue and puts results in output_queue."""
    logger.info(f"Worker {worker_index} of stage {stage_name} starting")
    if iterator_factory is not None:
        iterator = iterator_factory(**iterator_kwargs)
    else:
        iterator = target_func
    begin_time = time.time()
    for item in iterator:
        end_time = time.time()
        
        with stats['lock']:
            stats['total_process_time'] += end_time - begin_time
            stats['items_processed'] += 1
        
        output_queue.put(item)
        if should_stop.is_set():
            logger.info(f"Worker {worker_index} of stage {stage_name} found should_stop signal, stopping")
            break
        begin_time = end_time
    output_queue.put(WorkerEndSignal())
    stats['release_workers'].wait() # Block here forever until the release_workers event is set
    logger.info(f"Worker {worker_index} of stage {stage_name} released workers")


class InitialStage(Stage):

    def __init__(self, name: str, target_func: Iterable[Any], output_queue_maxsize: int = 1000, iterator_factory: Callable = None, iterator_kwargs: Dict[str, Any] = None):
        super().__init__(name, target_func, n_workers=1, input_queue_maxsize=1, output_queue_maxsize=output_queue_maxsize)
        self.input_queue = None
        self.iterator_factory = iterator_factory
        self.iterator_kwargs = iterator_kwargs
        self.stats = self.manager.dict({
            'total_process_time': 0.0,
            'items_processed': 0,
            'total_workers': 1,
            'exceptions': self.manager.list(),
            'status': StageStatus.NOT_STARTED,
            'lock': self.manager.Lock(),
            'release_workers': self.manager.Event(),
        })

    def put(self, item: Any):
        raise Exception("InitialStage does not support put")

    def _init_workers(self):
        self.workers = [
            Process(target=_initial_worker_run, args=(self.name, i, self.target_func, self.output_queue, self.should_stop, self.stats, self.iterator_factory, self.iterator_kwargs))
            for i in range(self.n_workers)  
        ]
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Return a dictionary with current processing statistics."""
        with self.stats['lock']:
            stats_dict = dict(self.stats)
            # Convert manager.list to regular list for exceptions
            stats_dict['exceptions'] = list(stats_dict['exceptions'])
            # Remove lock from returned dict (not serializable)
            stats_dict.pop('lock', None)
            
            # Calculate derived metrics
            if stats_dict['items_processed'] > 0:
                stats_dict['avg_process_time'] = 0.0 if stats_dict['items_processed'] == 0 else stats_dict['total_process_time'] / stats_dict['items_processed']
            
            return stats_dict


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
    pprint(stage.get_stats())
    pprint(stage2.get_stats())
    pprint(stage3.get_stats())
    pprint(stage4.get_stats())