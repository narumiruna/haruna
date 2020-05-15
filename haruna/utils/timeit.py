import time
import logging

import torch

from ..metrics import Average

logger = logging.getLogger(__name__)


def sync_perf_counter():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.perf_counter()


def timeit(func):
    average = Average()

    def timed(*args, **kwargs):
        start = sync_perf_counter()
        output = func(*args, **kwargs)
        t = sync_perf_counter() - start

        average.update(t)

        logger.debug('%s took %.6f seconds, average: %.6f seconds.', func.__qualname__, t, average.value)
        return output

    return timed
