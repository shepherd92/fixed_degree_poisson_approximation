from logging import info
from multiprocessing import Pool, Value
from typing import Any, Callable

from tqdm import tqdm


NUM_OF_SIMULATIONS = 0
SIMULATIONS_DONE = Value('i', 0)
PROGRESS_BAR = None


def multiprocessing(
    num_of_processes: int,
    function: Callable,
    argument_list: list[Any],
    progress_bar: bool = True,
) -> list[Any]:
    global NUM_OF_SIMULATIONS
    NUM_OF_SIMULATIONS = len(argument_list)

    if progress_bar:
        global PROGRESS_BAR
        PROGRESS_BAR = tqdm(total=NUM_OF_SIMULATIONS, desc='Simulation')

        def progress_callback(_):
            if progress_bar:
                with SIMULATIONS_DONE.get_lock():
                    SIMULATIONS_DONE.value += 1
                    PROGRESS_BAR.n = SIMULATIONS_DONE.value
                    PROGRESS_BAR.refresh()

    if num_of_processes > 1:

        with Pool(num_of_processes) as pool:
            info(f'Starting a pool of {num_of_processes} processes.')
            # results: list[Any] = pool.starmap(function, argument_list)
            async_results = [
                pool.apply_async(function, argument, callback=progress_callback)
                for argument in argument_list
            ]
            results = [res.get() for res in async_results]
            info(f'Multiprocessing with {num_of_processes} processes finished.')
    else:
        results = []
        for args in argument_list:
            results.append(function(*args,))
            if progress_bar:
                progress_callback(None)

    PROGRESS_BAR.close()
    return results


# def f(a: int) -> int:
#     print(a)
#     return a + 1
# 
# 
# results = multiprocessing(1, f, [[3], [4]], True)
# print(results)
