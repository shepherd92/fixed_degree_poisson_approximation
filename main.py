#!/usr/bin/env python3

import cProfile
import io
import logging
from logging import basicConfig
from pathlib import Path
from pstats import Stats, SortKey
from subprocess import Popen, PIPE, call, check_output
from trace import Trace
import tracemalloc

import numpy as np
import pandas as pd

from point import generate_point_cloud
from plot import plot_typical_mark_vs_intensity
from tools.multiprocessing_tools import multiprocessing


def main(output_dir: Path) -> None:
    """Main function, entry point of the program."""
    num_of_processes = 4

    min_log_intensity = 1.
    max_log_intensity = 3.
    num_of_intensity_values = 10
    intensities = np.flip(10. ** np.linspace(
        min_log_intensity,
        max_log_intensity,
        num_of_intensity_values,
        endpoint=True,
    ))

    results = multiprocessing(
        num_of_processes,
        generate_point_cloud,
        list(zip(
            intensities,
            range(len(intensities)),
            [output_dir] * len(intensities)
        )),
    )

    results = sum(results, [])  # create a single list
    results = pd.DataFrame(results, columns=['intensity', 'typical_mark'])
    results.to_csv(output_dir / 'restuls.csv', index=False)
    typical_mark_vs_intensity_plot = plot_typical_mark_vs_intensity(results)
    typical_mark_vs_intensity_plot.savefig(output_dir / 'plot_typical_mark_vs_intensity.png')


def main_wrapper() -> None:
    """Wrap the main function to separate profiling, logging, etc. from actual algorithm."""
    output_dir = Path('../output')
    project_root_dir = Path(__file__).parent
    runtime_profiling = False
    memory_profiling = False
    tracing = False
    log_level = 'DEBUG'

    output_dir.mkdir(parents=True, exist_ok=True)

    basicConfig(
        filename=output_dir / 'log.txt',
        filemode='w',
        encoding='utf-8',
        level=log_level,
        format='%(asctime)s %(levelname)-8s %(filename)s.%(funcName)s.%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
    logging.captureWarnings(True)

    if runtime_profiling:
        runtime_profiler = cProfile.Profile()
        runtime_profiler.enable()

    if memory_profiling:
        tracemalloc.start()

    if tracing:
        tracer = Trace(
            trace=True,
            count=False,
        )
        tracer.runfunc(main, output_dir)
        tracer_results = tracer.results()
        tracer_results.write_results(show_missing=True, coverdir=".")

    else:
        main(output_dir)

    if memory_profiling:
        memory_usage_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        profile_output_dir = output_dir / 'profile_statistics'
        profile_output_dir.mkdir(parents=True, exist_ok=True)

        filter_ = tracemalloc.Filter(inclusive=True, filename_pattern=f'{str(project_root_dir)}/*')
        memory_statistics = memory_usage_snapshot.filter_traces([filter_]).statistics('lineno')
        memory_usage_snapshot.dump(profile_output_dir / 'memory_profile_results.prof')
        memory_profile_log_file = profile_output_dir / 'memory_profile_results.txt'
        with open(memory_profile_log_file, 'w', encoding='utf-8') as profile_log_file:
            profile_log_file.write('size,count,trace_back\n')
            for stat in memory_statistics:
                profile_log_file.write(f'{stat.size / 2**16},{stat.count},{stat.traceback}\n')
        tracemalloc.clear_traces()

    if runtime_profiling:
        runtime_profiler.disable()

        profile_output_dir = output_dir / 'profile_statistics'
        profile_output_dir.mkdir(parents=True, exist_ok=True)

        stream = io.StringIO()

        runtime_statistics = Stats(runtime_profiler, stream=stream).sort_stats(SortKey.CUMULATIVE)
        runtime_statistics.print_stats(str(project_root_dir), 100)
        runtime_statistics.dump_stats(profile_output_dir / 'runtime_profile_results.prof')
        runtime_profile_log_file = profile_output_dir / 'runtime_profile_results.txt'
        with open(runtime_profile_log_file, 'w', encoding='utf-8') as profile_log_file:
            profile_log_file.write(stream.getvalue())

        with Popen(
            ('gprof2dot', '-f', 'pstats', profile_output_dir / 'runtime_profile_results.prof'),
            stdout=PIPE
        ) as gprof_process:
            check_output(
                ('dot', '-Tpng', '-o', profile_output_dir / 'runtime_profile_results.png'),
                stdin=gprof_process.stdout
            )
            gprof_process.wait()

        call(('snakeviz', '--server', profile_output_dir / 'runtime_profile_results.prof'))


if __name__ == '__main__':
    main_wrapper()
