import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import mapel
import csv

from mapel.main.features.distortion import calculate_distortion
from mapel.main.features.monotonicity import calculate_monotonicity
from mapel.main.features.stability import calculate_stability


def test_algorithm_on_dataset(experiment_id, algorithm, save_as, fixed_positions_path=None, **algorithm_options):
    instance_type = 'ordinal'
    distance_id = 'emd-positionwise'

    experiment = mapel.prepare_experiment(experiment_id=experiment_id,
                                          instance_type=instance_type,
                                          distance_id=distance_id)
    if fixed_positions_path is not None:
        init_pos = _read_fixed_positions(fixed_positions_path)
    else:
        init_pos = None
    start_time = time.time()
    experiment.embed(
        algorithm=algorithm,
        saveas=save_as,
        init_pos=init_pos,
        algorithm_options=algorithm_options,
        random_permute_distances=True
    )
    return time.time() - start_time


def visualize_embedding(experiment_id, coordinate_name):
    instance_type = 'ordinal'
    distance_id = 'emd-positionwise'

    experiment = mapel.prepare_experiment(experiment_id=experiment_id,
                                          instance_type=instance_type,
                                          distance_id=distance_id,
                                          coordinates_names=[coordinate_name])

    experiment.print_map()


def test_on_baseline_dataset():
    ALGORITHMS = {
        'kamada-kawai',
        'mds',
        'isomap',
        'spring'
    }

    RE_RUNS = 10
    FIXED_PATH = 'experiments/baseline-dataset/coordinates/emd-positionwise-paths-big-ID-UN-AN-ST-bb.csv'

    running_times = defaultdict(list)
    for run in range(RE_RUNS):
        for alg in ALGORITHMS:
            save_as = f'{alg}/{alg}_{run}'
            running_time = test_algorithm_on_dataset(
                'baseline-dataset',
                alg,
                save_as,
                fixed_positions_path=FIXED_PATH
            )
            running_times[alg].append(running_time)

    _save_running_times('baseline_running_times.csv', running_times)


def evaluate_results(experiment_id, algorithm_save_folder, algorithm_name):
    instance_type = 'ordinal'
    distance_id = 'emd-positionwise'

    save_root = os.path.join(os.getcwd(), "experiments", experiment_id, 'coordinates', algorithm_save_folder)

    paths = [Path(algorithm_save_folder, p.name) for p in Path(save_root).glob('*.csv')]

    experiment = mapel.prepare_experiment(experiment_id=experiment_id,
                                          instance_type=instance_type,
                                          distance_id=distance_id,
                                          coordinates_names=paths)

    stability = calculate_stability(experiment)

    save_root = os.path.join(os.getcwd(), "experiments", experiment_id,
                             "features", algorithm_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    _save_dict_as_csv(stability, ['name', 'stability'], os.path.join(save_root, 'stability.csv'))

    all_distortion_values = []
    all_monotonicity_values = []

    for path in paths:
        experiment = mapel.prepare_experiment(experiment_id=experiment_id,
                                              instance_type=instance_type,
                                              distance_id=distance_id,
                                              coordinates_names=[path])
        distortion = calculate_distortion(experiment)
        distortion_name = f'{os.path.basename(path)}_distortion.csv'
        _save_dict_as_csv(distortion, ['name', 'distortion'], os.path.join(save_root, distortion_name))
        all_distortion_values.extend(distortion.values())

        monotonicity = calculate_monotonicity(experiment, max_distance_percentage=0.5)
        monotonicity_name = f'{os.path.basename(path)}_monotonicity.csv'
        _save_dict_as_csv(monotonicity, ['name', 'monotonicity'], os.path.join(save_root, monotonicity_name))
        all_monotonicity_values.extend(monotonicity.values())

    avg_stability = np.mean(list(stability.values()))
    avg_distortion = np.mean(all_distortion_values)
    avg_mono = np.mean(all_monotonicity_values)

    return avg_stability, avg_distortion, avg_mono


def evaluate_baseline_dataset():
    ALGORITHMS = {
        'kamada-kawai',
        'mds',
        'isomap',
        'spring'
    }

    results = {}

    for alg in ALGORITHMS:
        save_root = f'experiments/baseline-dataset/coordinates/{alg}'
        avg_stability, avg_distortion, avg_mono = evaluate_results(
            'baseline-dataset',
            alg,
            alg
        )
        results[alg] = [avg_stability, avg_distortion, avg_mono]

    save_path = f'experiments/baseline-dataset/features/results.csv'
    _save_dict_as_csv(results, ['name', 'mean_stability', 'mean_distortion', 'mean_monotonicity'], save_path)


def _read_fixed_positions(path):
    fixed_positions = {}
    with open(path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            fixed_positions[row['instance_id']] = (float(row['x']), float(row['y']))
    return fixed_positions


def _save_running_times(path, running_times):
    avg_running_time = {
        k: np.mean(v) for k, v in running_times.items()
    }

    std_running_time = {
        k: np.std(v) for k, v in running_times.items()
    }

    header = ['algorithm_name', 'mean_time', 'std_time']
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for alg_name in avg_running_time.keys():
            avg, std = avg_running_time[alg_name], std_running_time[alg_name]

            writer.writerow([alg_name, avg, std])


def _save_dict_as_csv(dict_to_save, column_names, path):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(column_names)

        for name, values in dict_to_save.items():
            if not (isinstance(values, list) or isinstance(values, tuple)):
                values = [values]
            writer.writerow([name, *values])


if __name__ == '__main__':
    # test_on_baseline_dataset()
    # visualize_embedding('baseline-dataset', 'kamada-kawai/kamada-kawai_8.csv')
    evaluate_baseline_dataset()
