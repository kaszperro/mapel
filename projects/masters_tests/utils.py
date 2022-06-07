import csv
import os
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import mapel
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
        init_pos = read_fixed_positions(fixed_positions_path)
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


def evaluate_results(experiment_id,
                     algorithm_save_folder,
                     algorithm_name,
                     evaluate=True,
                     plot=False,
                     dataset_name=None,
                     distortion_plot_min=None,
                     distortion_plot_max=None,
                     monotonicity_plot_min=None,
                     monotonicity_plot_max=None):
    instance_type = 'ordinal'
    distance_id = 'emd-positionwise'

    save_root = os.path.join(os.getcwd(), "experiments", experiment_id, 'coordinates', algorithm_save_folder)

    paths = [Path(algorithm_save_folder, p.name) for p in Path(save_root).glob('*.csv')]

    stability = []
    if evaluate:
        experiment = mapel.prepare_experiment(experiment_id=experiment_id,
                                              instance_type=instance_type,
                                              distance_id=distance_id,
                                              coordinates_names=paths)
        stability = calculate_stability(experiment)

        save_root = os.path.join(os.getcwd(), "experiments", experiment_id,
                                 "features", algorithm_name)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        stability_save_path = os.path.join(save_root, 'stability.csv')
        save_dict_as_csv(stability, ['name', 'stability'], stability_save_path)

    all_distortion_values = []
    all_monotonicity_values = []

    for path in paths:
        coordinates_path = Path(
            os.getcwd(), "experiments", experiment_id, 'coordinates', path
        )
        distortion_name = f'{Path(path).stem}_distortion.csv'
        distortion_save_path = os.path.join(save_root, distortion_name)

        monotonicity_name = f'{Path(path).stem}_monotonicity.csv'
        monotonicity_save_path = os.path.join(save_root, monotonicity_name)
        if evaluate:
            experiment = mapel.prepare_experiment(experiment_id=experiment_id,
                                                  instance_type=instance_type,
                                                  distance_id=distance_id,
                                                  coordinates_names=[path])
            distortion = calculate_distortion(experiment)
            save_dict_as_csv(distortion, ['name', 'distortion'], distortion_save_path)
            all_distortion_values.append(list(distortion.values()))

            monotonicity = calculate_monotonicity(experiment, max_distance_percentage=0.8, error_tolerance=0.01)
            save_dict_as_csv(monotonicity, ['name', 'monotonicity'], monotonicity_save_path)
            all_monotonicity_values.append(list(monotonicity.values()))

        if plot:
            visualize_coordinates_with_metric(
                coordinates_path,
                distortion_save_path,
                'distortion',
                f'Distortion - {dataset_name}',
                show=False,
                vmin=distortion_plot_min,
                vmax=distortion_plot_max
            )
            plt.close()

            visualize_coordinates_with_metric(
                coordinates_path,
                monotonicity_save_path,
                'monotonicity',
                f'Monotonicity - {dataset_name}',
                show=False,
                vmin=monotonicity_plot_min,
                vmax=monotonicity_plot_max
            )
            plt.close()
    if evaluate:
        avg_stability = np.mean(list(stability.values()))

        avg_distortion = np.mean(all_distortion_values)
        max_distortion = max(np.mean(all_distortion_values, axis=1))
        min_distortion = min(np.mean(all_distortion_values, axis=1))

        avg_mono = np.mean(all_monotonicity_values)
        max_mono = max(np.mean(all_monotonicity_values, axis=1))
        min_mono = min(np.mean(all_monotonicity_values, axis=1))

        return (
            avg_stability,
            avg_distortion,
            max_distortion,
            min_distortion,
            avg_mono,
            max_mono,
            min_mono
        )


def get_mean_values_from_file(file_path):
    """
    used for distortion and monotonicity files
    :param file_path:
    :return:
    """
    metric_dict = read_metric_file(file_path)

    return np.mean(metric_dict.values())


def visualize_coordinates_with_metric(coordinates_path,
                                      metric_path,
                                      metric_name,
                                      title,
                                      save=True,
                                      show=True,
                                      vmin=None,
                                      vmax=None):
    metric_dict = read_metric_file(metric_path)
    coordinates_dict = read_coordinates_file(coordinates_path)

    xs = []
    ys = []
    zs = []

    for name, (x, y) in coordinates_dict.items():
        z = metric_dict[name]

        xs.append(x)
        ys.append(y)
        zs.append(z)

    if vmin is None:
        vmin = np.min(zs)

    if vmax is None:
        vmax = np.max(zs)

    fig, ax = plt.subplots(figsize=(10, 6))
    pcm = ax.scatter(xs, ys, c=zs, vmin=vmin, vmax=vmax,
                     cmap='jet')
    # pcm = ax.pcolor(xs, ys, zs,
    #                    norm=colors.LogNorm(vmin=np.min(zs), vmax=np.max(zs)),
    #                    cmap='PuBu_r', shading='auto')
    fig.colorbar(pcm, ax=ax)
    plt.title(title)
    ax.set_aspect('equal')
    if show:
        plt.show()
    if save:
        save_root = Path(metric_path).parent.resolve()
        save_path = Path(save_root, f'{Path(metric_path).stem}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)


def save_running_times(path, running_times):
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


def read_fixed_positions(path):
    fixed_positions = {}
    with open(path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            fixed_positions[row['instance_id']] = (float(row['x']), float(row['y']))
    return fixed_positions


def save_dict_as_csv(dict_to_save, column_names, path):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(column_names)

        for name, values in dict_to_save.items():
            if not (isinstance(values, list) or isinstance(values, tuple)):
                values = [values]
            writer.writerow([name, *values])


def read_metric_file(file_path):
    all_values = {}

    available_keys = {
        'monotonicity', 'distortion', 'stability'
    }

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in available_keys:
                if k in row:
                    desired_key = k

            all_values[row['name']] = float(row[desired_key])

    return all_values


def read_coordinates_file(file_path):
    all_values = {}

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            all_values[row['instance_id']] = (
                float(row['x']), float(row['y'])
            )

    return all_values
