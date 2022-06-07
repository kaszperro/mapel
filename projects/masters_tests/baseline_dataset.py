from collections import defaultdict

from projects.masters_tests.utils import test_algorithm_on_dataset, evaluate_results, save_running_times, \
    save_dict_as_csv, visualize_embedding, get_mean_values_from_file, visualize_coordinates_with_metric


def test_on_baseline_dataset():
    ALGORITHMS = {
        'kamada-kawai': True,
        'mds': False,
        'isomap': False,
        'spring': False,
        # 'simulated-annealing': True,
    }

    RE_RUNS = 10
    FIXED_PATH = 'experiments/baseline-dataset/coordinates/emd-positionwise-paths-big-ID-UN-AN-ST-bb.csv'
    FIXED_PATH_SUFIX = 'fixed-4'
    NOT_FIXED_PATH_SUFIX = 'not-fixed'

    running_times = defaultdict(list)
    for run in range(RE_RUNS):
        for alg, can_fix_positions in ALGORITHMS.items():

            if can_fix_positions:
                to_test = [
                    (f'{alg}_{NOT_FIXED_PATH_SUFIX}', False),
                    (f'{alg}_{FIXED_PATH_SUFIX}', True),
                ]
            else:
                to_test = [
                    (f'{alg}_{NOT_FIXED_PATH_SUFIX}', False)
                ]

            for alg_save_name, should_fix in to_test:
                save_as = f'{alg_save_name}/{alg_save_name}_{run}'
                running_time = test_algorithm_on_dataset(
                    'baseline-dataset',
                    alg,
                    save_as,
                    fixed_positions_path=FIXED_PATH if should_fix else None
                )
                running_times[alg_save_name].append(running_time)

    save_running_times('baseline_running_times.csv', running_times)


def evaluate_baseline_dataset():
    ALGORITHMS = {
        'kamada-kawai_fixed-4',
        'kamada-kawai_not-fixed',
        'mds_not-fixed',
        'isomap_not-fixed',
        'spring_not-fixed',
        # 'simulated-annealing-not-fixed',
        # 'simulated-annealing-fixed-4'
    }

    results = {}

    for alg in ALGORITHMS:
        save_root = f'experiments/baseline-dataset/coordinates/{alg}'
        results[alg] = evaluate_results(
            'baseline-dataset',
            alg,
            alg,
            plot=True,
            dataset_name='Baseline dataset',
            distortion_plot_min=1.1,
            distortion_plot_max=1.8,
            monotonicity_plot_min=0.7,
            monotonicity_plot_max=1.0
        )

    save_path = f'experiments/baseline-dataset/features/results.csv'
    save_dict_as_csv(results, [
        'name',
        'mean_stability',
        'mean_distortion',
        'max_distortion',
        'min_distortion',
        'mean_monotonicity',
        'max_monotonicity',
        'min_monotonicity'
    ], save_path)


if __name__ == '__main__':
    # test_on_baseline_dataset()
    visualize_embedding('only-mallows', 'simulated-annealing_not-fixed/simulated-annealing_not-fixed_0.csv')
    # evaluate_baseline_dataset()
