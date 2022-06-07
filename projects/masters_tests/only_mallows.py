from collections import defaultdict

from projects.masters_tests.utils import test_algorithm_on_dataset, evaluate_results, save_running_times, \
    save_dict_as_csv, visualize_embedding, get_mean_values_from_file, visualize_coordinates_with_metric


def test_on_only_mallows_dataset():
    ALGORITHMS = [
        ('kamada-kawai', 'kamada-kawai-kk', True, {'optim_method': 'kk', 'epsilon': 0.05}),
        ('kamada-kawai', 'kamada-kawai-bb', True, {'optim_method': 'bb'}),
        ('mds', 'mds', False, {}),
        ('isomap', 'isomap', False, {}),
        ('spring', 'spring', False, {}),
        ('simulated-annealing', 'simulated-annealing', True, {
            'initial_temperature': 900,
            'cooling_temp_factor': 0.75,
            'num_stages': 15,
            'number_of_trials_for_temp': 40,
            'cooling_radius_factor': 0.75,
            'initial_radius': None,
        }),
    ]

    RE_RUNS = 10
    FIXED_PATH = 'experiments/only-mallows/coordinates/emd-positionwise-ID-AN-ST-UN-bb.csv'
    FIXED_PATH_SUFIX = 'fixed-4'
    NOT_FIXED_PATH_SUFIX = 'not-fixed'

    running_times = defaultdict(list)
    for run in range(RE_RUNS):
        for algorithm_name, algorithm_rename_name, can_fix_positions, additional_parameters in ALGORITHMS:

            if can_fix_positions:
                to_test = [
                    (f'{algorithm_rename_name}_{NOT_FIXED_PATH_SUFIX}', False),
                    (f'{algorithm_rename_name}_{FIXED_PATH_SUFIX}', True),
                ]
            else:
                to_test = [
                    (f'{algorithm_rename_name}_{NOT_FIXED_PATH_SUFIX}', False)
                ]

            for alg_save_name, should_fix in to_test:
                save_as = f'{alg_save_name}/{alg_save_name}_{run}'
                running_time = test_algorithm_on_dataset(
                    'only-mallows',
                    algorithm_name,
                    save_as,
                    fixed_positions_path=FIXED_PATH if should_fix else None,
                    **additional_parameters
                )
                running_times[alg_save_name].append(running_time)

    save_running_times('only_mallows_running_times.csv', running_times)


def evaluate_only_mallows_dataset():
    ALGORITHMS = {
        'kamada-kawai-kk_fixed-4',
        'kamada-kawai-kk_not-fixed',
        'kamada-kawai-bb_fixed-4',
        'kamada-kawai-bb_not-fixed',
        'mds_not-fixed',
        'isomap_not-fixed',
        'spring_not-fixed',
        'simulated-annealing_not-fixed',
        'simulated-annealing_fixed-4'
    }

    results = {}

    for alg in ALGORITHMS:
        save_root = f'experiments/only-mallows/coordinates/{alg}'
        results[alg] = evaluate_results(
            'only-mallows',
            alg,
            alg,
            plot=True,
            dataset_name='Mallows dataset',
            distortion_plot_min=1.1,
            distortion_plot_max=1.8,
            monotonicity_plot_min=0.7,
            monotonicity_plot_max=1.0
        )

    save_path = f'experiments/only-mallows/features/results.csv'
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
    # test_on_only_mallows_dataset()
    # visualize_embedding('only-mallows', 'isomap/isomap_4.csv')
    evaluate_only_mallows_dataset()
