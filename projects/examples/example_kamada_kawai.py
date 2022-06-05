import csv

import mapel
from mapel.main.features.stability import calculate_stability


def import_experiment():
    experiment_id = 'emd-positionwise'
    instance_type = 'ordinal'
    distance_id = 'emd-positionwise'

    experiment = mapel.prepare_experiment(experiment_id=experiment_id,
                                          instance_type=instance_type,
                                          distance_id=distance_id,
                                          coordinates_names=[
                                              'emd-positionwise-paths-big-fixed-4_circle_kamada_bb_2_steps.csv',
                                              'emd-positionwise-paths-big-fixed-4_square_kamada_bb_2_steps.csv'
                                          ])
    init_pos_path = 'experiments/emd-positionwise/coordinates/emd-positionwise-paths-big-ID-UN-AN-ST-bb.csv'
    init_pos = _read_initial_positions(init_pos_path)

    experiment.embed(algorithm='kamada-kawai', init_pos=init_pos)
    experiment.print_map()
    print(calculate_stability(experiment))


def _read_initial_positions(path):
    output_dict = {}
    with open(path, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')

        for row in reader:
            output_dict[row['instance_id']] = (row['x'], row['y'])

    return output_dict


if __name__ == '__main__':
    import_experiment()
