import numpy as np
import math
from numpy import linalg


####################################################################################################
# Approval Euclidean Election Models
####################################################################################################

def get_range(params):
    if params['p_dist'] == 'beta':
        return np.random.beta(params['a'], params['b'])
    elif params['p_dist'] == 'uniform':
        return np.random.uniform(low=params['a'], high=params['b'])
    else:
        return params['p_dist']


def generate_approval_vcr_votes(num_voters: int = None, num_candidates: int = None,
                                params: dict = None) -> list:
    # v_a = 1.05  # params['v_a']
    # v_b = 10  # params['v_b']
    # c_a = 1.05  # params['c_a']
    # c_b = 10  # params['c_b']

    # max_range = params['max_range']

    dim = params['dim']

    votes = [set() for _ in range(num_voters)]

    # voters = np.random.rand(num_voters, dim)
    # candidates = np.random.rand(num_candidates, dim)

    name = f'{dim}d_{params["space"]}'
    print(name)

    voters = np.array([get_rand(name) for _ in range(num_voters)])
    candidates = np.array([get_rand(name) for _ in range(num_candidates)])

    # v_range = [np.random.beta(v_a, v_b) for _ in range(num_voters)]
    # c_range = [np.random.beta(c_a, c_b) for _ in range(num_candidates)]
    # [np.random.uniform(low=0.05, high=max_range) ...

    v_range = np.array([get_range(params) for _ in range(num_voters)])
    c_range = np.array([get_range(params) for _ in range(num_candidates)])

    for v in range(num_voters):
        for c in range(num_candidates):
            if v_range[v] + c_range[c] >= np.linalg.norm(voters[v] - candidates[c]):
                votes[v].add(c)

    return votes


def generate_approval_euclidean_votes(num_voters: int = None, num_candidates: int = None,
                                      params: dict = None) -> list:
    # 'p' should be lower than 0.5

    alpha = 4
    beta = alpha / (params['p']) - alpha

    dim = params['dim']

    rankings = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates])
    votes = []

    if 'shift' in params:
        shift = np.array([params['shift'] ** 2 for _ in range(dim)])
        voters = np.random.rand(num_voters, dim) + shift
        candidates = np.random.rand(num_candidates, dim)
    elif 'gauss' in params:
        voters = np.random.rand(num_voters, dim)
        params['gauss'] /= 2
        num_candidates_in_group_a = int(params['gauss'] * num_candidates)
        num_candidates_in_group_b = num_voters - num_candidates_in_group_a
        scale_group_a = params['gauss']
        scale_group_b = 1 - params['gauss']
        loc_a = [1. / 3 for _ in range(dim)]
        loc_b = [2. / 3 for _ in range(dim)]
        candidates_group_a = np.random.normal(loc=loc_a, scale=scale_group_a,
                                              size=(num_candidates_in_group_a, dim))
        candidates_group_b = np.random.normal(loc=loc_b, scale=scale_group_b,
                                              size=(num_candidates_in_group_b, dim))
        candidates = np.concatenate((candidates_group_a, candidates_group_b), axis=0)
    elif 'model_id' in params:
        model = params['model_id']
        voters = np.array([get_rand(model) for _ in range(num_voters)])
        candidates = np.array([get_rand(model) for _ in range(num_candidates)])
    else:
        voters = np.random.rand(num_voters, dim)
        candidates = np.random.rand(num_candidates, dim)

    for v in range(num_voters):
        for c in range(num_candidates):
            rankings[v][c] = c
            distances[v][c] = np.linalg.norm(voters[v] - candidates[c])
        rankings[v] = [x for _, x in sorted(zip(distances[v], rankings[v]))]

    for v in range(num_voters):
        k = int(np.random.beta(alpha, beta) * num_candidates)
        votes.append(set(rankings[v][0:k]))

    return votes


####################################################################################################
# Party Euclidean Election Models
####################################################################################################
def generate_1d_gaussian_party(num_voters=None, num_candidates=None, params=None):
    if params is None:
        params = {}
    if 'num_winners' not in params:
        params['num_winners'] = 1

    voters = [[] for _ in range(num_voters)]
    candidates = [[] for _ in range(num_candidates)]

    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    for j in range(params['num_parties']):
        for w in range(params['num_winners']):
            _id = j * params['num_winners'] + w
            candidates[_id] = [np.random.normal(params['party'][j][0], params['var'])]

    _min = min(candidates)[0]
    _max = max(candidates)[0]

    shift = [np.random.random() / 2. - 1 / 4.]
    for j in range(num_voters):
        voters[j] = [np.random.random() * (_max - _min) + _min + shift[0]]

    for j in range(num_voters):
        for k in range(num_candidates):
            votes[j][k] = k
            distances[j][k] = np.linalg.norm(voters[j] - candidates[k])

        votes[j] = [x for _, x in sorted(zip(distances[j], votes[j]))]

    return votes


def generate_2d_gaussian_party(num_voters=None, num_candidates=None, params=None):
    if params is None:
        params = {}
    if 'num_winners' not in params:
        params['num_winners'] = 1

    voters = [[] for _ in range(num_voters)]
    candidates = [[] for _ in range(num_candidates)]

    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    for j in range(params['num_parties']):
        for w in range(params['num_winners']):
            _id = j * params['num_winners'] + w
            candidates[_id] = [np.random.normal(params['party'][j][0][0], params['var']),
                               np.random.normal(params['party'][j][0][1], params['var'])]

    def column(matrix, i):
        return [row[i] for row in matrix]

    x_min = min(column(candidates, 0))
    x_max = max(column(candidates, 0))
    y_min = min(column(candidates, 1))
    y_max = max(column(candidates, 1))

    shift = [np.random.random() / 2. - 1 / 4., np.random.random() / 2. - 1 / 4.]
    for j in range(num_voters):
        voters[j] = [np.random.random() * (x_max - x_min) + x_min + shift[0],
                     np.random.random() * (y_max - y_min) + y_min + shift[1]]

    for j in range(num_voters):
        for k in range(num_candidates):
            votes[j][k] = k
            distances[j][k] = np.linalg.norm(voters[j] - candidates[k])

        votes[j] = [x for _, x in sorted(zip(distances[j], votes[j]))]

    return votes


####################################################################################################
# Ordinal Euclidean Election Models
####################################################################################################
def generate_ordinal_euclidean_votes(model: str = None, num_voters: int = None,
                                     num_candidates: int = None,
                                     params: dict = None) -> np.ndarray:
    if 'dim' in params:
        dim = params['dim']
    else:
        dim = 2

    voters = np.zeros([num_voters, params['dim']])
    candidates = np.zeros([num_candidates, params['dim']])
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    if model == 'euclidean':
        if params['space'] == 'uniform':
            voters = np.random.rand(num_voters, dim)
            candidates = np.random.rand(num_candidates, dim)
        elif params['space'] == 'gaussian':
            voters = np.random.normal(loc=0.5, scale=0.15, size=(num_voters, dim))
            candidates = np.random.normal(loc=0.5, scale=0.15, size=(num_candidates, dim))
        elif params['space'] == 'sphere':
            voters = np.array([list(random_sphere(dim)[0]) for _ in range(num_voters)])
            candidates = np.array([list(random_sphere(dim)[0]) for _ in range(num_candidates)])
    else:
        for v in range(num_voters):
            voters[v] = get_rand(model)
        # voters = sorted(voters)

        for v in range(num_candidates):
            candidates[v] = get_rand(model)
        # candidates = sorted(candidates)

    for v in range(num_voters):
        for c in range(num_candidates):
            votes[v][c] = c
            distances[v][c] = np.linalg.norm(voters[v] - candidates[c])

        votes[v] = [x for _, x in sorted(zip(distances[v], votes[v]))]

    return votes


def generate_elections_2d_grid(num_voters=None, num_candidates=None):
    voters = np.zeros([num_voters, 2])
    candidates = []

    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    for j in range(num_voters):
        voters[j] = get_rand('2d_square')
    voters = sorted(voters)

    sq = int(num_candidates ** 0.5)
    d = 1. / sq

    for i in range(sq):
        for j in range(sq):
            x = d / 2. + d * i
            y = d / 2. + d * j
            point = [x, y]
            candidates.append(point)

    for j in range(num_voters):
        for k in range(num_candidates):
            votes[j][k] = k
            distances[j][k] = np.linalg.norm(voters[j] - candidates[k])

        votes[j] = [x for _, x in sorted(zip(distances[j], votes[j]))]

    return votes


# AUXILIARY
def random_ball(dimension, num_points=1, radius=1):
    random_directions = np.random.normal(size=(dimension, num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    random_radii = np.random.random(num_points) ** (1 / dimension)
    x = radius * (random_directions * random_radii).T
    return x


def random_sphere(dimension, num_points=1, radius=1):
    random_directions = np.random.normal(size=(dimension, num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    random_radii = 1.
    return radius * (random_directions * random_radii).T


def get_rand(model: str, cat: str = "voters") -> list:
    """ generate random values"""
    # print(model ==  "1d_uniform")

    point = [0]
    if model in {"1d_uniform",  "1d_interval"}:
        return np.random.rand()
    elif model in {'1d_asymmetric'}:
        if np.random.rand() < 0.3:
            return np.random.normal(loc=0.25, scale=0.15, size=1)
        else:
            return np.random.normal(loc=0.75, scale=0.15, size=1)
    elif model in {"1d_gaussian"}:
        point = np.random.normal(0.5, 0.15)
        while point > 1 or point < 0:
            point = np.random.normal(0.5, 0.15)
    elif model == "1d_one_sided_triangle":
        point = np.random.uniform(0, 1) ** 0.5
    elif model == "1d_full_triangle":
        point = np.random.choice([np.random.uniform(0, 1) ** 0.5, 2 - np.random.uniform(0, 1) ** 0.5])
    elif model == "1d_two_party":
        point = np.random.choice([np.random.uniform(0, 1), np.random.uniform(2, 3)])
    elif model in {"2d_disc", "2d_range_disc"}:
        phi = 2.0 * 180.0 * np.random.random()
        radius = math.sqrt(np.random.random()) * 0.5
        point = [0.5 + radius * math.cos(phi), 0.5 + radius * math.sin(phi)]
    elif model == "2d_range_overlapping":
        phi = 2.0 * 180.0 * np.random.random()
        radius = math.sqrt(np.random.random()) * 0.5
        if cat == "voters":
            point = [0.25 + radius * math.cos(phi), 0.5 + radius * math.sin(phi)]
        elif cat == "candidates":
            point = [0.75 + radius * math.cos(phi), 0.5 + radius * math.sin(phi)]
    elif model in {"2d_square", "2d_uniform"}:
        point = [np.random.random(), np.random.random()]
    elif model in {'2d_asymmetric'}:
        if np.random.rand() < 0.3:
            return np.random.normal(loc=0.25, scale=0.15, size=2)
        else:
            return np.random.normal(loc=0.75, scale=0.15, size=2)
    elif model == "2d_sphere":
        alpha = 2 * math.pi * np.random.random()
        x = 1. * math.cos(alpha)
        y = 1. * math.sin(alpha)
        point = [x, y]
    elif model in ["2d_gaussian", "2d_range_gaussian"]:
        point = [np.random.normal(0.5, 0.15), np.random.normal(0.5, 0.15)]
        while np.linalg.norm(point - np.array([0.5, 0.5])) > 0.5:
            point = [np.random.normal(0.5, 0.15), np.random.normal(0.5, 0.15)]
    elif model in ["2d_range_fourgau"]:
        r = np.random.randint(1, 4)
        size = 0.06
        if r == 1:
            point = [np.random.normal(0.25, size), np.random.normal(0.5, size)]
        if r == 2:
            point = [np.random.normal(0.5, size), np.random.normal(0.75, size)]
        if r == 3:
            point = [np.random.normal(0.75, size), np.random.normal(0.5, size)]
        if r == 4:
            point = [np.random.normal(0.5, size), np.random.normal(0.25, size)]
    elif model in ["3d_cube", "3d_uniform"]:
        point = [np.random.random(), np.random.random(), np.random.random()]
    elif model in {'3d_asymmetric'}:
        if np.random.rand() < 0.3:
            return np.random.normal(loc=0.25, scale=0.15, size=3)
        else:
            return np.random.normal(loc=0.75, scale=0.15, size=3)
    elif model == "4d_cube":
        dim = 4
        point = [np.random.random() for _ in range(dim)]
    elif model == "5d_cube":
        dim = 5
        point = [np.random.random() for _ in range(dim)]
    elif model == "10d_cube":
        dim = 10
        point = [np.random.random() for _ in range(dim)]
    elif model == "20d_cube":
        dim = 20
        point = [np.random.random() for _ in range(dim)]
    elif model == "3d_sphere":
        dim = 3
        point = list(random_sphere(dim)[0])
    elif model == "4d_sphere":
        dim = 4
        point = list(random_sphere(dim)[0])
    elif model == "5d_sphere":
        dim = 5
        point = list(random_sphere(dim)[0])
    else:
        print('unknown model_id', model)
        point = [0, 0]
    return point
