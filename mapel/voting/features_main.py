#!/usr/bin/env python

import math

import networkx as nx
import numpy as np
import scipy.special

import mapel.voting.features.cohesive as cohesive
import mapel.voting.features.partylist as partylist
import mapel.voting.features.proportionality_degree as prop_deg
import mapel.voting.features.scores as scores
from mapel.voting.metrics.inner_distances import l2


# MAPPING #
def get_feature(feature_id):
    return {'borda_std': borda_std,
            'highest_borda_score': scores.highest_borda_score,
            'highest_plurality_score': scores.highest_plurality_score,
            'highest_copeland_score': scores.highest_copeland_score,
            'lowest_dodgson_score': scores.lowest_dodgson_score,
            'avg_distortion_from_guardians': avg_distortion_from_guardians,
            'worst_distortion_from_guardians': worst_distortion_from_guardians,
            'graph_diameter': graph_diameter,
            'graph_diameter_log': graph_diameter_log,
            'max_approval_score': max_approval_score,
            'largest_cohesive_group': cohesive.count_largest_cohesiveness_level_l_of_cohesive_group,
            'number_of_cohesive_groups': cohesive.count_number_of_cohesive_groups,
            'number_of_cohesive_groups_brute': cohesive.count_number_of_cohesive_groups_brute,
            'proportionality_degree_av': prop_deg.proportionality_degree_av,
            'proportionality_degree_pav': prop_deg.proportionality_degree_pav,
            'proportionality_degree_cc': prop_deg.proportionality_degree_cc,
            'abstract': abstract,
            'monotonicity_1': monotonicity_1,
            'monotonicity_2': monotonicity_2,
            'partylist': partylist.partylistdistance,
            'num_large_parties': partylist.partylistdistance,
            'distortion_from_all': distortion_from_all,
            'distortion_from_top_100': distortion_from_top_100,
            }.get(feature_id)


def monotonicity_1(experiment, election) -> float:
    e0 = election.election_id
    c0 = np.array(experiment.coordinates[e0])
    distortion = 0
    for i, e1 in enumerate(experiment.elections):
        for j, e2 in enumerate(experiment.elections):
            if i < j and e1 != e0 and e2 != e0:
                original_d1 = experiment.distances[e0][e1]
                original_d2 = experiment.distances[e0][e2]
                original_proportion = original_d1 / original_d2
                embedded_d1 = np.linalg.norm(c0 - experiment.coordinates[e1])
                embedded_d2 = np.linalg.norm(c0 - experiment.coordinates[e2])
                embedded_proportion = embedded_d1 / embedded_d2
                _max = max(original_proportion, embedded_proportion)
                _min = min(original_proportion, embedded_proportion)
                distortion += _max / _min
    return distortion


def monotonicity_2(experiment, election) -> float:
    epsilon = 0.1
    e0 = election.election_id
    c0 = np.array(experiment.coordinates[e0])
    distortion = 0.
    ctr = 0.
    for i, e1 in enumerate(experiment.elections):
        for j, e2 in enumerate(experiment.elections):
            if i < j and e1 != e0 and e2 != e0:
                original_d1 = experiment.distances[e0][e1]
                original_d2 = experiment.distances[e0][e2]
                embedded_d1 = np.linalg.norm(c0 - experiment.coordinates[e1])
                embedded_d2 = np.linalg.norm(c0 - experiment.coordinates[e2])
                if (original_d1 < original_d2 and embedded_d1 > embedded_d2 * (1. + epsilon)) or \
                        (original_d2 < original_d1 and embedded_d2 > embedded_d1 * (1. + epsilon)):
                    distortion += 1.
                ctr += 1.
    distortion /= ctr
    return distortion


def abstract(election) -> float:
    n = election.num_voters
    election.votes_to_approvalwise_vector()
    vector = election.approvalwise_vector
    total_value = 0
    for i in range(election.num_candidates):
        k = vector[i] * n
        x = scipy.special.binom(n, k)
        x = math.log(x)
        total_value += x
    return total_value


def borda_std(election):
    all_scores = np.zeros(election.num_candidates)

    vectors = election.votes_to_positionwise_matrix()

    for i in range(election.num_candidates):
        for j in range(election.num_candidates):
            all_scores[i] += vectors[i][j] * (election.num_candidates - j - 1)

    std = np.std(all_scores)
    return std


def get_effective_num_candidates(election, mode='Borda') -> float:
    """ Compute effective number of candidates """

    c = election.num_candidates
    vectors = election.votes_to_positionwise_matrix()

    if mode == 'Borda':
        all_scores = [sum([vectors[j][i] * (c - i - 1) for i in range(c)]) / (c * (c - 1) / 2)
                      for j in range(c)]
    elif mode == 'Plurality':
        all_scores = [sum([vectors[j][i] for i in range(1)]) for j in range(c)]
    else:
        all_scores = []

    return 1. / sum([x * x for x in all_scores])


########################################################################
def map_diameter(c: int) -> float:
    """ Compute the diameter """
    return 1 / 3 * (c + 1) * (c - 1)


def distortion_from_guardians(experiment, election_id) -> np.ndarray:
    values = np.array([])
    election_id_1 = election_id

    for election_id_2 in experiment.elections:
        if election_id_2 in {'identity_10_100_0', 'uniformity_10_100_0',
                             'antagonism_10_100_0', 'stratification_10_100_0'}:
            if election_id_1 != election_id_2:
                m = experiment.elections[election_id_1].num_candidates
                true_distance = experiment.distances[election_id_1][election_id_2]
                true_distance /= map_diameter(m)
                embedded_distance = l2(experiment.coordinates[election_id_1],
                                       experiment.coordinates[election_id_2])

                embedded_distance /= \
                    l2(experiment.coordinates['identity_10_100_0'],
                       experiment.coordinates['uniformity_10_100_0'])
                ratio = float(true_distance) / float(embedded_distance)
                values = np.append(values, ratio)

    return values


def distortion_from_all(experiment, election_id) -> np.ndarray:
    values = np.array([])
    election_id_1 = election_id

    for election_id_2 in experiment.elections:
        # if election_id_2 in {'identity_10_100_0', 'uniformity_10_100_0',
        #                      'antagonism_10_100_0', 'stratification_10_100_0'}:
        if election_id_1 != election_id_2:
            m = experiment.elections[election_id_1].num_candidates
            true_distance = experiment.distances[election_id_1][election_id_2]
            true_distance /= map_diameter(m)
            embedded_distance = l2(np.array(experiment.coordinates[election_id_1]),
                                   np.array(experiment.coordinates[election_id_2]))

            embedded_distance /= \
                l2(np.array(experiment.coordinates['core_800']),
                   np.array(experiment.coordinates['core_849']))
            try:
                ratio = float(embedded_distance) / float(true_distance)
            except:
                ratio = 1.
            values = np.append(values, ratio)

    return np.mean(abs(1.-values))


def distortion_from_top_100(experiment, election_id) -> np.ndarray:
    values = np.array([])
    election_id_1 = election_id

    euc_dist = {}
    for election_id_2 in experiment.elections:
        if election_id_1 != election_id_2:
            euc_dist[election_id_2] = l2(np.array(experiment.coordinates[election_id_1]),
                                           np.array(experiment.coordinates[election_id_2]))

    all = (sorted(euc_dist.items(), key=lambda item: item[1]))
    top_100 = [x for x,_ in all[0:100]]


    # all = (sorted(experiment.distances[election_id_1].items(), key=lambda item: item[1]))
    # top_100 = [x for x,_ in all[0:100]]

    for election_id_2 in experiment.elections:
        if election_id_1 != election_id_2:
            if election_id_2 in top_100:
                m = experiment.elections[election_id_1].num_candidates
                true_distance = experiment.distances[election_id_1][election_id_2]
                true_distance /= map_diameter(m)
                embedded_distance = l2(np.array(experiment.coordinates[election_id_1]),
                                       np.array(experiment.coordinates[election_id_2]))

                embedded_distance /= \
                    l2(np.array(experiment.coordinates['core_800']),
                       np.array(experiment.coordinates['core_849']))
                try:
                    ratio = float(embedded_distance) / float(true_distance)
                except:
                    ratio = 1.
                values = np.append(values, ratio)

    return np.mean(abs(1.-values))


def avg_distortion_from_guardians(experiment, election_id):
    values = distortion_from_guardians(experiment, election_id)
    return np.mean(values)


def worst_distortion_from_guardians(experiment, election_id):
    values = distortion_from_guardians(experiment, election_id)
    return np.max(values)


def graph_diameter(election):
    try:
        return nx.diameter(election.votes)
    except Exception:
        return 100


def graph_diameter_log(election):
    try:
        return math.log(nx.diameter(election.votes))
    except Exception:
        return math.log(100)
##################################


def max_approval_score(election):
    score = np.zeros([election.num_candidates])
    for vote in election.votes:
        for c in vote:
            score[c] += 1
    return max(score)


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 12.10.2021 #
# # # # # # # # # # # # # # # #