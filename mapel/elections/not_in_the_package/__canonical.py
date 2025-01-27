#!/usr/bin/env python

import copy
import csv
import math
import os
import random as rand
import time
from collections import Counter
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
# from skopt import gp_minimize
from scipy.stats import stats

from mapel.voting import elections as el
from mapel.voting import metrics as metr
from mapel.voting import objects as obj
from mapel.voting.other import winners2 as win


# def separation(election) -> float:
#
#     if election.fake:
#         return 0
#
#     half = int(election.num_candidates / 2)
#
#     ranking = dev.get_borda_ranking(election.votes, election.num_voters,
#                                     election.num_candidates)
#     first_half = ranking[0:half]
#
#     distance = 0
#
#     for i in range(election.num_voters):
#         for j in range(half):
#             if election.votes[i][j] not in first_half:
#                 distance += half - j
#
#     for i in range(election.num_voters):
#         for j in range(half, election.num_candidates):
#             if election.votes[i][j] in first_half:
#                 distance += j - half
#
#     return distance
#
#
# def both(election) -> float:
#     v1 = borda_std(election) / 2.9
#     v2 = separation(election) / 1235.
#     return v1 + v2

def prepare_approx_cc_order(experiment_id, metric="positionwise"):
    """ Copy all the elections and the change the order according to approx_cc order """

    file_name = os.path.join(os.getcwd(), "experiments", str(experiment_id),
                             "controllers", "orders",
                             str(metric) + "_approx_cc.txt")
    file_ = open(file_name, 'r')

    file_.readline()  # skip this line
    num_elections = int(file_.readline())
    file_.readline()  # skip this line

    for i in range(num_elections):
        target = str(file_.readline().replace("\n", ""))

        src = os.path.join(os.getcwd(), "experiments", str(experiment_id),
                           "elections", "soc_original",
                           "core_" + str(target) + ".soc")

        dst = os.path.join(os.getcwd(), "experiments", str(experiment_id),
                           "elections",
                           "soc_" + str(metric) + "_approx_cc",
                           "core_" + str(i) + ".soc")

        copyfile(src, dst)

### VERIONS WITH DICT INSTEAD OF LIST
def compute_plurality_winners(election=None, num_winners=1):

    scores = {}
    for c in election.votes[0]:
        scores[c] = 0

    for vote in election.votes:
        scores[vote[0]] += 1

    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ranking = ranking[0:num_winners]
    winners = []

    for winner in ranking:
        winners.append(winner[0])

    return winners


def compute_borda_winners(election=None, num_winners=1):

    scores = {}
    for c in election.votes[0]:
        scores[c] = 0

    for vote in election.votes:
        for pos, c in enumerate(vote):
            scores[c] += election.num_candidates - pos - 1

    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ranking = ranking[0:num_winners]
    winners = []

    for winner in ranking:
        winners.append(winner[0])

    return winners


def compute_stv_winners(election=None, num_winners=1):

    winners = []  # [0] * params['orders']

    droop_quota = math.floor(
        election.num_voters / (num_winners + 1.)) + 1

    votes_on_1 = {}
    active = {}
    for c in election.votes[0]:
        votes_on_1[c] = 0
        active[c] = True

    for vote in election.votes:
        votes_on_1[vote[0]] += 1

    v_power = [1.] * election.num_voters

    keys = list(votes_on_1.keys())

    while len(winners) + sum(active.values()) > num_winners:

        ctr = election.num_candidates
        iterator = 0
        winner_id = keys[iterator]
        while ctr > 0:

            if active[winner_id] and votes_on_1[winner_id] >= droop_quota:

                winners += [winner_id]

                total = 0
                for i in range(election.num_voters):
                    for j in range(election.num_candidates):
                        if active[election.votes[i][j]]:
                            if election.votes[i][j] == winner_id:
                                for k in range(j + 1,
                                               election.num_candidates):
                                    if active[election.votes[i][k]]:
                                        v_power[i] *= float(votes_on_1[
                                                                winner_id] - droop_quota) / float(
                                            votes_on_1[winner_id])
                                        votes_on_1[election.votes[i][k]] += 1. * \
                                                                   v_power[
                                                                       i]
                                        total += 1. * v_power[i]
                                        ctr = election.num_candidates
                                        break
                            break

                votes_on_1[winner_id] = 0
                active[winner_id] = False

            ctr -= 1
            iterator += 1
            iterator %= election.num_candidates
            winner_id = keys[iterator]

        loser_votes = droop_quota
        loser_id = keys[0]
        for key in keys:
            if active[key] and votes_on_1[key] < loser_votes:
                loser_votes = votes_on_1[key]
                loser_id = key

        votes_on_1[loser_id] = 0
        for i in range(election.num_voters):
            for j in range(election.num_candidates):
                if active[election.votes[i][j]]:
                    if election.votes[i][j] == loser_id:
                        for k in range(j + 1, election.num_candidates):
                            if active[election.votes[i][k]]:
                                votes_on_1[election.votes[i][k]] += 1. * v_power[i]
                                break
                    break
        active[loser_id] = False

    for key in keys:
        if active[key]:
            winners += [key]

    winners = sorted(winners)

    return winners[0:num_winners]
#########################


def compute_approx(experiment_id, method='hb', algorithm='greedy', num_winners=10):

    model = obj.Experiment(experiment_id)

    file_name = "experiments/" + experiment_id + "/controllers/approx/" + method + "_" + algorithm + ".txt"
    file_output = open(file_name, 'w')
    num_lines = model.num_elections
    file_output.write(str(num_lines) + "\n")

    Z = 0
    for fam in range(model.num_families):

        for _ in range(model.families[fam].size):

            print(Z)
            election_id = "core_" + str(Z)
            # # exact
            # file_name = "experiments/" + experiment_id + "/controllers/orders/" + election_id + "_" + method + ".txt"
            # file_controllers = open(file_name, 'r')
            # num_electitons = int(file_controllers.readline())
            # num_winners = int(file_controllers.readline())
            # time = float(file_controllers.readline())
            # winners = []
            # for i in range(num_winners):
            #     winner = file_controllers.readline()
            #     winners.append(int(winner))

            params = {}
            params['orders'] = num_winners
            params['voters'] = model.families[fam].num_voters
            params['candidates'] = model.families[fam].num_candidates
            params['elections'] = model.num_elections


            # print(orders)
            if method == "pav":
                winners_1 = win.get_winners(params, copy.deepcopy(model.elections[Z].votes), method)
            elif method == "hb":
                rule = {}
                rule['election_id'] = 'hb'
                rule['length'] = num_winners
                rule['type'] = 'borda_owa'
                winners_1 = win.get_winners(params, copy.deepcopy(model.elections[Z].votes), rule)


            # print(orders)
            if method == "pav":
                score_1 = win.check_pav_score(copy.deepcopy(model.elections[Z].votes), params, winners_1)
            elif method == "hb":
                score_1 = win.check_hb_score(copy.deepcopy(model.elections[Z].votes), params, winners_1)

            # approx
            # """
            if method == "pav":
                winners_2 = win.get_winners_approx_pav(copy.deepcopy(model.elections[Z].votes), params, algorithm)
            elif method == "hb":
                winners_2 = win.get_winners_approx_hb(copy.deepcopy(model.elections[Z].votes), params, algorithm)

            # print(orders)
            if method == "pav":
                score_2 = win.check_pav_score(copy.deepcopy(model.elections[Z].votes), params, winners_2)
            elif method == "hb":
                score_2 = win.check_hb_score(copy.deepcopy(model.elections[Z].votes), params, winners_2)

            print(score_1, score_2)
            output = score_2 / score_1
            print(output)

            file_output.write(str(output) + "\n")

            Z = Z+1

    file_output.close()

    print("\nDone.")


def merge_segments(experiment_id, num_segments):

    file_name = "experiments/" + experiment_id + "/results/distances/" + str(experiment_id) + "_info.txt"
    file_input = open(file_name, 'r')

    file_name = "experiments/" + experiment_id + "/results/distances/" + str(experiment_id) + ".txt"
    file_output = open(file_name, 'w')

    file_output.write(file_input.readline())  # first line
    file_output.write(file_input.readline())  # second line
    file_output.write(file_input.readline())  # third line
    file_input.close()

    for segment in range(num_segments):

        file_name = "experiments/" + experiment_id + "/results/distances/" + str(experiment_id) + "_" + str(segment) + ".txt"
        file_input = open(file_name, 'r')
        num_distances = int(file_input.readline())

        for i in range(num_distances):
            file_output.write(file_input.readline())

        file_input.close()


def prepare_elections_extended(experiment_id):

    model = obj.Experiment(experiment_id)

    id_ = 800
    for i in range(30, model.num_families):
        elections_type = model.families[i].model_id
        special_1 = model.families[i].special_1
        special_2 = model.families[i].special_2
        num_elections = 1

        # list of IDs larger than 10
        if elections_type == 'dublin':
            folder = 'dublin_s1'
            ids = [1, 3]
        elif elections_type == 'glasgow':
            folder = 'glasgow_s1'
            ids = [2,3,4,5,6,7,8,9,11,13,16,19,21]
        elif elections_type == 'formula':
            elections_type = 'formula'
            folder = 'formula_s1'
            ids = [i for i in range(48)]
        elif elections_type == 'skate':
            folder = 'skate_ic'
            ids = [i for i in range(48)]
        elif elections_type == 'sushi':
            folder = 'sushi_ff'
            ids = [1]

        rand_ids = rand.choices(ids, k=model.families[i].size)
        for ri in rand_ids:
            elections_id = "core_" + str(id_)
            tmp_elections_type = elections_type + '_' + str(ri)
            print(tmp_elections_type)
            el.generate_elections_preflib(experiment_id, tmp_elections_type, elections_id, num_elections,
                                  model.num_voters, model.num_candidates, special_1, folder=folder)
            id_ += 1


def prepare_elections_unid(experiment_id):

    model = obj.Experiment(experiment_id)

    id_ = 900
    #for i in range(35, model_id.num_families):
    #id_ = 0
    for i in range(model.num_families):
        elections_type = model.families[i].model_id
        special_1 = model.families[i].special_1
        special_2 = model.families[i].special_2
        num_elections = 1

        print(model.families[0].size)

        for j in range(model.families[i].size):
            special_1 = j
            elections_id = "core_" + str(id_)
            elections_type = 'unid'
            el.generate_elections(experiment_id, elections_type, elections_id, num_elections,
                                  model.num_voters, model.num_candidates, special_1)
            id_ += 1


def compute_distances_between_elections_by_segments(segment, num_segments, experiment_id,
                                                    metric_type="emd", distance_name="positionwise"):

    num_elections = 1
    special = 0

    file_controllers = open("experiments/" + experiment_id + "/controllers/basic/map.txt", 'r')
    num_voters = int(file_controllers.readline())
    num_candidates = int(file_controllers.readline())
    num_families = int(file_controllers.readline())
    family_name = [0 for _ in range(num_families)]
    family_special = [0 for _ in range(num_families)]
    family_size = [0 for _ in range(num_families)]

    for i in range(num_families):
        line = file_controllers.readline().replace(" ", "").rstrip("\n").split(',')

        family_name[i] = line[1]
        if family_name[i] in {"didi", "pl"}:
            family_special[i] = str(line[2])
        else:
            family_special[i] = float(line[2])
        family_size[i] = int(line[0])

    file_controllers.close()

    number_of_elections = sum(family_size)


    #2 COMPUTE DISTANCES

    print("START", segment)

    results = []

    total = number_of_elections*(number_of_elections-1)/2

    ctr = -1

    for i in range(number_of_elections):
        #print(i)

        for j in range(i + 1, number_of_elections):
            ctr += 1

            if not (segment * total / float(num_segments) <= ctr < (segment + 1) * total / float(num_segments)):
                continue

            elections_id_a = "core_" + str(i)
            elections_id_b = "core_" + str(j)
            elections_ids = [elections_id_a, elections_id_b]

            result = metr.get_distance(experiment_id, distance_name, elections_ids, metric_type)
            print(i,j,result[0])
            results.append(result[0])

    #print(results)


    """
    results2 = []

    for i in range(number_of_elections):
        print(i)
        for j in range(i + 1, number_of_elections):
            elections_id_a = "core_" + str(i)
            elections_id_b = "core_" + str(j)
            elections_ids = [elections_id_a, elections_id_b]

            distance_id = "matrix_metric"
            distance = metr.get_distance(distance_id, elections_ids)

            results2.append(distance[0])
    """

    #3 SAVE RESULTS TO FILE

    if segment == 0:

        file_name = "experiments/" + experiment_id + "/results/distances/" + str(experiment_id) + "_info.txt"
        file_ = open(file_name, 'w')
        file_.write(str(number_of_elections) + "\n")
        file_.write(str(num_families) + "\n")
        file_.write(str(total) + "\n")
        file_.close()

    ctr = -1
    real_ctr = 0
    file_name = "experiments/" + experiment_id + "/results/distances/" + str(experiment_id) + "_" + str(segment) +".txt"
    file_ = open(file_name, 'w')
    file_.write(str(len(results)) + "\n")

    for i in range(number_of_elections):

        for j in range(i + 1, number_of_elections):
            ctr += 1

            if not (segment * total / float(num_segments) <= ctr < (segment + 1) * total / float(num_segments)):
                continue

            file_.write(str(i) + ' ' + str(j) + ' ' + str(results[real_ctr]) + "\n")
            real_ctr += 1

    file_.close()

    #4 ANALYSIS

    """
    print(stats.pearsonr(results, results2))
    plt.scatter(results, results2)
    plt.show()
    """


def compute_lacknerwise_distances_between_elections_by_segments(segment, num_segments, experiment_id):
    compute_distances_between_elections_by_segments(segment, num_segments, experiment_id,
                                                                distance_name="lacknerwise", metric_type="l2")



def generate_votes_from_distances(experiment_id, num_points=800):

    tmp = "positionwise"

    file_name = "experiments/" + experiment_id + "/controllers/distances/" + str(tmp) + ".txt"
    file_ = open(file_name, 'r')

    real_num_points = int(file_.readline())
    num_families = int(file_.readline())
    num_distances = int(file_.readline())

    votes = [[k for k in range(num_points)] for _ in range(num_points)]
    distances = [[0. for _ in range(num_points)] for _ in range(num_points)]

    for i in range(real_num_points):
        for j in range(i + 1, real_num_points):
            line = file_.readline().split(' ')
            if i < num_points and j < num_points:
                distances[i][j] = float(line[2])
                if distances[i][j] == 0.:
                    distances[i][j] == 0.01
                distances[j][i] = distances[i][j]

    for j in range(num_points):
        votes[j] = [x for _, x in sorted(zip(distances[j], votes[j]))]

    file_name = "experiments/" + experiment_id + "/elections/votes/" + str(experiment_id) + ".txt"
    file_votes = open(file_name, 'w')

    randomized_order = [x for x in range(num_points)]
    #rand.shuffle(randomized_order)

    for j in range(num_points):
        for k in range(num_points):
            r = randomized_order[j]
            file_votes.write(str(votes[r][k]) + "\n")

    file_votes.close()

    file_name = "experiments/" + experiment_id + "/elections/info/" + str(experiment_id) + ".txt"
    file_info = open(file_name, 'w')
    file_info.write(str(0) + "\n")
    file_info.write(str(1) + "\n")
    file_info.write(str(num_points) + "\n")
    file_info.write(str(num_points) + "\n")
    file_info.close()


def compute_canonical_winners(experiment_id, method="approx_cc", num_winners=800):

    generate_votes_from_distances(experiment_id)

    rule = {'type': method,
            'election_id': 0,
            'length': 0,
            'special': 0,
            'pure': False}

    winners = win.generate_winners(experiment_id, num_winners, rule, method, experiment_id)

    return winners


def compute_time(experiment_id):

    rule = {'type': 'borda_owa',
             'election_id': 0,
             'length': 0,
             'special': 0,
             'pure': False}

    #file_read = open("abbey/hist_data_md/cc.txt", 'r')
    #spam_line_1 = int(file_read.readline())
    #num_winners = int(file_read.readline())

    time_table = []

    #for w in range(num_winners):
        #id = int(file_read.readline())

    method = "hb"
    num_winners = 10

    #for w in {680, 681, 682, 740, 741, 742}:
    for w in range(100):

        elections_id = "core_" + str(w)
        print(elections_id)

        start_time = time.time()

        win.generate_winners(experiment_id, num_winners, rule, method, elections_id)

        elapsed_time = time.time() - start_time

        print(w, elapsed_time)
        time_table.append(elapsed_time)

    file_write = open("experiments/" + experiment_id + "/controllers/times/" + str(method) + ".txt", 'w')
    file_write.write(str(len(time_table)) + "\n")

    for i in range(len(time_table)):
        file_write.write(str(time_table[i]) + "\n")
    file_write.close()


def save_to_soc(experiment_id, winners):

    for idx, winner in enumerate(winners):

        elections_id = "core_" + str(int(winner))

        elections, params = el.import_elections(experiment_id, elections_id)

        num_elections = 1
        num_candidates = params['candidates']
        num_voters = params['voters']

        file_name = "experiments/" + experiment_id + "/preflib/soc/" + str(experiment_id) + '_' + str(idx) + ".soc"

        file_ = open(file_name, 'w')

        file_.write(str(num_candidates) + "\n")

        for i in range(num_candidates):
            file_.write(str(i+1) + ', ' + chr(97+i) + "\n")

        c = Counter(map(tuple, elections['votes'][0]))
        counted_votes = [[count, list(row)] for row, count in c.items()]
        counted_votes = sorted(counted_votes, reverse=True)

        file_.write(str(num_voters) + ', ' + str(num_voters) + ', ' + str(len(counted_votes)) + "\n")

        for i in range(len(counted_votes)):

            file_.write(str(counted_votes[i][0]) + ', ')

            for j in range(num_candidates):
                file_.write(str(counted_votes[i][1][j]))
                if j < num_candidates - 1:
                    file_.write(", ")
                else:
                    file_.write("\n")

        file_.close()


def compute_canonical_order(experiment_id, method, num_winners):

    """
    rule = {'type': method,
            'election_id': 0,
            'length': num_winners,
            'special': 0,
            'pure': False}

    orders = win.generate_winners(experiment_id, num_winners, rule, method, experiment_id)
    win.generate_winners(experiment_id, num_winners, rule, method, elections_id)

    return orders
    """
    return 0




def get_num_candidates(experiment_id, short_id, folder):

    path = os.path.join(os.getcwd(), 'real_data', folder, short_id + '.txt')
    with open(path, 'r') as txtfile:
        num_voter = int(txtfile.readline().strip())
        num_candidates = int(txtfile.readline().strip())

    return num_candidates


def get_num_voters(experiment_id, short_id, folder):

    path = os.path.join(os.getcwd(), 'real_data', folder, short_id + '.txt')
    with open(path, 'r') as txtfile:
        num_voter = int(txtfile.readline().strip())
        num_candidates = int(txtfile.readline().strip())

    return num_voter





#
# def rearrange_time_format(experiment_id):
#
#     experiment_id = "final"
#     type = "hb"
#
#     num_elections = 860
#     time_table = []
#
#     for i in range(num_elections):
#         file_name = "experiments/" + experiment_id + "/controllers/orders" + \
#                     "/core_" + str(i) + "_" + str(type) + ".txt"
#         file_ = open(file_name, 'r')
#         spam_line_1 = int(file_.readline())
#         spam_line_2 = int(file_.readline())
#         time = float(file_.readline())
#         time_table.append(time)
#         file_.close()
#
#     file_name = "experiments/" + experiment_id + "/controllers/times/" + experiment_id + "_" + str(type) + ".txt"
#     file_ = open(file_name, 'w')
#     file_.write(str(num_elections) + "\n")
#
#     for i in range(num_elections):
#         file_.write(str(time_table[i]) + "\n")
#
#     file_.close()
#




# chyba stare
def print_matrix(experiment_id, scale=10.):
    # IMPORT DISTANCES

    #experiment_id = "example_100_3"

    ######
    file_controllers = open("experiments/" + experiment_id + "/controllers/basic/map.txt", 'r')
    num_families = int(file_controllers.readline())
    num_families = int(file_controllers.readline())
    num_families = int(file_controllers.readline())
    family_name = [0 for _ in range(num_families)]
    family_special = [0 for _ in range(num_families)]
    family_size = [0 for _ in range(num_families)]
    family_full_name = [0 for _ in range(num_families)]

    labels = [0 for _ in range(num_families)]
    colors = [0 for _ in range(num_families)]
    alphas = [0 for _ in range(num_families)]

    for i in range(num_families):
        line = file_controllers.readline().rstrip("\n").split(',')

        family_name[i] = str(line[1])
        family_special[i] = float(line[2])
        family_size[i] = int(line[0])
        family_full_name[i] = str(line[5])

        labels[i] = str(line[1])
        colors[i] = str(line[3])
        alphas[i] = float(line[4])
    ######

    file_name = "experiments/" + str(experiment_id) + "/controllers/distances/" + experiment_id + ".txt"
    file_ = open(file_name, 'r')

    num_elections = int(file_.readline())
    second_line = int(file_.readline())
    third_line = int(file_.readline())

    matrix = [[0. for _ in range(second_line)] for _ in range(second_line)]
    quan = [[0 for _ in range(second_line)] for _ in range(second_line)]

    code = [0 for _ in range(num_elections)]

    ctr = 0
    for i in range(num_families):
        for j in range(family_size[i]):
            code[ctr] = i
            ctr += 1

    #print(code)

    for i in range(third_line):
        line = file_.readline().split(' ')
        # print(code[int(line[0])])
        a = code[int(line[0])]
        b = code[int(line[1])]
        value = float(line[2])
        matrix[a][b] += value
        quan[a][b] += 1

    ######################
    # ODPOWIEDNIO PRZESKALOWAC WYNIKI !!!

    for i in range(second_line):
        for j in range(i, second_line):
            matrix[i][j] /= float(quan[i][j])
            matrix[i][j] *= scale
            matrix[i][j] = int(round(matrix[i][j], 0))
            matrix[j][i] = matrix[i][j]

    #print(matrix[0][0])

    print("ok")
    file_.close()



    order = [i for i in range(num_families)]
    file_name = "experiments/" + str(experiment_id) + "/controllers/matrix.txt"
    file_ = open(file_name, 'r')

    num_families_order = int(file_.readline())

    for i in range(num_families_order):
        line = str(file_.readline().replace("\n","").replace(" ", ""))
        for j in range(num_families):
            #print(line, family_full_name[j])
            if family_full_name[j].replace(" ","") == line:
                order[i] = j


    fig, ax = plt.subplots()

    matrix_order = [[0. for _ in range(num_families_order)] for _ in range(num_families_order)]

    for i in range(num_families_order):
        for j in range(num_families_order):
            c = int(matrix[order[i]][order[j]])
            matrix_order[i][j] = c
            ax.text(i, j, str(c), va='center', ha='center')

    family_full_name_order = []
    for i in range(num_families_order):
        family_full_name_order.append(family_full_name[order[i]])


    min_val, max_val = 0, num_families_order

    ax.matshow(matrix_order, cmap=plt.cm.Blues)

    x_values = family_full_name_order
    y_values = family_full_name_order
    y_axis = np.arange(0, num_families_order, 1)
    x_axis = np.arange(0, num_families_order, 1)

    # plt.barh(y_axis, x_values, align='center')
    plt.yticks(y_axis, y_values)
    plt.xticks(x_axis, x_values, rotation='vertical')

    plt.savefig("matrix_luty.png")
    plt.show()




def compute_canonical_order_op():


    experiment_id = "hidalgo"
    num_winners = 20

    start_time = time.time()

    print("A")
    generate_votes_from_distances(experiment_id, 200)

    print("B")

    rule = {'type': "borda_owa",
            'election_id': 0,
            'length': 0,
            'special': 0,
            'pure': False}

    utopia_type = 'cc'

    winners = win.generate_winners(experiment_id, num_winners, rule, utopia_type, experiment_id)

    print("C")
    #can.save_to_soc(experiment_id, orders)

    elapsed_time = time.time() - start_time
    print("time ", elapsed_time)

    print("\nDone.")




def compare_metrics_new():
    experiment_id = "example_100_100"

    file_1 = open("experiments/" + experiment_id + "/controllers/distances/" + "positionwise" + ".txt", 'r')
    num_families = int(file_1.readline())
    second_line = int(file_1.readline())
    num_lines = int(file_1.readline())

    file_2 = open("experiments/" + experiment_id + "/controllers/distances/" + "bordawise" + ".txt", 'r')
    num_families = int(file_2.readline())
    second_line = int(file_2.readline())
    num_lines = int(file_2.readline())

    m1 = []
    m2 = []

    for i in range(num_families):
        for j in range(i + 1, num_families):
            line_1 = file_1.readline().rstrip("\n").split(' ')
            line_2 = file_2.readline().rstrip("\n").split(' ')

            x = float(line_1[2])
            m1.append(x)
            y = float(line_2[2])
            m2.append(y)

    print(stats.pearsonr(m1, m2))

    plt.scatter(m1, m2, color='purple', s=1, alpha=0.01)
    plt.savefig("positionwise vs bordawise.png")
    plt.show()


def compute_dissat():

    experiment_id = "final"
    algorithm = "greedy"
    method = "hb"

    file_name = "experiments/" + experiment_id + "/controllers/approx/" + experiment_id + "_"+method+"_" + algorithm +"_dis.txt"
    file_output = open(file_name, 'w')
    num_lines = 860
    file_output.write(str(num_lines)+"\n")

    for Z in range(num_lines):

        print(Z)
        election_id = "core_" + str(Z)

        # exact
        file_name = "experiments/" + experiment_id + "/controllers/orders/" + election_id + "_"+method+".txt"
        file_controllers = open(file_name, 'r')
        num_electitons = int(file_controllers.readline())
        num_winners = int(file_controllers.readline())
        time = float(file_controllers.readline())
        winners = []
        for i in range(num_winners):
            winners.append(int(file_controllers.readline()))

        elections_1, params = el.import_elections(experiment_id, election_id)
        params['orders'] = num_winners

        #print(orders)
        if method == "hb":
            score_1 = win.check_hb_dissat(copy.deepcopy(elections_1['votes'][0]), params, winners)
        elif method == "pav":
            score_1 = win.check_pav_dissat(copy.deepcopy(elections_1['votes'][0]), params, winners)

        # approx
        #"""
        if method == "hb":
            winners = win.get_winners_approx_hb(copy.deepcopy(elections_1['votes'][0]), params, algorithm)
        elif method == "pav":
            winners = win.get_winners_approx_pav(copy.deepcopy(elections_1['votes'][0]), params, algorithm)

        #print(orders)
        if method == "hb":
            score_2 = win.check_hb_dissat(copy.deepcopy(elections_1['votes'][0]), params, winners)
        elif method == "pav":
            score_2 = win.check_pav_dissat(copy.deepcopy(elections_1['votes'][0]), params, winners)



        #print(score_1, score_2)
        #output = score_2 / score_1

        if score_1 == 0:
            score_1 = 1.

        if (score_1 > score_2):
            print("si")
            score_2 = score_1

        output = score_2 / score_1

        #print(output)

        #if output > 1.:
        #    output = 1.

        #print(output)

        file_output.write(str(output)+"\n")

    file_output.close()


    print("\nDone.")


def compute_approx_table():

    experiment_id = "hidalgo"
    #algorithm = "removal"
    method = "pav"

    file_name = "experiments/" + experiment_id + "/controllers/approx/" + experiment_id + "_"+method+"_all.txt"
    file_output = open(file_name, 'w')
    num_lines = 760
    file_output.write(str(num_lines)+"\n")

    for Z in range(num_lines):

        print(Z)
        election_id = "core_" + str(Z)

        # exact
        file_name = "experiments/" + experiment_id + "/controllers/winners_"+method+"/" + election_id + "_"+method+".txt"
        file_controllers = open(file_name, 'r')
        num_elections = int(file_controllers.readline())
        num_winners = int(file_controllers.readline())
        time = float(file_controllers.readline())
        winners = []
        for i in range(num_winners):
            winners.append(int(file_controllers.readline()))

        elections_1, params = el.import_elections(experiment_id, election_id)
        params['orders'] = num_winners

        #print(orders)
        if method == "pav":
            score = win.check_pav_score(copy.deepcopy(elections_1['votes'][0]), params, winners)
        elif method == "hb":
            score = win.check_hb_score(copy.deepcopy(elections_1['votes'][0]), params, winners)

        # approx
        #"""
        if method == "pav":
            winners_1 = win.get_winners_approx_pav(copy.deepcopy(elections_1['votes'][0]), params, "greedy")
            winners_2 = win.get_winners_approx_pav(copy.deepcopy(elections_1['votes'][0]), params, "removal")
        elif method == "hb":
            winners_1 = win.get_winners_approx_hb(copy.deepcopy(elections_1['votes'][0]), params, "greedy")
            winners_2 = win.get_winners_approx_hb(copy.deepcopy(elections_1['votes'][0]), params, "removal")

        #print(orders)
        if method == "pav":
            score_1 = win.check_pav_score(copy.deepcopy(elections_1['votes'][0]), params, winners_1)
            score_2 = win.check_pav_score(copy.deepcopy(elections_1['votes'][0]), params, winners_2)
        elif method == "hb":
            score_1 = win.check_hb_score(copy.deepcopy(elections_1['votes'][0]), params, winners_1)
            score_2 = win.check_hb_score(copy.deepcopy(elections_1['votes'][0]), params, winners_2)


        print(score, score_1, score_2)

        if (score_1 > score):
            score = score_1

        if (score_2 > score):
            score = score_2


        file_output.write(str(score)+" "+str(score_1)+" "+str(score_2)+"\n")


    file_output.close()


    print("\nDone.")


def save_selected_approx(experiment_id, core):
    # Import controllers: basic
    print("# Import controllers: basic")

    file_controllers = open("experiments/" + experiment_id + "/controllers/models/basic.txt", 'r')
    num_basic_families = int(file_controllers.readline())
    basic_family_name = [0 for _ in range(num_basic_families)]
    basic_family_special = [0 for _ in range(num_basic_families)]
    basic_family_size = [0 for _ in range(num_basic_families)]

    for i in range(num_basic_families):
        line = file_controllers.readline().replace(" ", "").rstrip("\n").split(',')

        basic_family_name[i] = line[1]
        basic_family_special[i] = float(line[2])
        basic_family_size[i] = int(line[0])

    file_controllers.close()

    number_of_basic_elections = sum(basic_family_size)

    # Generate mapping
    print("# Generate mapping")

    mapping = [0 for _ in range(number_of_basic_elections)]

    ctr = 0
    for i in range(num_basic_families):
        for j in range(basic_family_size[i]):
            mapping[ctr] = str(basic_family_name[i]) + str(basic_family_special[i])
            ctr += 1

    # Import controllers: core
    print("# Import controllers: core")

    file_controllers = open("experiments/" + experiment_id + "/controllers/models/" + core + ".txt", 'r')
    num_core_families = int(file_controllers.readline())
    core_family_name = [0 for _ in range(num_core_families)]
    core_family_special = [0 for _ in range(num_core_families)]
    core_family_size = [0 for _ in range(num_core_families)]

    for i in range(num_core_families):
        line = file_controllers.readline().replace(" ", "").rstrip("\n").split(',')

        core_family_name[i] = line[1]
        core_family_special[i] = float(line[2])
        core_family_size[i] = int(line[0])

    file_controllers.close()

    number_of_core_elections = sum(core_family_size)

    # Generate core lis
    print("# Generate core list")

    core_list = []
    for i in range(num_core_families):
        core_list.append(str(core_family_name[i]) + str(core_family_special[i]))

    print(core_list)

    # Import distances
    print("# Import values")

    name = experiment_id + "_hb_" + type + "_dis"
    file_name = "experiments/" + experiment_id + "/controllers/approx/" + name + ".txt"
    file_ = open(file_name, 'r')

    number_of_values = int(file_.readline())

    hist_data = [[0 for _ in range(number_of_core_elections)] for _ in range(number_of_core_elections)]

    controllers = []

    for i in range(number_of_values):
        line = file_.readline()
        print(i, mapping[i])
        if mapping[i] in core_list:
            controllers.append(float(line))

    # Save hist_data to file
    print("# Save hist_data to file")

    name = core + "_hb_" + type + "_dis"
    file_name = "experiments/" + str(experiment_id) + "/controllers/approx/" + name + ".txt"
    file_ = open(file_name, 'w')

    file_.write(str(number_of_core_elections) + "\n")

    print(number_of_core_elections)
    for i in range(number_of_core_elections):
        file_.write(str(controllers[i]) + "\n")

    file_.close()


def save_selected_distances():

    experiment_id = "final"
    core = "mini_core"

    num_elections = 1
    special = 0

    # Import controllers: basic
    print("# Import controllers: basic")

    file_controllers = open("experiments/" + experiment_id + "/controllers/models/basic.txt", 'r')
    num_basic_families = int(file_controllers.readline())
    basic_family_name = [0 for _ in range(num_basic_families)]
    basic_family_special = [0 for _ in range(num_basic_families)]
    basic_family_size = [0 for _ in range(num_basic_families)]

    for i in range(num_basic_families):
        line = file_controllers.readline().replace(" ", "").rstrip("\n").split(',')

        basic_family_name[i] = line[1]
        basic_family_special[i] = float(line[2])
        basic_family_size[i] = int(line[0])

    file_controllers.close()

    number_of_basic_elections = sum(basic_family_size)

    # Generate mapping
    print("# Generate mapping")

    mapping = [0 for _ in range(number_of_basic_elections)]

    ctr = 0
    for i in range(num_basic_families):
        for j in range(basic_family_size[i]):
            mapping[ctr] = str(basic_family_name[i]) + str(basic_family_special[i])
            ctr += 1

    # Import controllers: core
    print("# Import controllers: core")

    file_controllers = open("experiments/" + experiment_id + "/controllers/models/" + core + ".txt", 'r')
    num_core_families = int(file_controllers.readline())
    core_family_name = [0 for _ in range(num_core_families)]
    core_family_special = [0 for _ in range(num_core_families)]
    core_family_size = [0 for _ in range(num_core_families)]

    for i in range(num_core_families):
        line = file_controllers.readline().replace(" ", "").rstrip("\n").split(',')

        core_family_name[i] = line[1]
        core_family_special[i] = float(line[2])
        core_family_size[i] = int(line[0])

    file_controllers.close()

    number_of_core_elections = sum(core_family_size)

    # Generate core lis
    print("# Generate core list")

    core_list = []
    for i in range(num_core_families):
        core_list.append(str(core_family_name[i]) + str(core_family_special[i]))

    # Import distances
    print("# Import distances")

    file_name = "experiments/" + experiment_id + "/controllers/distances/" + experiment_id + ".txt"
    file_ = open(file_name, 'r')

    number_of_elections = int(file_.readline())
    number_of_families = int(file_.readline())
    number_of_distances = int(file_.readline())

    hist_data = [[0 for _ in range(number_of_core_elections)] for _ in range(number_of_core_elections)]

    results = []

    for a in range(number_of_elections):

        for b in range(a+1, number_of_elections):
            line = file_.readline()
            line = line.split(' ')

            if mapping[a] in core_list and mapping[b] in core_list:
                results.append(float(line[2]))

    # Save hist_data to file
    print("# Save hist_data to file")

    file_name = "experiments/" + str(experiment_id) + "/controllers/distances/" + str(core) + ".txt"
    file_ = open(file_name, 'w')

    file_.write(str(number_of_core_elections) + "\n")
    file_.write(str(num_core_families) + "\n")
    file_.write(str(len(results)) + "\n")

    ctr = 0
    print(number_of_core_elections)
    for i in range(number_of_core_elections):
        for j in range(i + 1, number_of_core_elections):
            file_.write(str(i) + ' ' + str(j) + ' ' + str(results[ctr]) + "\n")
            ctr += 1

    file_.close()


def save_selected_time():


    experiment_id = "final"
    core = "final_core"

    num_elections = 1
    special = 0

    # Import controllers: basic
    print("# Import controllers: basic")

    file_controllers = open("experiments/" + experiment_id + "/controllers/models/basic.txt", 'r')
    num_basic_families = int(file_controllers.readline())
    basic_family_name = [0 for _ in range(num_basic_families)]
    basic_family_special = [0 for _ in range(num_basic_families)]
    basic_family_size = [0 for _ in range(num_basic_families)]

    for i in range(num_basic_families):
        line = file_controllers.readline().replace(" ", "").rstrip("\n").split(',')

        basic_family_name[i] = line[1]
        basic_family_special[i] = float(line[2])
        basic_family_size[i] = int(line[0])

    file_controllers.close()

    number_of_basic_elections = sum(basic_family_size)

    # Generate mapping
    print("# Generate mapping")

    mapping = [0 for _ in range(number_of_basic_elections)]

    ctr = 0
    for i in range(num_basic_families):
        for j in range(basic_family_size[i]):
            mapping[ctr] = str(basic_family_name[i]) + str(basic_family_special[i])
            ctr += 1

    # Import controllers: core
    print("# Import controllers: core")

    file_controllers = open("experiments/" + experiment_id + "/controllers/models/" + core + ".txt", 'r')
    num_core_families = int(file_controllers.readline())
    core_family_name = [0 for _ in range(num_core_families)]
    core_family_special = [0 for _ in range(num_core_families)]
    core_family_size = [0 for _ in range(num_core_families)]

    for i in range(num_core_families):
        line = file_controllers.readline().replace(" ", "").rstrip("\n").split(',')

        core_family_name[i] = line[1]
        core_family_special[i] = float(line[2])
        core_family_size[i] = int(line[0])

    file_controllers.close()

    number_of_core_elections = sum(core_family_size)

    # Generate core lis
    print("# Generate core list")

    core_list = []
    for i in range(num_core_families):
        core_list.append(str(core_family_name[i]) + str(core_family_special[i]))

    print(core_list)

    # Import distances
    print("# Import values")

    name = experiment_id + "_hb"
    file_name = "experiments/" + experiment_id + "/controllers/times/" + name + ".txt"
    file_ = open(file_name, 'r')

    number_of_values = int(file_.readline())

    hist_data = [[0 for _ in range(number_of_core_elections)] for _ in range(number_of_core_elections)]

    results = []

    for i in range(number_of_values):
        line = file_.readline()
        print(i, mapping[i])
        if mapping[i] in core_list:
            results.append(float(line))

    # Save hist_data to file
    print("# Save hist_data to file")

    name = core + "_hb"
    file_name = "experiments/" + str(experiment_id) + "/controllers/times/" + name + ".txt"
    file_ = open(file_name, 'w')

    file_.write(str(number_of_core_elections) + "\n")

    print(number_of_core_elections)
    for i in range(number_of_core_elections):
        file_.write(str(results[i]) + "\n")

    file_.close()


def after():

    experiment_id = "final"
    K = 10

    print("A")
    generate_votes_from_distances(experiment_id, 800)

    print("B")
    winners = compute_canonical_winners(experiment_id, method="approx_cc", num_winners=K)

    print("C")
    save_to_soc(experiment_id, winners)

    #print("D")
    #can.compute_time(experiment_id, orders, method="pav", num_winners=int(sys.argv[3]))

    print(winners)
    print("\nDone.")

##############
###        ###
###  DiDi  ###
###        ###
##############



def convert_to_soc(experiment, num_elections=None, name=None):

    for id in range(num_elections):

        elections_id = "core_" + str(id)

        elections, params = el.import_elections(experiment, elections_id)

        #num_elections = 1
        num_candidates = params['candidates']
        num_voters = params['voters']

        file_name = "experiments/" + experiment + "/elections/soc_original/core_" + str(id) + ".soc"

        file_ = open(file_name, 'w')

        file_.write("# " + name + "\n")
        file_.write(str(num_candidates) + "\n")

        for i in range(num_candidates):
            file_.write(str(i+1) + ', ' + 'c' + str(i+1) + "\n")

        c = Counter(map(tuple, elections['votes'][0]))
        counted_votes = [[count, list(row)] for row, count in c.items()]
        counted_votes = sorted(counted_votes, reverse=True)

        file_.write(str(num_voters) + ', ' + str(num_voters) + ', ' + str(len(counted_votes)) + "\n")

        for i in range(len(counted_votes)):

            file_.write(str(counted_votes[i][0]) + ', ')

            for j in range(num_candidates):
                file_.write(str(counted_votes[i][1][j]))
                if j < num_candidates - 1:
                    file_.write(", ")
                else:
                    file_.write("\n")

        file_.close()


def prepare_approx_cc_order(name, limit=200):

    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
    file_name = os.path.join(file_name, "experiments", str(name), "controllers", "orders", "positionwise_approx_cc.txt")
    file_ = open(file_name, 'r')

    spam_ = int(file_.readline())
    num_elections = int(file_.readline())
    spam_ = str(file_.readline())

    for i in range(limit):

        target = str(file_.readline().replace("\n", ""))

        src = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
        src = os.path.join(src, "experiments", str(name), "../elections", "soc_original", "core_" + str(target) + ".soc")

        dst = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
        dst = os.path.join(dst, "experiments", str(name), "../elections", "soc_positionwise_approx_cc", "core_" + str(i) + ".soc")

        copyfile(src, dst)


def show_new_ilp(name, metric="bordawise"):

    name = "example_100_100"
    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
    file_name = os.path.join(file_name, "experiments", str(name), "controllers", "times", "final_core_hb.txt")
    file_ = open(file_name, 'r')

    num_elections = int(file_.readline())

    times = [float(file_.readline()) for _ in range(num_elections)]

    target = [i for i in range(0,30)]

    name = "example_100_100"
    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
    file_name = os.path.join(file_name, "experiments", str(name), "controllers", "distances", "bordawise.txt")
    file_ = open(file_name, 'r')

    num_elections = int(file_.readline())
    file_.readline()
    file_.readline()

    distances = [0. for _ in range(800)]

    for i in range(num_elections):
        for j in range(i+1, num_elections):
            line = file_.readline().replace("\n", "").split(" ")
            #print(i,j, line[2])
            dist = float(line[2])

            if j in target:
                distances[i] += dist

            if i in target:
                distances[j] += dist

    #print(distances)

    print(len(times), len(distances))

    """
    num_candidates = 300
    for i in range(num_elections):
        if times[i] > num_candidates:
            times[i] = num_candidates
    """

    distances = [x/30. for x in distances]

    #times = np.log(times)
    times = np.log(times)
    pear = stats.pearsonr(times, distances)
    pear = round(pear[0],2)
    print(pear)

    exp_name = name
    model = obj.Model_xd(exp_name, metric)

    print(model.families[0].size)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.axis('off')

    left = 0
    for k in range(model.num_families):
        right = left + model.families[k].size
        ax.scatter(distances[left:right], times[left:right],
                   color=model.families[k].color, label=model.families[k].label,
                   alpha=model.families[k].alpha, s=9)
        left = right

    title_text = str(model.num_voters) + " voters  x  " + str(model.num_candidates) + " candidates"
    pear_text = "PCC = " + str(pear)
    add_text = ax.text(0.7, 0.8, pear_text, transform=ax.transAxes)
    plt.title(title_text)
    plt.xlabel("average distance from IC elections")
    plt.ylabel("log ( time )")
    core = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
    file_name = os.path.join(core, "images", str(exp_name) + "_map.png")
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(file_name, bbox_extra_artists=(lgd, add_text), bbox_inches='tight')
    plt.show()


def show_new_zip(name):

    name = "example_100_100"
    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
    file_name = os.path.join(file_name, "experiments", str(name), "controllers", "zip_size.txt")
    file_ = open(file_name, 'r')

    num_elections = 800 #int(file_.readline())

    times = [float(file_.readline()) for _ in range(num_elections)]

    target = [i for i in range(0,30)]

    name = "example_100_100"
    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
    file_name = os.path.join(file_name, "experiments", str(name), "controllers", "distances", "positionwise.txt")
    file_ = open(file_name, 'r')

    num_elections = int(file_.readline())
    file_.readline()
    file_.readline()

    distances = [0. for _ in range(800)]

    for i in range(num_elections):
        for j in range(i+1, num_elections):
            line = file_.readline().replace("\n", "").split(" ")
            #print(i,j, line[2])
            dist = float(line[2])

            if j in target:
                distances[i] += dist

            if i in target:
                distances[j] += dist

    #print(distances)

    print(len(times), len(distances))

    """
    num_candidates = 300
    for i in range(num_elections):
        if times[i] > num_candidates:
            times[i] = num_candidates
    """

    distances = [x/30. for x in distances]

    pear = stats.pearsonr(times, distances)
    pear = round(pear[0],2)
    print(pear)

    exp_name = name
    model = obj.Model_xd(exp_name)

    print(model.families[0].size)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.axis('off')

    left = 0
    for k in range(model.num_families):
        right = left + model.families[k].size
        ax.scatter(distances[left:right], times[left:right],
                   color=model.families[k].color, label=model.families[k].label,
                   alpha=model.families[k].alpha, s=9)
        left = right

    title_text = str(model.num_voters) + " voters  x  " + str(model.num_candidates) + " candidates"
    add_text = ax.text(0.7, 0.8, "", transform=ax.transAxes)
    plt.title(title_text)
    plt.xlabel("average distance from IC elections")
    plt.ylabel("normalized zip size")
    core = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
    file_name = os.path.join(core, "images", str(exp_name) + "_zip.png")
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(file_name, bbox_extra_artists=(lgd, add_text), bbox_inches='tight')
    plt.show()


def show_new_approx_removal(name):

    name = "example_100_100"
    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
    file_name = os.path.join(file_name, "experiments", str(name), "controllers", "approx", "final_core_hb_removal_dis.txt")
    file_ = open(file_name, 'r')

    num_elections = int(file_.readline())

    times = [float(file_.readline()) for _ in range(num_elections)]

    target = [i for i in range(0,30)]

    name = "example_100_100"
    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
    file_name = os.path.join(file_name, "experiments", str(name), "controllers", "distances", "positionwise.txt")
    file_ = open(file_name, 'r')

    num_elections = int(file_.readline())
    file_.readline()
    file_.readline()

    distances = [0. for _ in range(800)]

    for i in range(num_elections):
        for j in range(i+1, num_elections):
            line = file_.readline().replace("\n", "").split(" ")
            #print(i,j, line[2])
            dist = float(line[2])

            if j in target:
                distances[i] += dist

            if i in target:
                distances[j] += dist

    #print(distances)

    print(len(times), len(distances))


    distances = [x/30. for x in distances]

    pear = stats.pearsonr(times, distances)
    pear = round(pear[0],2)
    print(pear)

    exp_name = name
    model = obj.Model_xd(exp_name)

    print(model.families[0].size)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.axis('off')


    plt.ylim(0.98,1.25)

    left = 0
    for k in range(model.num_families):
        right = left + model.families[k].size
        ax.scatter(distances[left:right], times[left:right],
                   color=model.families[k].color, label=model.families[k].label,
                   alpha=model.families[k].alpha, s=9)
        left = right

    title_text = str(model.num_voters) + " voters  x  " + str(model.num_candidates) + " candidates"
    add_text = ax.text(0.7, 0.8, "", transform=ax.transAxes)
    plt.title(title_text)
    plt.xlabel("average distance from IC elections")
    plt.ylabel("REMOVAL")
    core = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
    file_name = os.path.join(core, "images", str(exp_name) + "_removal.png")
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(file_name, bbox_extra_artists=(lgd, add_text), bbox_inches='tight')
    plt.show()


def show_new_approx_greedy(name):

    name = "example_100_100"
    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
    file_name = os.path.join(file_name, "experiments", str(name), "controllers", "approx", "final_core_hb_greedy_dis.txt")
    file_ = open(file_name, 'r')

    num_elections = int(file_.readline())


    times = [float(file_.readline()) for _ in range(num_elections)]

    target = [i for i in range(0, 30)]

    name = "example_100_100"
    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
    file_name = os.path.join(file_name, "experiments", str(name), "controllers", "distances", "positionwise.txt")
    file_ = open(file_name, 'r')

    num_elections = int(file_.readline())
    file_.readline()
    file_.readline()

    distances = [0. for _ in range(800)]

    for i in range(num_elections):
        for j in range(i + 1, num_elections):
            line = file_.readline().replace("\n", "").split(" ")
            # print(i,j, line[2])
            dist = float(line[2])

            if j in target:
                distances[i] += dist

            if i in target:
                distances[j] += dist

    # print(distances)

    print(len(times), len(distances))

    """
    num_candidates = 300
    for i in range(num_elections):
        if times[i] > num_candidates:
            times[i] = num_candidates
    """

    distances = [x / 30. for x in distances]

    pear = stats.pearsonr(times, distances)
    pear = round(pear[0], 2)
    print(pear)

    exp_name = name
    model = obj.Model_xd(exp_name)

    print(model.families[0].size)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.axis('off')

    plt.ylim(0.98,1.25)

    left = 0
    for k in range(model.num_families):
        right = left + model.families[k].size
        ax.scatter(distances[left:right], times[left:right],
                   color=model.families[k].color, label=model.families[k].label,
                   alpha=model.families[k].alpha, s=9)
        left = right

    title_text = str(model.num_voters) + " voters  x  " + str(model.num_candidates) + " candidates"
    add_text = ax.text(0.7, 0.8, "", transform=ax.transAxes)
    plt.title(title_text)
    plt.xlabel("average distance from IC elections")
    plt.ylabel("GREEDY")
    core = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
    file_name = os.path.join(core, "images", str(exp_name) + "_greedy.png")
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(file_name, bbox_extra_artists=(lgd, add_text), bbox_inches='tight')
    plt.show()


#### NEW ####

#def compute_cloud_distance(experiment_id, x=-1, num_clouds=100, cloud_size=10,  method="", num_candidates=0):
def compute_cloud_distance(experiment_id, x=-1, num_clouds=100, cloud_size=1, method="", num_candidates=0):

    num_voters = 100
    num_elections = 1
    elections_type = method
    #print(method, x)

    distance_name = "positionwise"
    metric_type = "emd"

    # COMPUTE ALL 'GUESS' ELECTIONS

    for T1 in range(num_clouds * cloud_size):
        elections_id_a = "guess_" + str(T1)
        el.generate_elections(experiment_id, elections_type, elections_id_a, num_elections,
                              num_voters, num_candidates, x)

    # COMPUTE ALL DISTANCES
    distances = np.zeros(num_clouds)
    for c in range(num_clouds):
        #print(c)
        cost_table = np.zeros([cloud_size, cloud_size])
        for T1 in range(cloud_size):
            for T2 in range(cloud_size):
                elections_id_a = "guess_" + str(c*cloud_size + T1)
                elections_id_b = "core_" + str(c*cloud_size + T2)
                elections_ids = [elections_id_a, elections_id_b]
                cost_table[T1][T2] = metr.get_distance(experiment_id, distance_name, elections_ids, metric_type)[0]

        # FIND MINIMAL MATCHING
        measure = metr.get_minimal_matching_cost(cost_table, cloud_size)
        measure /= float(cloud_size)
        #print(measure)
        distances[c] = measure

    for T1 in range(num_clouds * cloud_size):
        file_name = "guess_" + str(T1) + '.soc'
        path = os.path.join(os.getcwd(), 'experiments', experiment_id, '../elections', 'soc_original', file_name)
        try:
            os.remove(path)
        except:
            print("Cannot remove the file!")

    return np.mean(distances)


def approx_real_data(exp_name, method, num_candidates):

    if method in ["impartial_culture", "identity", "netflix", "sushi", "formula",
                  "meath", "dublin_north",  "dublin_west", "equality",
                  "1d_interval", "2d_square", "single_crossing"]:
        min_value = compute_cloud_distance(exp_name, method=method, num_candidates=num_candidates)

    elif method == "urn_model" or method == "mallows"\
            or method == "didi" or method == "pl":

        params = import_params(exp_name, method, num_candidates)
        min_value = compute_cloud_distance(exp_name, x=params, method=method, num_candidates=num_candidates)

    #elif method == "mallows05":
    #    min_value = compute_cloud_distance(exp_name, x=[0.5], method=method, num_candidates=num_candidates)


    print(round(min_value,5))

    file_name = os.path.join("experiments", str(exp_name), "controllers", "distances", method +  ".txt")
    file_ = open(file_name, 'w')
    file_.write(str(min_value) + "\n")
    file_.close()
    #print("DONE")


def import_params(experiment_id, method, num_candidates):

    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')
    file_name = os.path.join(file_name, "experiments", experiment_id, "controllers", "params", method + ".txt")
    file_ = open(file_name, 'r')
    params = []

    if method == "urn_model" or method == "mallows":
        params = float(file_.readline())

    elif method == "didi" or method == "pl":
        [params.append(float(file_.readline())) for _ in range(num_candidates)]

    file_.close()
    return params

"""
def compute_training_distance(experiment_id, x=-1, num_clouds=1, cloud_size=20,  method="", num_candidates=0):

    num_voters = 100
    num_elections = 1
    elections_type = method
    second_x = -1
    print(method, x)

    distance_id = "positionwise"
    metric_type = "emd"

    # compute all 'guess' elections
    for T1 in range(num_clouds * cloud_size):
        elections_id_a = "guess_" + str(T1)
        el.generate_elections(experiment_id, elections_type, elections_id_a, num_elections,
                              num_voters, num_candidates, x)
        elections_id_b = "train_" + str(T1)
        el.generate_elections(experiment_id, "netflix", elections_id_b, num_elections,
                              num_voters, num_candidates, second_x)

    #  compute all distances
    distances = np.zeros(num_clouds)
    for c in range(num_clouds):
        print(c)
        cost_table = np.zeros([cloud_size, cloud_size])
        for T1 in range(cloud_size):
            for T2 in range(cloud_size):
                elections_id_a = "guess_" + str(c*cloud_size + T1)
                elections_id_b = "train_" + str(c*cloud_size + T2)
                elections_ids = [elections_id_a, elections_id_b]
                cost_table[T1][T2] = metr.get_distance(experiment_id, distance_id, elections_ids, metric_type)[0]

        #  find minimal matching
        measure = metr.get_minimal_matching_cost(cost_table, cloud_size)
        measure /= float(cloud_size)
        #print(measure)
        distances[c] = measure

    #print(distances)
    return np.mean(distances)
"""


ctr = 0

# def approx_training(exp_name, method, target, num_candidates, n_calls=100, num_clouds=5, cloud_size=10):
#
#     def local_optim(x):
#
#         if len(x) == 1:
#             x = x[0]
#
#         num_voters = 100
#         num_elections = 1
#         second_x = -1
#
#         distance_id = "positionwise"
#         metric_type = "emd"
#
#         # compute all 'guess' elections
#         for T1 in range(num_clouds * cloud_size):
#             elections_id_a = "guess_" + str(T1)
#             el.generate_elections(exp_name, method, elections_id_a, num_elections,
#                                   num_voters, num_candidates, x)
#             elections_id_b = "train_" + str(T1)
#             el.generate_elections(exp_name, target, elections_id_b, num_elections,
#                                   num_voters, num_candidates, second_x)
#
#         #  compute all distances
#         distances = np.zeros(num_clouds)
#         for c in range(num_clouds):
#             cost_table = np.zeros([cloud_size, cloud_size])
#             for T1 in range(cloud_size):
#                 for T2 in range(cloud_size):
#                     elections_id_a = "guess_" + str(c * cloud_size + T1)
#                     elections_id_b = "train_" + str(c * cloud_size + T2)
#                     elections_ids = [elections_id_a, elections_id_b]
#                     cost_table[T1][T2] = \
#                         metr.get_distance(exp_name, distance_id, elections_ids, metric_type)[0]
#
#             #  find minimal matching
#             measure = metr.get_minimal_matching_cost(cost_table, cloud_size)
#             measure /= float(cloud_size)
#             distances[c] = measure
#
#         global ctr
#         ctr += 1
#         print(ctr, round(np.mean(distances), 2))
#         return np.mean(distances)
#
#     if method in ["urn_model", "mallows"]:
#
#         space = [(0., 1.) for _ in range(1)]
#         res = gp_minimize(local_optim, space, n_calls=n_calls)
#         print(res.x)
#
#     elif method in ["didi", "pl"]:
#
#         space = [(0.00001, 1.) for _ in range(num_candidates)]
#         res = gp_minimize(local_optim, space, n_calls=n_calls)
#
#     print(res.x)
#     print(res.fun)
#     file_name = os.path.join("experiments", str(exp_name), "controllers", "params", method +  ".txt")
#     file_ = open(file_name, 'w')
#     for value in res.x:
#         file_.write(str(value) + "\n")
#     file_.close()
#     print("DONE")


def diagonal_math(num_candidates):

    diagonal = sum([sum([abs(i-j) / float(num_candidates) for j in range(num_candidates)]) for i in range(num_candidates)])
    #print(diagonal)
    return diagonal


def diagonal(experiment_id, num_candidates, size=100):

    num_voters = 100
    num_elections = 1
    x = 0.1

    distance_name = "positionwise"
    metric_type = "emd"

    for num_candidates in [5,10,20,50,100]:
        print("----")

        elections_id_a = "identity"
        el.generate_elections(experiment_id, "identity", elections_id_a, num_elections,
                              num_voters, num_candidates, x)

        for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 2., 3., 4., 5.]:
            #print(x)
            # compute all 'guess' elections
            for T1 in range(size):
                elections_id_b = "urn_" + str(T1)
                el.generate_elections(experiment_id, "urn_model", elections_id_b, num_elections,
                                      num_voters, num_candidates, x)

            distances = np.zeros(size)
            for T1 in range(size):
                    elections_id_b = "urn_" + str(T1)
                    elections_ids = [elections_id_a, elections_id_b]
                    distances[T1] = metr.get_distance(experiment_id, distance_name, elections_ids, metric_type)[0]

            output = np.mean(distances) / diagonal_math(num_candidates)
            print(output)


# NEW 30.04.2020
def pure_distance(experiment_id, num_candidates, target, size=200):

    num_voters = 100
    num_elections = 1
    x = 0.1

    distance_name = "positionwise"
    metric_type = "emd"

    for num_candidates in [5,10,20,50,100]:
        print("----")

        for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 2., 3., 4., 5.]:
            print(x)

            distances = np.zeros(size)
            for T1 in range(size):
                    elections_id = "core_" + str(T1)
                    distances[T1] = metr.get_pure_distance(experiment_id, elections_id, target)

            output = np.mean(distances) / diagonal_math(num_candidates)
            print(output)


def bordawise_diagonal(experiment_id="workshop"):

    size = 1

    num_voters = 100
    num_elections = 1
    x = -1

    distance_name = "bordawise"
    metric_type = "emd"

    for num_candidates in [3]:

        distances = np.zeros(size)
        for T1 in range(size):
            elections_id_a = "core_a_" + str(T1)
            el.generate_elections(experiment_id, "impartial_culture", elections_id_a, num_elections,
                                  num_voters, num_candidates, x)
            elections_id_b = "core_b_" + str(T1)
            el.generate_elections(experiment_id, "antagonism", elections_id_b, num_elections,
                                  num_voters, num_candidates, x)
            elections_ids = [elections_id_a, elections_id_b]
            distances[T1] = metr.get_distance(experiment_id, distance_name, elections_ids, metric_type)[0]

        output = np.mean(distances) / diagonal_math(num_candidates)
        print(output/num_voters)





"""
    elif method == "didi":
      
        srednia = 0.
        t = 0
        QQ = []
        for T in range(TESTS):
            elections_id_a = "core_" + str(T)
            ele_1 = obj.Elections(experiment_id, elections_id_a)

            vectors = ele_1.votes_to_positionwise_vectors()
            W = []; Q = []; R = []; S = []

            for v in vectors:

                w = 0
                for i in range(num_candidates):
                    w += v[i] * (num_candidates - i)
                my_std = 0.
                for i in range(num_candidates):
                    my_std += (abs(((num_candidates - i) - w)) * v[i])
                Q.append(my_std)
                W.append(w)

            srednia += np.mean(Q)
            t += 1. / np.mean(Q)
            W=sorted(W, reverse=True)
            QQ.append(W)

        print(t/TESTS)

        QQ = np.array(QQ)
        FQ = [np.mean(QQ[:, i]) for i in range(num_candidates)]
      
        QQ = []
        for T in range(TESTS):

            elections_id_a = "core_" + str(T)
            ele_1 = obj.Elections(experiment_id, elections_id_a)

            vectors = ele_1.votes_to_positionwise_vectors()
            W = []
            for v in vectors:
                w = 0
                for i in range(num_candidates):
                    w += v[i] * (num_candidates - i)
                W.append(w)
            Q = [0. for _ in range(num_candidates)]
            for q in range(num_candidates):
                div = 0.
                for d in range(num_candidates):
                    if d != q:
                        div += W[d] ** 2
                Q[q] = W[q] ** 2 / div
                # Q[q] = Q[q]**2

            for q in range(num_candidates):
                if Q[q] < 0.000001:
                    Q[q] = 0.000001

            Q = sorted(Q, reverse=True)
            QQ.append(Q)
        QQ = np.array(QQ)
        FQ = [np.mean(QQ[:, i]) for i in range(num_candidates)]

        print(sum(FQ))

        print("---")
        for i in range(num_candidates):
            FQ[i] /= float(num_candidates)
       
        print("razem: ", sum(FQ))
        for i in range(num_candidates):
            print(FQ[i])

        #FQ = [2.982506523, 2.745119716, 2.42748847, 2.036244664, 1.613205311, 1.225901409, 0.8325162247, 0.5462964273, 0.3007031591, 0.1301391838]
        print(FQ)

        print("OPTIM")
        #space = [(0.99, 1.) for _ in range(1)]

        #FQ = [6.8169,6.6988,6.5255,6.2811,5.9656,5.6090,5.1318,4.6516,4.0380,3.2820]
    

    
    print("start")
    space = [(0.001, 0.1) for _ in range(num_candidates)]
    res = gp_minimize(local_optim, space, n_calls=100)

    print(res)

elif method == "pl":

        QQ = []
        for T in range(TESTS):

            elections_id_a = "core_" + str(T)
            ele_1 = obj.Elections(experiment_id, elections_id_a)


            vectors = ele_1.votes_to_positionwise_vectors()
            W = []
            for v in vectors:
                w = 0
                for i in range(num_candidates):
                    w += v[i] * (num_candidates - i)
                W.append(w)
            Q = [0. for _ in range(num_candidates)]
            for q in range(num_candidates):
                div = 0.
                for d in range(num_candidates):
                    if d != q:
                        div += W[d] ** 2
                Q[q] = W[q] ** 2 / div
                #Q[q] = Q[q]**2

            for q in range(num_candidates):
                if Q[q] < 0.000001:
                    Q[q] = 0.000001

            Q=sorted(Q, reverse=True)
            QQ.append(Q)
        QQ = np.array(QQ)
        FQ = [np.mean(QQ[:,i]) for i in range(num_candidates)]
   

        print(FQ)
        ########################

        file_name = "experiments/" + str(experiment_id) + "/controllers/params/pl.txt"
        file_info = open(file_name, 'w')
        for i in range(len(FQ)):
            file_info.write(str(FQ[i]) + "\n")
        file_info.close()

        ########################

        #x0 = FQ
        #y0 = local_optim(FQ)

        print("start")
        space = [(0.001, 1.) for _ in range(num_candidates)]
        res = gp_minimize(local_optim, space, n_calls=100)

        print(res)

    """


def map_compute_IDEQ(experiment_id, metric_type="emd", distance_name="positionwise"):

    num_lines = 800

    # can be transform into helper function
    results_ID = []

    for i in range(num_lines):
        election_id = "core_" + str(i)
        target = "identity"
        distance = metr.get_pure_distance(experiment_id, election_id, target)
        results_ID.append(round(distance, 2))
        print(i, distance)

    file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "distances", "positionwise_ID.csv")
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["id", "distance"])
        [writer.writerow([it, value]) for it, value in enumerate(results_ID)]

    # can be transform into helper function
    results_EQ = []

    for i in range(num_lines):
        election_id = "core_" + str(i)
        target = "equality"
        distance = metr.get_pure_distance(experiment_id, election_id, target)
        results_EQ.append(round(distance, 2))
        print(i, distance)

    file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "distances", "positionwise_EQ.csv")
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["id", "distance"])
        [writer.writerow([it, value]) for it, value in enumerate(results_EQ)]


def map_normalize_IDEQ(experiment_id, metric_type="emd", distance_name="positionwise"):

    num_lines = 800
    num_candidates = 100

    # IMPORT DISTANCES FROM IDENTITY
    results_ID = []
    file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "distances", "positionwise_ID.csv")
    with open(file_name, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            results_ID.append(float(row['distance']))

    # IMPORT DISTANCES FROM EQUALITY
    results_EQ = []
    file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "distances", "positionwise_EQ.csv")
    with open(file_name, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            results_EQ.append(float(row['distance']))

    # SAVE NORMALIZED VALUES
    file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced", "IDEQ.txt")
    file_output = open(file_name, 'w')

    values = []
    for i in range(num_lines):
        value = results_ID[i] + results_EQ[i]
        values.append(value)

    maximum = max(values)
    minimum = (num_candidates*num_candidates - 1) / 3
    print(maximum, minimum)
    print(maximum/minimum)

    for i in range(num_lines):

        values[i] = (values[i] - minimum) / (maximum - minimum)
        file_output.write(str(round(values[i],4))+"\n")

    file_output.close()






