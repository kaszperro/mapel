ELECTION_FEATURES_WITH_PARAMS = ['largest_cohesive_group',
                                 'number_of_cohesive_groups',
                                 'number_of_cohesive_groups_brute',
                                 'proportionality_degree_pav',
                                 'proportionality_degree_av',
                                 'proportionality_degree_cc',
                                 'justified_ratio',
                                 'cohesiveness',
                                 'partylist',
                                 'highest_cc_score',
                                 'highest_hb_score',
                                 'highest_pav_score',
                                 'greedy_approx_cc_score',
                                 'removal_approx_cc_score',
                                 'greedy_approx_hb_score',
                                 'removal_approx_hb_score',
                                 'greedy_approx_pav_score',
                                 'removal_approx_pav_score',
                                 'rand_approx_pav_score',
                                 'banzhaf_cc_score',
                                 'ranging_cc_score']

MAIN_LOCAL_FEATUERS = []

MAIN_GLOBAL_FEATUERS = ['calculate_distortion', 'calculate_monotonicity']

ELECTION_GLOBAL_FEATURES = {'clustering', 'clustering_kmeans', 'distortion_from_all'}

COLORS = ['blue', 'green', 'black', 'red', 'orange', 'purple', 'brown', 'lime', 'cyan', 'grey']

RULE_NAME_MATRIX = {
    "av": "AV",
    "sav": "SAV",
    "pav": "PAV",
    "slav": "SLAV",
    "cc": "CC",
    "seqpav": "Seq.PAV",
    "revseqpav": "Rev.Seq.PAV",
    "seqslav": "Seq. SLAV",
    "seqcc": "Seq. CC",
    "seqphragmen": "Seq. Phrag.",
    "greedy-monroe": "Gr. Monroe",
    "rule-x": "Rule X",
    "phragmen-enestroem": "Phrag. Ene.",
    "consensus-rule": "Consensus",
    "geom2": "Geometric 2",
    "geom3": "Geometric 3",
    "geom4": "Geometric 4",
    "geom5": "Geometric 5",
}

RULE_NAME_MAP = {
    "av": "AV",
    "sav": "SAV",
    "pav": "PAV",
    "slav": "SLAV",
    "cc": "CC",
    "seqpav": "S.PAV",
    "revseqpav": "R.S.PAV",
    "seqslav": "S.SLAV",
    "seqcc": "S.CC",
    "seqphragmen": "S.Phrag.",
    "greedy-monroe": "Gr.Monroe",
    "rule-x": "Rule X",
    "phragmen-enestroem": "Phrag. Ene.",
    "consensus-rule": "Cons.",
    "geom2": "G2",
    "geom3": "G3",
    "geom4": "G4",
    "geom5": "G5",
}

NICE_NAME = {
    '1d_interval': '1D_Interval',
    '1d_gaussian': '1D_Gaussian',
    '2d_disc': '2D_Disc',
    '2d_square': '2D_Square',
    '2d_gaussian': '2D_Gaussian',
    '3d_cube': '3D_Cube',
    '4d_cube': '4D_Cube',
    '5d_cube': '5D_Cube',
    '10d_cube': '10D_Cube',
    '20d_cube': '20D_Cube',
    '2d_sphere': '2D_Sphere',
    '3d_sphere': '3D_Sphere',
    '4d_sphere': '4D_Sphere',
    '5d_sphere': '5D_Sphere',
    'impartial_culture': 'Impartial_Culture',
    'iac': 'Impartial_Anonymous_Culture',
    'urn_model': 'Urn_Model',
    'conitzer': 'Single-Peaked_(by_Conitzer)',
    'spoc_conitzer': 'SPOC',
    'walsh': 'Single-Peaked_(by_Walsh)',
    'mallows': 'Mallows',
    'norm-mallows': 'Norm-Mallows',
    'single-crossing': 'Single-Crossing',
    'didi': 'DiDi',
    'pl': 'Plackett-Luce',
    'netflix': 'Netflix',
    'sushi': 'Sushi',
    'formula': 'Formula',
    'meath': 'Meath',
    'dublin_north': 'North_Dublin',
    'dublin_west': 'West_Dublin',
    'identity': 'Identity',
    'antagonism': 'Antagonism',
    'uniformity': 'Uniformity',
    'stratification': 'Stratification',
    'real_identity': 'Real_Identity',
    'real_uniformity': 'Real_Uniformity',
    'real_antagonism': 'Real_Antagonism',
    'real_stratification': 'Real_Stratification',
    'group-separable': 'Group-Separable',
    'approval_id': 'Approval_ID',
    'approval_id_0.5': 'Approval_ID_0.5',
    'approval_ic': 'Approval_IC',
    'approval_ic_0.5': 'Approval_IC_0.5',
    'approval_resampling': 'Approval_Resampling',
    'approval_noise_model': 'Approval_Noise_Model',
    'approval_disjoint_resampling': 'approval_disjoint_resampling',
    'approval_urn': 'Approval_Urn',
    'core': 'Core',
    'approval_truncated_mallows': 'Approval_Truncated_Mallows',
    'approval_truncated_urn': 'Approval_Truncated_Urn',
    'approval_simplex_shumallows': 'approval_simplex_shumallows',
    'approval_vcr': 'Approval_VCR',
    'all_votes': 'All_Votes',
    'all': 'All',
    'approval_moving_resampling': 'Approval_Moving_Resampling',
    'unid': 'UNID',
    'anid': 'ANID',
    'stid': 'STID',
    'stan': 'STAN',
    'stun': 'STUN',
    'anun': 'ANUN',
    'approval_jaccard': 'Approval_Jaccard',
    'approval_skeleton': 'approval_skeleton',
    'approval_anti_pjr': 'approval_anti_pjr',
    'hamming': 'hamming',
    'noise_model': 'noise_model',
    'approval_partylist': 'Approval_Partylist',
    'norm-mallows_mixture': 'norm-mallows_mixture',
    'ic': 'IC',
    'id': 'ID',
    'urn': 'Urn',
}

LIST_OF_PREFLIB_MODELS = {'sushi', 'irish', 'glasgow', 'skate', 'formula',
                          'tshirt', 'cities_survey', 'aspen', 'ers',
                          'marble', 'cycling_tdf', 'cycling_gdi',
                          'ice_races', 'grenoble', 'speed_skating',
                          'irish_bis', 'speed_skating_bis', 'skate_bis'}

LIST_OF_FAKE_MODELS = {'identity', 'uniformity', 'antagonism',
                       'stratification', 'mallows_matrix_path',
                       'unid', 'anid', 'stid', 'anun', 'stun', 'stan',
                       'crate', 'walsh_matrix', 'conitzer_matrix',
                       'single-crossing_matrix', 'gs_caterpillar_matrix',
                       'norm-mallows_matrix', 'sushi_matrix',
                       'walsh_path', 'conitzer_path'}

PATHS = {'unid', 'stan', 'anid', 'stid', 'anun', 'stun',
         'mallows_matrix_path', 'walsh_path', 'conitzer_path', }

PARTY_MODELS = {'2d_gaussian_party', '1d_gaussian_party', 'ic_party',
                'walsh_party', 'conitzer_party', 'mallows_party'}

APPROVAL_MODELS = {'impartial_culture', 'ic', 'resampling', 'id',
                   'empty', 'full', 'truncated_urn',
                   'urn', 'euclidean', 'noise_model',
                   'zeros', 'ones',
                   'id_0.5', 'ic_0.5',
                   'half_1', 'half_2', 'disjoint_resampling',
                   'simplex_shumallows',
                   'vcr', 'truncated_mallows', 'moving_resampling',
                   'jaccard', 'skeleton', 'anti_pjr',
                   'partylist'}

APPROVAL_FAKE_MODELS = {'approval_half_1', 'approval_half_2', 'approval_skeleton'}

GRAPH_MODELS = {'erdos_renyi_graph', 'watts_strogatz_graph', 'barabasi_albert_graph',
                'random_geometric_graph', 'random_tree',
                'cycle_graph', 'wheel_graph', 'star_graph', 'ladder_graph', 'circular_ladder_graph',
                'erdos_renyi_graph_path'}

NOT_ABCVOTING_RULES = {'borda_c4', 'random'}

EMBEDDING_RELATED_FEATURE = ['monotonicity_triplets', 'distortion_from_all']

ROOMMATES_PROBLEM_MODELS = {'roommates_ic'}

FEATURES_WITH_DISSAT = {'highest_hb_score', 'highest_pav_score'}

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 12.10.2021 #
# # # # # # # # # # # # # # # #
