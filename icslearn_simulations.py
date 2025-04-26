import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import random
import copy
from scipy.special import loggamma
from itertools import combinations, product
from math import floor

import cstrees.cstree as ct
import cstrees.stage as st
import cstrees.scoring as sc

from icslearn import iCStree, merge_istages, find_stage

# generate a random interventional CStree for binary data with 1 interventional data set
def generate_iCStree(num_vars, sample_size = 10, max_cvars = 2, prob_cvar = 0.5, prop_nonsingleton = 1, stage_prop = 0.5, seed = 57):
    np.random.seed(seed)
    random.seed(seed)

    p = num_vars
    cards = [2] * p 
    obscstree = ct.sample_cstree(cards, max_cvars, prob_cvar, prop_nonsingleton=prop_nonsingleton)
    obscstree.labels = ['X' + str(i) for i in range(1, p + 1)]
    obscstree.sample_stage_parameters(alpha = 1)
    obscstreedf = obscstree.to_df(write_probs=True)

    icstreedf = iCStree(obscstreedf, 1)
    icstree = ct.df_to_cstree(icstreedf)
    I = [[1]]

    # need to sample stages to merge in each level with fixed probability
    invariances = []
    for level in range(obscstree.p):
        stages = obscstree.stages[level]
        numstages = len(obscstree.stages[level])
        stageidxs = np.random.choice(numstages, floor(numstages * stage_prop), replace = False)
        selected_stages = [stages[x] for x in stageidxs]
        invariances += selected_stages

    merged_tree = merge_istages(icstree, obscstree, [invariances], I)
    merged_tree.sample_stage_parameters(alpha=1)
    merged_tree.get_stage([]).probs = np.array([0.5,0.5])
    sample = merged_tree.sample(sample_size)

    return [merged_tree, sample, obscstree]

# invariance array
    # an array with a row and column for each observational stage
    # a 1 in the entry of the array if the 
def invariance_list(icstree, obscstree):
    #list of all observational stages:
    obs_stages = []
    for level in range(obscstree.p):
        obs_stages += obscstree.stages[level]
    
    inv_list = [0 for stage in obs_stages]
    for stage in obs_stages:
        stage_element = []
        for i in range(len(stage.list_repr)):
            if isinstance(stage.list_repr[i], set):
                stage_element += [0]
            else:
                stage_element += [stage.list_repr[i]]
        istage = find_stage([1] + stage_element, icstree)

        if isinstance(istage.list_repr[0], set):
            inv_list[obs_stages.index(stage)] = 1

    return inv_list


def acc(truetree, learnedtree, obscstree):
    acc = 0 
    true_inv_list = np.array(invariance_list(truetree, obscstree))
    learned_inv_list = np.array(invariance_list(learnedtree, obscstree))
    diffs = sum(np.logical_xor(true_inv_list.astype(bool), learned_inv_list.astype(bool)))
    acc = (len(true_inv_list) - diffs) / len(true_inv_list)

    return acc

