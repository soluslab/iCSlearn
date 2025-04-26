import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import copy
from scipy.special import loggamma
from itertools import combinations, product

import cstrees.cstree as ct
import cstrees.stage as st
import cstrees.scoring as sc

def iCol(obstree, numint):
    # obstree -- a pandas df representation of an observational CStree
    # numint -- the number of interventional data sets 
    numstages = obstree.shape[0] - 1
    iColumn = [numint + 1]
    for i in range(numint + 1):
        iColumn += [i for j in range(numstages)]
    return iColumn

def iCStree(obstree, numint):
    intTree = obstree.copy()
    for j in range(numint):
        newIntBranch = obstree.copy().drop([0])
        intTree = pd.concat([intTree, newIntBranch], ignore_index=True)
    intTree.insert(0,"I",iCol(obstree, numint),True)
    intTree.loc[len(intTree)] = ["-" for i in range(intTree.shape[1])]

    if numint > 1:
        for i in range(numint -1):
            intTree["PROB_" + str(i + 1)]
            intTree.loc[[intTree.shape[0] - 1],["PROB_" + str(i + 2)]] = 1/(numint + 1)
    
    intTree.loc[[intTree.shape[0] - 1],["PROB_0"]] = 1/(numint + 1)
    intTree.loc[[intTree.shape[0] - 1],["PROB_1"]] = 1/(numint + 1)
    
    return intTree #currently returns a dataframe.  Update to return a CStree object.

# the following only really works with one interventional data set.  
def merge_istages_oneint(inttree, num_obs_stages, obs_stages):
    # obs_stage given as a list of row indices in observational tree
    newIntTree = inttree.copy()
    for x in obs_stages:
        newIntTree.loc[[x],["I"]] = "*"
        newIntTree = newIntTree.drop([x + num_obs_stages])

    return newIntTree.reset_index(drop=True)

# need a function for merging stages for interventions where the resulting tree
# is not a CStree.  Need to do this using the dictionary structure of stages in CSlearn.
def merge_istage(inttree, obstree, obs_stage, ints):
    # obstree a cstree object
    # inttree a interventional cstree as cstree object
    # obs_stage a stage in the observational tree
    # ints a list of indices of interventions for which to merge the corresponding stages
    intT = copy.deepcopy(inttree)
    ostage = intT.get_stage([0] + obs_stage.list_repr)
    oI = ostage.list_repr[0]
    if type(oI) == set:
        ostage_ints = [t for t in oI]
    else:
        ostage_ints = [oI]
    lev = len(ostage.list_repr) - 1
    stages_in_level = intT.stages[lev]
    
    for t in ints:
        for s in stages_in_level:
            if s.list_repr == ([t] + obs_stage.list_repr):
                stages_in_level.remove(s)

    ostage.list_repr[0] = {t for t in ints + ostage_ints} 

    return intT

# produce an interventional CStree with multiple stages merged
#     ints becomes a list of lists of length equal to obs_stages
#     actually ints is only a list of singletons, one for each intervention
#     obs_stages becomes a list of lists of obs_stages of length equal to ints
def merge_istages(inttree, obstree, obs_stages, int_list):
    n = len(obs_stages)
    iT = copy.deepcopy(inttree)
    for i in range(n):
        for os in obs_stages[i]:
            iT = merge_istage(iT, obstree, os, int_list[i])

    return iT


#general stage proportion function
def stage_prop(cstree, stage):
    numerator = 1
    denominator = 1
    for i, val in enumerate(stage.list_repr):
        denominator *= 1/cstree.cards[i]
        if isinstance(val, set):
            numerator *= len(val)
    prop = numerator * denominator
    
    return prop

# general stage containment function:
def stage_contains(node, stage):
    #node given as list
    if len(node) == 0:
        if len(stage.list_repr) == 0:
            return True
        
    for i, val in enumerate(stage.list_repr):
        if (isinstance(val, set)) and (node[i] not in val):
            return False
        
        if (isinstance(val, int)) and (node[i] != val):
            return False

    return True

def find_stage(node, cstree):
    # node given as list
    assert cstree.stages is not None
    lev = len(node) - 1

    stage = None
    if lev in cstree.stages:
        for s in cstree.stages[lev]:
            if stage_contains(node, s):
                stage = s
                break

    if stage is None:
        print("No stage found for {}".format(node))

    assert stage is not None
    return stage

# adapted get stage counts function

def level_counts(cstree, level: int, data):
    stage_counts = {}

    dataperm = data[cstree.labels].values[1:, :]

    for i in range(len(dataperm)):  # iterate over the samples
        pred_vals = dataperm[i, :level]
        stage = find_stage(pred_vals, cstree) 

        if stage is None:  # singleton stage.
            print('singleton stage')
        
        if stage not in stage_counts:
            stage_counts[stage] = {}

        if dataperm[i, level] in stage_counts[stage]:
            stage_counts[stage][dataperm[i, level]] += 1
        else:
            stage_counts[stage][dataperm[i, level]] = 1

    return stage_counts

# fit parameters
def fit_parameters(cstree, data, alpha_tot = 1.0):
    for lev in range(cstree.p):
        stage_counts = level_counts(cstree, lev, data)
        for stage in stage_counts.keys():
            alpha_stage = alpha_tot * stage_prop(cstree, stage)
            alpha_obs = alpha_stage / cstree.cards[lev]

            stage_counts_total = sum(stage_counts[stage].values())
            probs = [None] * cstree.cards[lev]

            for i in range(cstree.cards[lev]):
                if i not in stage_counts[stage]:
                    if alpha_obs == 0:
                        probs[i] = 0
                    else:
                        probs[i] = alpha_obs / alpha_stage
                else:
                    probs[i] = (alpha_obs + stage_counts[stage][i]) / (
                    alpha_stage + stage_counts_total
                    )
            stage.probs=probs
    return cstree


# scoring a stage in a CStree

# to get stages_counts at level i:
#     stagecounts_i = sc._counts_at_level(cstree, i + 1, dataframe)
#     if outcomes at level i are range(k) for j in range k get counts for 
#     outcome k by doing
#     stagecounts_i[stage][j]
def score_stage(cstree, stage, stage_counts, alpha_tot=1.0):
    level = stage.level + 1
    alpha_stage = alpha_tot * stage_prop(cstree, stage)
    alpha_obs = alpha_stage / cstree.cards[level]
    stage_outs = []
    for i in range(cstree.cards[level]):
        if stage in stage_counts.keys():
            if i in list(stage_counts[stage].keys()):
                stage_outs += [i]
    
    if stage_outs != []:
        stage_counts_tot = sum([stage_counts[stage][j] for j in stage_outs])
    else:
        stage_counts_tot = 0

    score = loggamma(alpha_stage) - loggamma(alpha_stage + stage_counts_tot)

    for i in range(cstree.cards[level]):
        if stage in stage_counts.keys():
            if i in list(stage_counts[stage].keys()):
                score += loggamma(alpha_obs + stage_counts[stage][i]) - loggamma(alpha_obs)

    return score

# scoring a level in a CStree
def score_level(cstree, level: int, data, alpha_tot=1.0):
    stages = cstree.stages[level]
    stage_counts = level_counts(cstree, level + 1, data) 
    level_score = 0
    for stage in stages:
        level_score += score_stage(cstree, stage, stage_counts, alpha_tot=alpha_tot)

    return level_score

# scoring a CStree
def score_tree(cstree, data, alpha_tot=1.0):
    tree_score = 0 
    num_levels = cstree.p  - 1
    for i in range(1, num_levels):
        tree_score += score_level(cstree, i, data, alpha_tot=alpha_tot)

    return tree_score 


# produce a list of all interventional CStrees
#    requires a list of all observational stages
#    a list of all subsets of the above list

def get_best_ilevel(obstree, level: int, num_ints, data, alpha_tot=1.0):
    interventions = [[i] for i in range(1, num_ints + 1)]

    obstreedf = obstree.to_df(write_probs=True)
    base_itreedf = iCStree(obstreedf, num_ints)
    base_itree = ct.df_to_cstree(base_itreedf)

    obsstages = obstree.stages[level]
    obsstages_sublists = [list(combinations(obsstages, r)) for r in range(len(obsstages) + 1)]
    obsstages_sublists = [list(sublist) for g in obsstages_sublists for sublist in g]
    obsstages_lists = [list(element) for element in product(*[obsstages_sublists for i in range(num_ints)])]

    best_level = base_itree.stages[level + 1]
    best_score = score_level(base_itree, level + 1, data, alpha_tot=alpha_tot)
    best_merging = [[] for i in range(num_ints)]
    for L in obsstages_lists:
        merged_itree = merge_istages(base_itree, obstree, L, interventions)
        lscore = score_level(merged_itree, level + 1, data, alpha_tot= alpha_tot)
        
        if lscore > best_score:
            best_level = merged_itree.stages[level + 1]
            best_score = lscore
            best_merging = L


    return [best_level, best_score, best_merging]

def fit_itree(obstree, num_ints, data, alpha_tot=1.0):
    obstreedf = obstree.to_df(write_probs=True)
    base_itreedf = iCStree(obstreedf, num_ints)
    base_itree = ct.df_to_cstree(base_itreedf)

    best_staging = []
    for level in range(obstree.p - 1):
        best_staging += [get_best_ilevel(obstree, level, num_ints, data, alpha_tot=alpha_tot)]
    
    for level in range(obstree.p - 1):
        base_itree = merge_istages(base_itree, obstree, best_staging[level][2], [[i] for i in range(1,num_ints + 1)])

    treescore = sum([best_staging[level][1] for level in range(obstree.p - 1)])
    
    learned_tree = fit_parameters(base_itree, data)
    
    return [learned_tree, treescore] 



# Producing interventional LDAG:

def get_i_edges(icstree):
    i_edges = {}
    for i in range(1, icstree.cards[0]):
        i_edges[i] = []
        for level in range(1, icstree.p - 1):
            stages = icstree.stages[level]
            for stage in stages:
                if not isinstance(stage.list_repr[0], set):
                    obs_stage = [stage.list_repr[j] for j in range(1,len(stage.list_repr))]
                    if level not in i_edges[i] and stage.list_repr[0] == i:
                        i_edges[i] += [level]

    return i_edges

def get_i_labels(icstree, i_edges):
    i_labels= {}
    for i in range(1, icstree.cards[0]):
        i_labels[i] = {}
        for level in i_edges[i]:
            i_labels[i][level] = []
            stages = icstree.stages[level]
            for stage in stages:
                stagelist = stage.list_repr
                stagelength = len(stagelist)
                if isinstance(stagelist[0], set):
                    if i in stagelist[0]:
                        obs_vanish = []
                        for j in range(1,level + 1):
                            if isinstance(stagelist[j],set):
                                obs_vanish += ['*']
                            else:
                                obs_vanish += [stagelist[j]]
                        if len(obs_vanish) < (icstree.p - 1): # can replace
                            for k in range(icstree.p - 1 - len(obs_vanish)): # can replace
                                obs_vanish += ['*']
                        i_labels[i][level] += [obs_vanish]

    return i_labels

def get_iLDAG(icstree, obscstree, icolor = 'red'):
    LDAG = obscstree.to_LDAG()
    newedges = get_i_edges(icstree)
    newedgelabels = get_i_labels(icstree, newedges)
    
    for i in range(1, icstree.cards[0]):
        LDAG.add_node('I'+ str(i), color=icolor)
    
    for i in range(1, icstree.cards[0]):
        for level in newedges[i]:
            LDAG.add_edge(('I' + str(i)),(icstree.labels[level + 1]))

    ilabels = {}
    for i in range(1, icstree.cards[0]):
        for level in newedges[i]:
            ilabels[('I' + str(i), icstree.labels[level + 1])] = []
            for label in newedgelabels[i][level]:
                ilabels[('I' + str(i), icstree.labels[level + 1])] += [label]

    return [LDAG, ilabels]