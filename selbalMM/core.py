"""
Balance selection on mixed effect models.
Daniel Ian McSkimming, PhD
dmcskimming@usf.edu
University of South Florida
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
#from scipy.stats import dirichlet
#import pickle
from multiprocessing import Process
#from collections import defaultdict
from sklearn.model_selection import GroupKFold
from itertools import permutations
import sys
from tqdm import tqdm
from copy import copy
import gpboost as gp
from numba import jit
import logging

# Not the best option, but many models will not converge
#  or have other issues. Especially in early stages (eg, _initial_balance).
if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')

@jit(nopython=True, parallel=True)
def _groupKFold(groups, k=5):
    nsamp = len(groups)
    grpSet = groups.unique()
    assert(len(grpSet) < k)
    np.random.shuffle(grpSet)
    assignments = {grp:i%k for i,grp in enumerate(grpSet)}
    grps = [assignments[x] for x in groups]
    test_groups = [[i for i,grp in enumerate(groups) if grps[i] == k] for k in\
                   np.arange(k)]
    all_groups = [([x for x in np.arange(nsamp) if x not in ctest], ctest) for ctest in test_groups]
    return all_groups

@jit(nopython=True)
def _dirichlet(alpha):
    alpha = alpha/np.sum(alpha)
    samp = np.array([np.random.gamma(a, 1) for a in alpha])
    samp = [v / np.sum(samp) for v in samp]
    assert(len(samp) == len(alpha))
    return samp

@jit(nopython=True)
def _mse(real, predicted):
    error = real - predicted
    error = np.power(error, 2)
    se = np.sum(error)
    mse = se / len(error)
    return mse

@jit(nopython=True)
def _get_coefs(tcount, bcount):
    ### Returns coefficients for the balance ###
    common = np.sqrt((tcount*bcount)/(tcount + bcount))
    return common, 1/tcount, -1/bcount

@jit(nopython=True)
def _build_balance(top, bot, wdata, ndata):
    ### Builds balance given membership for top, bottom.
    #    wdata: contains independent variables and covariates
    #    ndata: contains microbiome data
    ###
    #print(top, bot, wdata.shape, ndata.shape)
    kc, kt, kb = _get_coefs(len(top), len(bot))
    n = wdata.shape[0]
    tp = [kt*np.log2(x) for x in ndata[:, top].prod(axis=1)]
    bt = [kb*np.log2(x) for x in ndata[:, bot].prod(axis=1)]

    #minus for bot taken care of with _get_coefs
    balance = np.add(tp, bt)

    #return intercept, covariates, and balance as one array
    return np.concatenate((np.ones(n)[:, None], copy(wdata), balance[:, None]), axis=1)

#@jit(nopython=True)
@jit(parallel=True)
def _initial_balance(x, y, m, group, test=None):
    ## build and check all two part balances
    top, bot = [], []
    results = []

    best_mse = 10000
    cmse = 0
    best_model = None
    ti = 0

    #considers all pairwise taxa, could do something clever and move 
    # to combinations to shorten the loop. Won't reduce number of models
    # in need of fitting.
    #print('building initial balance')
    for a,b in permutations(range(m.shape[1]), 2):
        ti += 1
        ttop = [a]
        tbot = [b]
        bdata = _build_balance([a], [b], x, m)

        try:
            gpmod = gp.GPModel(group_data=group, likelihood='gaussian')
            gpmod.fit(y=y, X=bdata, params={"std_dev":True})
            #mdf = md.fit()
            btmp = gpmod.predict(group_data_pred=group,\
                                 X_pred=bdata, predict_var=True)
            bpred = btmp['mu']
            tty = y

            if test is not None:
                tx, ty, tm, tg = test
                bdata = _build_balance([a], [b], tx, tm) 
                btmp = gpmod.predict(group_data_pred=tg,\
                                     X_pred=bdata, predict_var=True)
                bpred = btmp['mu']
                tty = ty

            cmse = _mse(tty, bpred)
            results.append([ttop, tbot, cmse])
            if cmse < best_mse:
                top = ttop
                bot = tbot
                best_mse = cmse
                best_model = [bdata.copy(), gpmod] #[md,mdf]
        #consider adding logging, better exception handling
        except:
            #print(error)
            pass
    #print(top, bot, best_mse)
    return top, bot, best_mse, best_model

#@jit(nopython=True)
@jit(parallel=True)
def _add_balance(top, bot, x, y, m, group, test=None):
    ## add a new taxon to balance
    #results = []
    best_mse = 10000
    best_model = None
    ttop, tbot = [],[]

    #don't consider elements already in the balance
    ccols = list(set(range(m.shape[1])) - set(top+bot))

    #print(top, bot, 'first add balance')
    for c in ccols:
        # add c to top and build balance
        ctop = top + [c]
        cbot = bot
        bdata = _build_balance(ctop, cbot, x, m)

        try:
            gpmod = gp.GPModel(group_data=group, likelihood='gaussian')
            gpmod.fit(y=y, X=bdata, params={"std_dev":True})
            btmp = gpmod.predict(group_data_pred=group,\
                                 X_pred=bdata, predict_var=True)
            bpred = btmp['mu']
            tty = y

            if test is not None:
                tx, ty, tm, tg = test
                tty = ty
                #bdata = _build_balance([a], [b], tx, tm) 
                bdata = _build_balance(ctop, cbot, tx, tm)
                btmp = gpmod.predict(group_data_pred=tg,\
                                     X_pred=bdata, predict_var=True)
                bpred = btmp['mu']

            cmse = _mse(tty, bpred)
            
            #results.append([ctop, cbot, cmse])
            if cmse < best_mse:
                best_mse = cmse
                best_model = [bdata.copy(), gpmod]
                ttop = ctop
                tbot = cbot
        #except (ValueError, np.linalg.LinAlgError, gp.basic.GPBoostError) as error:
        except:
            continue
            #pass

        # add c to bot and build balance
        ctop = top
        cbot = bot + [c]
        bdata = _build_balance(ctop, cbot, x, m)
        try:
            gpmod = gp.GPModel(group_data=group, likelihood='gaussian')
            gpmod.fit(y=y, X=bdata, params={"std_dev":True})
            #md = smf.mixedlm("{0} ~ {1}".format(LHS, RHS), data=bdata, groups=bdata[group])
            #mdf = md.fit()
            btmp = gpmod.predict(group_data_pred=group,\
                                 X_pred=bdata, predict_var=True)
            bpred = btmp['mu']
            tty = y

            if test is not None:
                tx, ty, tm, tg = test
                tty = ty
                #bdata = _build_balance([a], [b], tx, tm) 
                bdata = _build_balance(ctop, cbot, tx, tm)
                btmp = gpmod.predict(group_data_pred=tg,\
                                     X_pred=bdata, predict_var=True)
                bpred = btmp['mu']

            cmse = _mse(tty, bpred)

            #results.append([ctop, cbot, cmse])
            if cmse < best_mse:
                best_mse = cmse
                best_model = [bdata.copy(), gpmod]
                ttop = ctop
                tbot = cbot
        #except (ValueError, np.linalg.LinAlgError) as error:
        #except (ValueError, np.linalg.LinAlgError, gp.basic.GPBoostError) as error:
        except:
            continue
            #pass
    return ttop, tbot, best_mse, best_model

#@jit(parallel=True)
@jit(nopython=True, parallel=True)
def select_balance(x, y, m, group, num_taxa, test=None):
    #build initial balance
    #print('building initial balance')
    top, bot, initial_mse, initial_model = _initial_balance(x, y, m, group, test)
    #print('***', top, bot, initial_mse)

    rtop, rbot, mse, rmodel = {2:top}, {2:bot}, {2:initial_mse}, {2:initial_model}

    #add to balance as needed
    while len(top) + len(bot) < np.min([num_taxa, m.shape[1]]):
        top, bot, cmse, cmodel = _add_balance(top, bot, x, y, m, group, test)
        #print(len(top), len(bot), cmse, np.min([num_taxa, m.shape[1]]))

        ntax = len(top) + len(bot)
        rtop[ntax] = top
        rbot[ntax] = bot
        mse[ntax] = cmse
        rmodel[ntax] = cmodel

    return rtop, rbot, mse, rmodel

@jit(nopython=True, parallel=True)
def cv_balance(x, y, m, group, num_taxa=20, nfolds=5, niter=100):
    ## goal: identify optimal number of taxa using 1se, explore robustness
    #res_mse = defaultdict(list)
    #res_tops = defaultdict(list)
    #res_bots = defaultdict(list)
    res_mse = {}
    res_tops = {}
    res_bots = {}
    dsum = m.sum(axis=1)
    #for titer in tqdm(range(niter)):

    def _mmult(a, b):
        return [a*x for x in b]

    for titer in range(niter):
        #print('next iteration')
        #take dirichlet sample for each row
        temp = [_dirichlet(m[tx,:]) for tx in range(m.shape[0])]
        #tdat = np.array(temp)
        #tdata = pd.DataFrame(temp, index=ndata.index,\
                                 #columns=ndata.columns)
        #multiply by the row total
        tdata = [_mmult(dsum[i], temp[i]) for i in range(len(dsum))]
        tdata = np.array(tdata)
        #tdata = ttmp * tdat
        #tdata = np.multiply(tdata.T, dsum).T

        #find cross-validated balances
        tmse, tops, bots = _cv_balance(x, y, tdata, group, num_taxa, nfolds)
        for k,v in tmse.items():
            if k not in res_mse:
                res_mse[k] = []
            res_mse[k].extend(v)

            if k not in res_tops:
                res_tops[k] = []
            res_tops[k].append(tops[k])

            if k not in res_bots:
                res_bots[k] = []
            res_bots[k].append(bots[k])
    # mse for calculating number of taxa, tops/bots for frequency used
    return res_mse, res_tops, res_bots # tests, tops, bots, models

@jit(nopython=True, parallel=True)
def _cv_balance(x, y, m, group, num_taxa, nfolds):
    #a little error checking
    #assert(np.all(wdata.index == ndata.index))
    assert(x.shape[0] == y.shape[0])
    assert(x.shape[0] == m.shape[0])
    assert(~np.any(np.isnan(m)))

    #res_mse = defaultdict(list)
    #res_tops = defaultdict(list)
    #res_bots = defaultdict(list)
    res_mse = {}
    res_tops = {}
    res_bots = {}

    #group_kfold = GroupKFold(n_splits=nfolds)
    #print(x.shape, group)
    for train_ind, test_ind in _groupKFold.split(groups=group):
        xtrain = x[train_ind, :]
        mtrain = m[train_ind, :]
        ytrain = y[train_ind]
        gtrain = group[train_ind]

        xtest = x[test_ind, :]
        mtest = m[test_ind, :]
        ytest = y[test_ind]
        gtest = group[test_ind]

        tops, bots, mses, models = select_balance(xtrain, ytrain, mtrain,\
                group=gtrain, num_taxa=num_taxa, test=[xtest, ytest, mtest,\
                                                       gtest])
        for k,v in mses.items():
            if k not in res_mse:
                res_mse[k] = []
            res_mse[k].append(v)

            if k not in res_tops:
                res_tops[k] = []
            res_tops[k].append(tops[k])

            if k not in res_bots:
                res_bots[k] = []
            res_bots[k].append(bots[k])
            #res_mse[k].append(v)
            #res_tops[k].append(tops[k])
            #res_bots[k].append(bots[k])
    return res_mse, res_tops, res_bots #, tests, tops, bots, models

