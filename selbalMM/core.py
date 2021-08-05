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
from scipy.stats import dirichlet
import pickle
from multiprocessing import Process
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from itertools import permutations
import sys
from tqdm import tqdm

# Not the best option, but many models will not converge
#  or have other issues. Especially in early stages (eg, _initial_balance).
if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')

def _mse(real, predicted):
    error = real - predicted
    error = np.power(error, 2)
    se = np.sum(error)
    mse = se / len(error)
    return mse

def _get_coefs(tcount, bcount):
    ### Returns coefficients for the balance ###
    common = np.sqrt((tcount*bcount)/(tcount + bcount))
    return common, 1/tcount, -1/bcount

def _build_balance(top, bot, wdata, ndata):
    ### Builds balance given membership for top, bottom.###
    kc, kt, kb = _get_coefs(len(top), len(bot))
    n = wdata.shape[0]
    top = [kt*np.log2(x) for x in ndata[top].product(axis=1)]
    bot = [kb*np.log2(x) for x in ndata[bot].product(axis=1)]

    #minus for bot taken care of with _get_coefs
    balance = kc*(top+bot)
    return np.column_stack(np.ones(n), copy(wdata), balance)

def _initial_balance(x, y, m, group, test=None):
    ## build and check all two part balances
    top, bot = [], []
    results = []

    best_mse = 10000
    best_model = None
    ti = 0

    #considers all pairwise taxa, could do something clever and move 
    # to combinations to shorten the loop. Won't reduce number of models
    # in need of fitting.
    for a,b in permutations(m.columns, 2):
        ti += 1
        ttop = [a]
        tbot = [b]
        bdata = _build_balance([a], [b], x, m)

        #***REDO***
        md = smf.mixedlm("{0} ~ {1}".format(LHS, RHS), data=bdata,\
                         groups=bdata[group])
        try:
            mdf = md.fit()
            bpred = mdf.fittedvalues 
            if test is not None:
                twtdata, tntdata = test
                bdata = _build_balance([a], [b], twtdata, tntdata) 
                bpred = mdf.predict(bdata)
            cmse = _mse(bdata[LHS], bpred)
            results.append([ttop, tbot, mdf.pvalues, cmse])
            if cmse < best_mse:
                top = ttop
                bot = tbot
                best_mse = cmse
                best_model = [bdata['balance'].copy(), md, mdf] #[md,mdf]
        #consider adding logging, better exception handling
        except (ValueError, np.linalg.LinAlgError) as error:
            continue

    return top, bot, best_mse, best_model

def _add_balance(top, bot, x, y, m, group, test=None):
    ## add a new taxon to balance
    results = []
    best_mse = 10000
    best_model = None
    ttop, tbot = [],[]
    ccols = list(ndata.columns)
    for x in top+bot:
        ccols.remove(x)

    for c in ccols:
        # add c to top and build balance
        ctop = top + [c]
        cbot = bot
        bdata = _build_balance(ctop, cbot, x, m)

    #***REDO***
        try:
            md = smf.mixedlm("{0} ~ {1}".format(LHS, RHS), data=bdata, groups=bdata[group])
            mdf = md.fit()
            bpred = mdf.fittedvalues 

            if test is not None:
                twtdata, tntdata = test
                bdata = _build_balance(ctop, cbot, twtdata, tntdata) 
                bpred = mdf.predict(bdata)
            cmse = _mse(bdata[LHS], bpred)
            
            results.append([ctop, cbot, mdf.pvalues, cmse])
            if cmse < best_mse:
                best_mse = cmse
                best_model = [bdata['balance'].copy(), md, mdf]
                ttop = ctop
                tbot = cbot
        except (ValueError, np.linalg.LinAlgError) as error:
            pass

        # add c to bot and build balance
        ctop = top
        cbot = bot + [c]
        bdata = _build_balance(ctop, cbot, wdata, ndata)
        try:
            md = smf.mixedlm("{0} ~ {1}".format(LHS, RHS), data=bdata, groups=bdata[group])
            mdf = md.fit()
            bpred = mdf.fittedvalues 

            if test is not None:
                twtdata, tntdata = test
                bdata = _build_balance(ctop, cbot, twtdata, tntdata) 
                bpred = mdf.predict(bdata)
            cmse = _mse(bdata[LHS], bpred)

            results.append([ctop, cbot, mdf.pvalues, cmse])
            if cmse < best_mse:
                best_mse = cmse
                best_model = [bdata['balance'].copy(), md, mdf]
                ttop = ctop
                tbot = cbot
        except (ValueError, np.linalg.LinAlgError) as error:
            pass
    return ttop, tbot, best_mse, best_model

def select_balance(wdata, ndata, LHS, RHS, group, num_taxa, test=None):
    top, bot, initial_mse, initial_model = _initial_balance(wdata, ndata, LHS, RHS, group, test)
    #print(top, bot, initial_mse)
    #for x in top:
        #print(x, x in ndata.columns)
    #for x in bot:
        #print(x, x in ndata.columns)
    #print(initial_model[1].summary())

    rtop, rbot, mse, rmodel = {}, {}, {}, {}

    while len(top) + len(bot) < np.min([num_taxa, ndata.shape[1]]):
        top, bot, cmse, cmodel = _add_balance(top, bot, wdata, ndata, LHS, RHS, group, test)

        ntax = len(top) + len(bot)
        rtop[ntax] = top
        rbot[ntax] = bot
        mse[ntax] = cmse
        rmodel[ntax] = cmodel

    return rtop, rbot, mse, rmodel

def cv_balance(wdata, ndata, LHS, RHS, group, num_taxa=20, nfolds=5, niter=100):
    ## goal: identify optimal number of taxa using 1se, explore robustness
    res_mse = defaultdict(list)
    res_tops = defaultdict(list)
    res_bots = defaultdict(list)

    # *** here is where we should parallelize ***
    dsum = ndata.sum(axis=1)
    for titer in tqdm(range(niter), desc='Iteration'):
        #take dirichlet sample for each row
        temp = [dirichlet.rvs(ndata.loc[x,:])[0] for x in ndata.index]
        tdata = pd.DataFrame(temp, index=ndata.index,\
                                 columns=ndata.columns)
        #multiply by the row total
        tdata = tdata.multiply(dsum, axis=0)

        #find cross-validated balances
        tmse, tops, bots = _cv_balance(wdata, tdata, LHS, RHS, group, num_taxa, nfolds)
        for k,v in tmse.items():
            res_mse[k].extend(v)
            res_tops[k].append(tops[k])
            res_bots[k].append(bots[k])
    # mse for calculating number of taxa, tops/bots for frequency used
    return res_mse, res_tops, res_bots # tests, tops, bots, models

def _cv_balance(wdata, ndata, LHS, RHS, group, num_taxa, nfolds):
    #a little error checking
    assert(np.all(wdata.index == ndata.index))
    assert(~np.any(np.isnan(ndata)))

    res_mse = defaultdict(list)
    res_tops = defaultdict(list)
    res_bots = defaultdict(list)

    group_kfold = GroupKFold(n_splits=nfolds)
    for train_ind, test_ind in group_kfold.split(wdata, groups=wdata[group]):
        wtrain = wdata.iloc[train_ind]
        ntrain = ndata.iloc[train_ind]
        wtest = wdata.iloc[test_ind]
        ntest = ndata.iloc[test_ind]

        tops, bots, mses, models = select_balance(wtrain, ntrain, LHS=LHS,\
            RHS=RHS, group=group, num_taxa=num_taxa, test=[wtest, ntest])
        for k,v in mses.items():
            res_mse[k].append(v)
            res_tops[k].append(tops[k])
            res_bots[k].append(bots[k])
    return res_mse, res_tops, res_bots #, tests, tops, bots, models

