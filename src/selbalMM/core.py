"""
Balance selection on mixed effect models.
Daniel Ian McSkimming, PhD
dmcskimming@usf.edu
University of South Florida
"""

### TODO: break out balance construction, model construction, model testing
###       into separate functions

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import dirichlet
import pickle
from multiprocessing import Process
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from itertools import permutations
import sys

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
    common = np.sqrt((tcount*bcount)/(tcount + bcount))
    return common, 1/tcount, -1/bcount

def _build_balance(top, bot, wdata, ndata):
    kc, kt, kb = _get_coefs(len(top), len(bot))
    cdata = wdata.copy()
    cdata.loc[:, 'top'] = [kt*np.log2(x) for x in ndata[top].product(axis=1)]
    cdata.loc[:, 'bot'] = [kb*np.log2(x) for x in ndata[bot].product(axis=1)]
    #minus for bot taken care of with _get_coefs
    cdata.loc[:, 'balance'] = kc*(cdata.loc[:, 'top'] + cdata.loc[:, 'bot'])
    return cdata

def _initial_balance(wdata, ndata, LHS, RHS, group, test=None):
    ## build and check all two part balances
    top, bot = [], []
    results = []

    #assert(np.all(wdata.index == ndata.index))
    best_mse = 10000
    best_model = None
    ti = 0
    #print(len(ndata.columns))
    for a,b in permutations(ndata.columns, 2):
        #print(a,b)
        ti += 1
        ttop = [a]
        tbot = [b]
        #print(ndata[ttop])
        #wdata.loc[:, 'top'] = [np.sqrt(1/2)*np.log2(x) for x in ndata[a]]
        #wdata.loc[:, 'bot'] = [-np.sqrt(1/2)*np.log2(x) for x in ndata[b]]
        #wdata.loc[:, 'balance'] = wdata['top'] - wdata['bot']
        bdata = _build_balance([a], [b], wdata, ndata)
        #print(bdata.columns)
        #display(bdata)
        #md = smf.mixedlm("BDI_total ~ balance", data=wdata, groups=wdata['ID'])#, re_formula="~TIME")
        #print(LHS, ' + '.join(RHS))
        md = smf.mixedlm("{0} ~ {1}".format(LHS, RHS), data=bdata,\
                         groups=bdata[group])
        try:
            mdf = md.fit()
            bpred = mdf.fittedvalues 
            if test is not None:
                twtdata, tntdata = test
                bdata = _build_balance([a], [b], twtdata, tntdata) 
                #bpred = mdf.predict(bdata[RHS+['ID']])
                bpred = mdf.predict(bdata)
            cmse = _mse(bdata[LHS], bpred)
            results.append([ttop, tbot, mdf.pvalues, cmse])
            if cmse < best_mse:
                top = ttop
                bot = tbot
                best_mse = cmse
                best_model = [bdata['balance'].copy(), md, mdf] #[md,mdf]
        #except np.linalg.LinAlgError as error:
        except (ValueError, np.linalg.LinAlgError) as error:
            continue
    return top, bot, best_mse, best_model

def _add_balance(top, bot, wdata, ndata, LHS, RHS, group, test=None):
    ## add a new taxon to balance
    results = []
    best_mse = 10000
    best_model = None
    ttop, tbot = [],[]
    ccols = list(ndata.columns)
    for x in top+bot:
        ccols.remove(x)
    #ccols.remove([x for x in top])
    #ccols.remove([x for x in bot])

    for c in ccols:
        # add c to top and build balance
        ctop = top + [c]
        cbot = bot
        bdata = _build_balance(ctop, cbot, wdata, ndata)
        try:
            md = smf.mixedlm("{0} ~ {1}".format(LHS, RHS), data=bdata, groups=bdata[group])
            mdf = md.fit()
            bpred = mdf.fittedvalues 

            if test is not None:
                twtdata, tntdata = test
                bdata = _build_balance(ctop, cbot, twtdata, tntdata) 
                #bpred = mdf.predict(bdata[RHS+['ID']])
                bpred = mdf.predict(bdata)
            cmse = _mse(bdata[LHS], bpred)
            
            #display(wdata)
            results.append([ctop, cbot, mdf.pvalues, cmse])
            if cmse < best_mse:
                #top3 = ttop
                #bot3 = tbot
                best_mse = cmse
                best_model = [bdata['balance'].copy(), md, mdf]
                ttop = ctop
                tbot = cbot
        #except ValueError, as err:
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
                #bpred = mdf.predict(bdata[RHS+['ID']])
                bpred = mdf.predict(bdata)
            cmse = _mse(bdata[LHS], bpred)

            results.append([ctop, cbot, mdf.pvalues, cmse])
            if cmse < best_mse:
                best_mse = cmse
                best_model = [bdata['balance'].copy(), md, mdf]
                ttop = ctop
                tbot = cbot
        except (ValueError, np.linalg.LinAlgError) as error:
        #except ValueError as err:
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
    
    rtop, rbot = [], []
    
    ntax = len(top) + len(bot)
    
    while len(top) + len(bot) < num_taxa:
        #print(num_taxa, len(top), len(bot))
        top, bot, cmse, cmodel = _add_balance(top, bot, wdata, ndata, LHS, RHS, group, test)
        #print('\t',top, bot, cmse)
        #print(cmodel[1].summary())
        
        ntax = len(top) + len(bot)
        #rtop[ntax] = top
        #rbot[ntax] = bot
        #mse[ntax] = cmse
        #rmodel[ntax] = cmodel
        rtop += top
        rbot += bot
        mse = cmse
        rmodel = cmodel

    return rtop, rbot, mse, rmodel

def cv_balance(wdata, ndata, LHS, RHS, group, num_taxa=20, nfolds=5, niter=16):
    ## goal: identify optimal number of taxa using 1se, explore robustness
    res_mse = defaultdict(list)
    #tests = []

    # *** here is where we should parallelize ***
    #tdata = []
    for titer in range(niter):
        temp = [dirichlet.rvs(ndata.loc[x,:])[0] for x in ndata.index]
        tdata = pd.DataFrame(temp, index=ndata.index,\
                                 columns=ndata.columns)
        #tdata.append(temp_data.copy())
        #shuffle wdata
        # instead of shuffle, build/use dirichlet distribution
        w2data = wdata
        n2data = tdata #ndata.loc[w2data.index, :]
        #n2data = ndata.loc[ndata.index.intersection(w2data.index), :]
        tmse = _cv_balance(wdata, ndata, LHS, RHS, group, num_taxa, nfolds)
        for k,v in tmse.items():
            res_mse[k].extend(v)
    # only need mse to assess number of taxa needed
    return res_mse #, tests, tops, bots, models

def _cv_balance(wdata, ndata, LHS, RHS, group, num_taxa, nfolds):
    #a little error checking
    assert(np.all(w2data.index == n2data.index))
    assert(~np.any(np.isnan(n2data)))

    res_mse = defaultdict(list)

    group_kfold = GroupKFold(n_splits=nfolds)
    for train_ind, test_ind in group_kfold.split(w2data, groups=w2data[group]):
        wtrain = w2data.iloc[train_ind]
        ntrain = n2data.iloc[train_ind]
        wtest = w2data.iloc[test_ind]
        ntest = n2data.iloc[test_ind]

        #tests.append(set(wtest.loc[:, group].values))

        tops, bots, mses, models = select_balance(wtrain, ntrain, LHS=LHS,\
            RHS=RHS, group=group, num_taxa=num_taxa, test=[wtest, ntest])

        for i in mses.keys():
            res_mse[i].append(mses[i])
    return res_mse #, tests, tops, bots, models

