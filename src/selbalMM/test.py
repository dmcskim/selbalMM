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
import pickle
from multiprocessing import Process
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from itertools import permutations
import sys
from ..src.selbalMM.selbalMM import selbalMM

def amalgamate(amal_data, tax, tax_lvl='Full'):
    ''' Amalgamate ASV table into annotation taxonomy. '''
    amal_data.columns = [tax.loc[x, tax_lvl] for x in amal_data.columns]

    #print(len(set(amal_data.columns)))
    genera = list(set(tax[tax_lvl]))
    namal = []
    ncols = []

    for x in genera:
        ttemp = amal_data.loc[:, x]

        if len(ttemp.shape) > 1:
            namal.append(ttemp.sum(axis=1).values)
        else:
            namal.append(ttemp)
        ncols.append(x)
    #print(len(genera), len(namal), len(ncols), len(ttemp.index), len(amal_data.index))
    namal = np.array(namal)
    gdata = pd.DataFrame(namal.T, index=ttemp.index, columns=ncols)

    #print(gdata.shape)
    return gdata

if __name__ == '__main__':
    ## load taxonomy data
    #ndata = pd.read_csv('anujit/nicu561.ASVs_counts.tsv', index_col=0, sep='\t').T
    bdir = '/home/dmcskimming/Dropbox/projects/maureen/neurodevelopment/anujit/'
    tax = pd.read_csv(bdir+'nicu561.ASVs_taxonomy.tsv', index_col=0, sep='\t').fillna('')
    tax['Full'] = tax['Kingdom'] + '; ' + tax['Phylum'] + '; ' + tax['Class'] + '; ' + tax['Order'] +\
    '; ' + tax['Family'] + '; ' + tax['Genus'] #+ '; ' + tax['Species']
    tax['Phylum'] = tax['Kingdom'] + '; ' + tax['Phylum']
    tax['Class'] = tax['Phylum'] + '; ' + tax['Class']
    tax['Order'] = tax['Class'] + '; ' + tax['Order']
    #print(ndata.shape)
    save_results = False

    ## load BDI/ asv data
    ## get better BDI data
    merged_jean = pd.read_excel(bdir+'Neuro_nicu_bat_cbcl_combined_data.xlsx')
    bdi_cols = [x for x in merged_jean.columns if x[:3] == 'BDI']

    ## lets look at raw scores
    #bdi_cols = [x for x in bdi_cols if 'tscore' not in x and 'percentile' not in x]
    #print(bdi_cols)
    ## just a few variables at first
    bdi_temp = merged_jean[['ID', 'TIMEPOINT', 'GESTAGE'] + bdi_cols].dropna()
    #bdi_temp['TIMEPOINT'] = [x[:2] for x in bdi_temp['TIMEPOINT']]
    bdi_temp['TIME'] = [int(x[0]) for x in bdi_temp['TIMEPOINT']]
    #bdi_temp['ID'] = bdi_temp['ID'].astype(str)
    #bdi_temp.dropna(inplace=True)
    #display(bdi_temp)

    ## t_ind = samples with all data (BDI_total)
    t_ind = bdi_temp.index
    #t_ind = [x for x in bdi_temp.index if bdi_temp.loc[x, 'ID'] not in [1,21,26,28,49,70]]
    #t2_ind = [x for x in bdi_temp.index if x not in t_ind]
    #display(bdi_temp.loc[t_ind])
    #display(bdi_temp.loc[t2_ind])

    ## asv abundance data
    asv_cols = [x for x in merged_jean.columns if x[:3] == 'ASV']
    #print(bdi_temp.index)
    temp_taxa = merged_jean.loc[t_ind, asv_cols]
    temp_taxa.dropna(inplace=True)
    t_ind = list(temp_taxa.index)
    #print(temp_taxa.shape)
    #display(merged_jean.loc[9,asv_cols])

    gdata = amalgamate(temp_taxa, tax, 'Full')
    print('post-amal shape:', gdata.shape)

    ## remove taxa found in < 10% of samples
    tsum = np.count_nonzero(gdata, axis=0) #.sum()
    nsamps = 40
    #print(tsum)
    k_ind = []
    for a,b in zip(gdata.columns, tsum):
        if b >= .1*nsamps:
            k_ind.append(a)
    #print(len(k_ind))

    ndata = gdata[k_ind]
    print('post-filter shape:', ndata.shape)
    #display(ndata.sample(10))


    ## build wdata for balance detection (same index as ndata needed)
    bdata = bdi_temp.loc[t_ind]
    working_data = bdata.copy()
    wdata = working_data.loc[working_data.index.intersection(ndata.index),:]

    selbal = selbalMM('BDI_cog', 'TIME + GESTAGE', 'ID', niter=20)
    selbal.fit(ndata+1, wdata)
    selbal.transform()

    ### try centered version, any difference? (no interaction terms yet)
    wwdata = wdata.copy()
    wwdata['TIME'] = [x-3 for x in wwdata['TIME']]
    gestage_mean = wwdata['GESTAGE'].mean()
    wwdata['GESTAGE'] = [x-gestage_mean for x in wwdata['GESTAGE']]

    selbal2 = selbalMM('BDI_cog', 'TIME + GESTAGE', 'ID', niter=20)
    selbal2.fit(ndata+1, wwdata)
    selbal2.transform()

    #res_mse, test_samps, tops, bots, models = cv_balance(wdata, ndata+1,\
                            #LHS=depvar, nfolds=5, niter=20, num_taxa=20)
    #results = {'mse':res_mse, 'test_groups':test_samps, 'tops':tops,\
                #'bots':bots, 'models':models}
    #pickle.dump(results, open('prod_cv_5_20_20_{0}.p'.format(depvar), 'wb'))

