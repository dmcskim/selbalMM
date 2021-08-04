"""
Balance selection on mixed effect models.
Daniel Ian McSkimming, PhD
dmcskimming@usf.edu
University of South Florida
"""

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from .core import cv_balance, select_balance, _build_balance
from numpy import var

class selbalMM(BaseEstimator, RegressorMixin, TransformerMixin):
    """ Selecting balances with mixed models

    Adaptation of ** for longitudinal data. Forward selection method
    to identify two groups of taxa whose longitudinal balance is associated 
    with a response variable of interest, adjusted for covariates.

    ** Rivera-Pinto et al. 2018 Balances: a new perspective for microbiome
    analysis https://msystems.asm.org/content/3/4/e00053-18

    Parameters
    ----------
    LHS : Left Hand Side of mixed model
    RHS : Right Hand Side of mixed model (interactions, covariates)
    group : Variable defining clusters
    cv : number of folds for GroupShuffleSplit (train_size = (cv-1)/cv)
    ncores : number of processes to run iterations on
    niter : number of iterations to run
    ntaxa : maximum number of taxa to consider
    """
    def __init__(self, LHS, RHS, group, cv=5, ncores=4, niter=20, ntaxa=20):
        self.cv_ = cv
        self.cores_ = ncores
        self.niter_ = niter
        self.ntaxa_ = ntaxa
        self.LHS_ = LHS
        if 'balance' not in RHS:
            RHS = 'balance + ' + RHS
        self.RHS_ = RHS
        self.group_ = group

    def fit(self, X, Y):
        """ Assess model with cross-validation. Both X and Y must have
        the same indices.

        Parameters
        ----------
        X : Microbiome abundance counts
        Y : Clinical data and covariates
        """
        # check input data
        #X, Y = check_X_y(X, Y)
        # build own with check_array 
        # store classes
        #self.classes_ = unique_labels(Y[self.LHS_])

        self.X_ = X
        self.Y_ = Y

        mses = cv_balance(self.Y_, self.X_,\
            LHS=self.LHS_, RHS=self.RHS_, group=self.group_,\
            num_taxa=self.ntaxa_, nfolds=self.cv_, niter=self.niter_)

        self.cv_mse = mses
        return self

    def transform(self):
        check_is_fitted(self)
        #get optimum number of taxa, set to 16 for now
        self.ntaxa_ = 16
        #create final balance
        ttop, tbot, tmse, tmodel = select_balance(self.Y_,\
            self.X_, self.LHS_, self.RHS_, self.group_, self.ntaxa_)

        self.top = ttop[self.ntaxa_]
        self.bot = tbot[self.ntaxa_]
        self.mse = tmse[self.ntaxa_]
        self.model = tmodel[self.ntaxa_]
        temp_Y = _build_balance(self.top, self.bot, self.Y_,\
                                            self.X_)
        self.Y_ = temp_Y
        return self

    def marg_cond_r2(self):
        ### Marginal and conditional R2, based on Nakagawa & Schielzeth (2014)
        ###
        check_is_fitted(self)

        feparams = mod2.fe_params
        fekeys = list(feparams.keys())
        fekeys.remove('Intercept')
        tdata = self.Y_[fekeys]
        tdata['Intercept'] = [1 for x in tdata.index]
        temp = feparams.mul(tdata)
        sigma_fe = var(temp)

        sigma_re = self.model[-1][0]
        sigma_e = mod2.scale

        denom = sigma_fe + sigma_re + sigma_e
        return sigma_fe/denom, (sigma_fe + sigma_re)/denom

