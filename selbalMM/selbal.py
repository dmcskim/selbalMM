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
#from core import cv_balance, select_balance, _build_balance
from numpy import var, mean, std, sqrt

class selbalMM(BaseEstimator, RegressorMixin, TransformerMixin):
    """ Selecting balances with mixed models

    Adaptation of ** for longitudinal data. Forward selection method
    to identify two groups of taxa whose longitudinal balance is associated 
    with a response variable of interest, adjusted for covariates.

    ** Rivera-Pinto et al. 2018 Balances: a new perspective for microbiome
    analysis https://msystems.asm.org/content/3/4/e00053-18

    Parameters
    ----------
    X : Independent variables and covariates
    Y : Dependent variable
    M : Compositional data
    group : Variable defining clusters
    cv : number of folds for GroupShuffleSplit (train_size = (cv-1)/cv)
    niter : number of iterations to run
    ntaxa : maximum number of taxa to consider
    """
    def __init__(self, group, cv=5, niter=100, ntaxa=20):
        self.cv_ = cv
        self.niter_ = niter
        self.ntaxa_ = ntaxa
        self.group_ = group

    def fit(self, X, Y, M):
        """ Assess model with cross-validation. Both X and Y must have
        the same indices.

        Parameters
        ----------
        X : Clinical data and covariates
        Y : Dependent variable of interest
        M : Microbiome abundance table
        """
        # check input data
        #X, Y = check_X_y(X, Y)
        # build own with check_array 
        # store classes
        #self.classes_ = unique_labels(Y[self.LHS_])

        self.X_ = X
        self.Y_ = Y
        self.M_ = M

        mses, tops, bots = cv_balance(self.X_, self.Y_,\
            self.M_, group=self.group_,\
            num_taxa=self.ntaxa_, nfolds=self.cv_, niter=self.niter_)
        
        tmse = {}
        all_mse = []
        for k,v in mses.items():
            tmse[k] = [mean(v), std(v)]
            all_mse += v

        #standard error for 1se rule
        tse = std(all_mse)/sqrt(len(all_mse))

        #1se rule for optimal number of taxa
        opt_n = 2
        prev = None
        for k,v in tmse.items():
            if prev != None:
                curr = prev - v[0]
                if curr < tse:
                    opt_n = k-1
                    break
                prev = v[0]
            else:
                prev = v[0]

        ### replace tops/bots with % inclusion ###
        self.onese_ = tse
        self.cv_mse_ = tmse
        self.ntaxa_ = opt_n
        self.cv_tops_ = tops
        self.cv_bots_ = bots

        #apply 1se
        #calculate taxa usage
        return self

    def transform(self):
        check_is_fitted(self, ['ntaxa_', 'cv_mse_', 'cv_tops_', 'cv_bots_'])
        #get optimum number of taxa, set to 16 for now
        #self.ntaxa_ = 16
        #create final balance
        ttop, tbot, tmse, tmodel = select_balance(self.X_,\
            self.Y_, self.M_, self.group_, self.ntaxa_)

        self.top = ttop[self.ntaxa_]
        self.bot = tbot[self.ntaxa_]
        self.mse = tmse[self.ntaxa_]
        self.model = tmodel[self.ntaxa_]
        temp_Y = _build_balance(self.top, self.bot, self.X_, self.M_)
        self.balance = temp_Y
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

