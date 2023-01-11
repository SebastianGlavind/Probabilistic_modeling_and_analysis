import numpy as np

##################################################################################
# Subset implementation with adaptive conditional sampling (Papaioannou et al., 2015)
##################################################################################

# Adaptive conditional sampling algorithm
##################################################################################
def aCS(N, lam0, b0, z0, g_func, pa=0.1, std_opt=0):
    """
    ---------------------------------------------------------------------------
    Adaptive conditional sampling algorithm of Papaioannou et al. (2015)
    (https://www.cee.ed.tum.de/era/software/reliability/subset-simulation/)
    NB! Failure is defined as g <= 0 (thus negative values), as in the paper.
    ---------------------------------------------------------------------------
    Input:
    * N - subset samples (default 1000) [scalar]
    * lam0 - scaling parameter (e.g., 0.6; or from previous subset) [scalar]
    * b0 - intermediate threshold level (e.g., 10%-quantile) [scalar]
    * z0 - seeds used to generate the new samples (seeds corr. b0) [Nc x M array]
    * g_func - limit state function in z-space (limit state function) [func.]
    * pa - adaption ratio (default 0.1) [scaler]
    * std_opt - sigma options (0: constant 1; or 1: sample estimate) [0,1] 
    ---------------------------------------------------------------------------
    Output:
    * z1 - next level samples [N x M array]
    * g1 - limit state function evaluations of the new samples [N array]
    * lam1 - next scaling parameter lambda [scaler]
    * sigma - spread of the proposal [scalar]
    * acc - acceptance rate of the method [scaler]
    ---------------------------------------------------------------------------
    """
    # initialization
    Nc, M = z0.shape # chains/seeds/particles, dimensions (Nc x M)
    Ns = int(N//Nc) # number of samples per chain
    Na = int(pa*Nc) # chains per adaption (corr. to Nc/N-ratio) 
    z1 = np.zeros((N,M)) # generated samples (N x M)
    g1 = np.zeros(N) # store lsf evaluations
    acc_hat = np.zeros(int(Nc//Na)) # store acceptance
    acc_star = 0.44 # optimal acceptance rate 
    lam = lam0 # initial scaling parameter \in (0,1)
    
    # 1. compute the standard deviation
    if std_opt == 0: # 1a. # equal, unit variance
        sigma0 = np.ones(M)  
    else: # 1b. sample standard deviations
        sigma0 = np.std(z0, axis=0)

    # 2. permute the seeds
    z0_perm = z0[np.random.permutation(np.arange(Nc)),:]
    
    # 3. iteration
    # a. compute correlation parameter
    i = 0 # index for adaptation of lambda
    acc_mu_i = 0
    sigma = np.minimum(lam*sigma0, np.ones(M)) # Ref. 1 Eq. 23
    rho = np.sqrt(1-sigma**2) # Ref. 1 Eq. 24
    
    # b. apply conditional sampling
    for k in range(Nc):
        z1_k = np.zeros((Ns,M)) # (re-)samples of seed k
        z1_k[0,:] = z0_perm[k,:]
        g1_k = np.zeros(Ns) # function evaluations resamples of seed k   
        g1_k[0] = g_func(z1_k[0,:])
        acc_ik = 1 # accept the first one (zero in base code)
        
        for t in range(1, Ns):    
            # generate candidate sample  
            v = np.random.normal(loc=rho*z1_k[t-1,:], scale=sigma)
            
            # accept or reject sample              
            g_pro = g_func(v)
            if g_pro <= b0:
                z1_k[t,:] = v # accept the candidate in failure region            
                g1_k[t] = g_pro # store the func. evaluation
                acc_ik += 1 # update acceptance
            else:
                z1_k[t,:] = z1_k[t-1,:] # reject the candidate and use the same state
                g1_k[t] = g1_k[t-1] # store the func. evaluation    

        # updates
        acc_mu_i += acc_ik/Ns
        z1[k*Ns:(k+1)*Ns, :] = z1_k
        g1[k*Ns:(k+1)*Ns] = g1_k
        
        if np.mod(k+1,Na) == 0:
            # c. evaluate average acceptance rate
            acc_hat[i] = acc_mu_i/Na # Ref. 1 Eq. 25

            # d. compute new scaling parameter
            zeta = 1/np.sqrt(i+1)  # ensures that the variation of lambda(i) vanishes
            lam = np.exp(np.log(lam) + zeta*(acc_hat[i] - acc_star))  # Ref. 1 Eq. 26

            # update parameters
            sigma = np.minimum(lam*sigma0, np.ones(M)) # Ref. 1 Eq. 23
            rho = np.sqrt(1-sigma**2) # Ref. 1 Eq. 24

            # update counter and acceptance
            i += 1
            acc_mu_i = 0
       
    # next level lambda
    lam1 = lam
    # compute mean acceptance rate of all chains
    acc = np.mean(acc_hat)  
    
    return(z1, g1, lam1, sigma, acc)

# Estimate squared coefficient of variation for each subset level
#######################################################################################
def CoV2_subset(g1, b1, Nc, Ns):
    """
    ----------------------------------------------------------------------------------------------------
    Approximate, squared coefficient of variation for subset conditional probability, see e.g.,
    [1] Au & Beck (2001), Estimation of small failure probabilities in high dimensions by subset simulation
    [2] Zuev et al. (2012) - Bayesian Post-Processor and other Enhancements of Subset Simulation for 
    Estimating Failure Probabilities in High Dimensions
    ----------------------------------------------------------------------------------------------------
    Input:
    * g_val - limit state function evaluations (not sorted, i.e., grouped acc. to chains)
    * Nc - number of Markov chains (seeds)
    * Ns - number of samples simulated from each Markov chain (N/Nc)
    ----------------------------------------------------------------------------------------------------
    Output:
    * CoV - coeffiecient of variation for subset level
    ----------------------------------------------------------------------------------------------------
    """
    # Initialize variables
    N = Nc*Ns # total number of samples
    Ipf = np.reshape(g1 <= b1, (Ns,Nc)) # indicator function for the failure samples
    pf = np.sum(Ipf[:])/N # ~=p0, sample conditional probability
    
    # correlation at lag 0 (i=0)
    R0 = pf*(1-pf) # just above Eq.(26) in [1] (variance of Bernoulli random variable)

    # correlation for lags > 0 (i>0)
    gamma = 0
    for i in range(1, Ns): # lags (related to Ri)
        sums = 0
        for k in range(Nc): # chains/seeds/particles
            for ip in range(Ns-i): # samples
                sums += Ipf[ip,k]*Ipf[ip+i,k]  # sums inside (Eq. 22 in [2])
        Ri = sums/(N - i*Nc) - pf**2 # autocovariance at lag i (Eq. 22 in [2])
        gamma += (1 - i/Ns) * (Ri/R0) # correlation factor (Eq. 20 in [2])
    gamma *= 2 # small values --> higher efficiency  (Eq. 20 in [2]) 
    
    CoV2 = ((1-pf)/(N*pf))*(1+gamma) # (Eq. 19 in [2])
    return(CoV2)

# Subset simulation algorithm
#######################################################################################
def SuS_aCS(N, M, g_func, p0=0.1, pa=0.1, std_opt=0, progress=0):
    """
    -------------------------------------------------------------------------------------------------------
    Subset Simulation (standard Normal space) using adaptive conditional sampling (Papaioannou et al.,2015)
    (https://www.cee.ed.tum.de/era/software/reliability/subset-simulation/)
    NB! Failure is defined as g <= 0 (thus negative values), as in the paper.
    -------------------------------------------------------------------------------------------------------
    Input:
    * N - subset samples (N*p0 and 1/p0 must be positive integers) [scalar]
    * M - random variables/dimensions [scalar]
    * g_func - limit state function (possible transformations are conducted inside this object)
    * p0 - conditional probability of each subset (N*p0 and 1/p0 must be positive integers) [scalar]
    * pa - adaption ratio (default 0.1) [scaler]
    * std_opt - sigma options (0: constant 1; or 1: sample estimate) [0,1] 
    -------------------------------------------------------------------------------------------------------
    Output:
    * pf_SuS - subset failure probability estimate
    * CoV_SuS - coefficient of variation of subset estimate 
    * pflist - list of imtermediate failure probabilities
    * blist - list intermediate threshold levels 
    * glist - list of limit state function evaluations per level
    * Zlist - list of realizations of the random variables per level
    * lam - optimized scaling parameter [scalar]
    --------------------------------------------------------------------------------------------------------
    """
    # Initialization ####################
    j = 0 # initial conditional level
    Nc = int(N*p0) # number of markov chains/seeds/particles
    Ns = int(1/p0) # number of samples simulated from each Markov chain
    lam = 0.6 # initial scaling parameter (0.6 is recommended) 
    it_max = 50 # maximum number of iterations
    Zlist = [] # list of realizations (ordered; first Nc-values are the seeds)
    glist = [] # list of limit state evaluations (ordered; first Nc-values are the seeds)
    blist = [] # list of intermediate failure levels
    pflist = [] # list of failure probs
    CoV2list = [] # list of squard CoVs (called delta in the paper)

    # SuS procedure ######################
    # initial MCS stage
    z1 = np.random.normal(size=(N,M))
    g1 = np.array([g_func(z1[n,:]) for n in range(N)])

    # SuS stage ##########################
    while True:
        # sort values in ascending total and assign intermediate level
        ids = np.argsort(g1)
        glist.append(g1[ids]) # store the ordered values (first Nc-values are the seeds)
        Zlist.append(z1[ids,:]) # store the ordered samples (first Nc-values are the seeds)
        blist.append(np.percentile(glist[j], p0*100)) # intermediate level
        # blist.append(glist[j][Nc-1]) # intermediate level (item Nc; Au&Wang(2014; Eq. 5.7))
        # blist.append(sum(glist[j][Nc-1:Nc+1])/2) # intermediate level (ABC-Subset)
        if progress==1: print('\n Intermediate threshold level ', j, ' = ', blist[j])
        
        # assign conditional probability to the level
        if blist[j] <= 0:
            nF = sum(glist[j] <= 0)
            pflist.append(nF/N)
        else:
            pflist.append(p0)      
        
        # compute coefficient of variation
        if j == 0:
            CoV2list.append( (1-p0)/(N*p0) ) # CoV^2 for p(1): MCS (Eq. 8; Au&Wang(2014; Eq.2.98))
        else:
            CoV2list.append( CoV2_subset(g1, blist[j], Nc, Ns) ) # CoV^2 (Eq. 9-10)     
        
        # next level
        j += 1   
        
        # check for convergence
        if blist[-1] <= 0 or j > it_max:
            break
        else:
            # sampling process using adaptive conditional sampling
            z1, g1, lam, sigma, acc = aCS(N, lam, blist[-1], Zlist[-1][:Nc,:], g_func, pa, std_opt)
            if progress==1: print('\t*aCS lambda =', lam, '\t*aCS sigma =', sigma[0], '\t*aCS acc =', acc)
        
    # probability of failure
    pf_SuS = np.prod(pflist) # (Eq. 6)
    # coefficient of variation estimate
    # CoV_SuS = np.sqrt( np.sum(CoV2list) ) # (Eq. 12)
    CoVvector = np.sqrt( np.array(CoV2list) ) # vector of CoVs
    CoVouter = np.outer( CoVvector, CoVvector) # outer product of CoVs (Au&Wang,2014; Eq.5.18-19)
    CoV_SuS = [ np.sqrt( np.trace(CoVouter) ), np.sqrt( np.sum(CoVouter) ) ] # Lower and upper bound
    if progress==1: print('\t*Pf =', pf_SuS, '\t*CoV = [ ', CoV_SuS[0], '; ', CoV_SuS[1], ' ]')
    
    return(pf_SuS, CoV_SuS, pflist, blist, glist, Zlist, lam)