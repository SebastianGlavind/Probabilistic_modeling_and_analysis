import numpy as np

#############################################################################################################
# Kalman filters and smoothers
#############################################################################################################

# Kalman Filter - Time invariant model without inputs
# See Shumway & Stoffer (2017), ch. 6.
# https://rdrr.io/cran/astsa/src/R/Kfilter0.R
def myKalmanFilter0(y, # data np.array
                    cQ, cR, 
                    mu0 = np.asarray([30,20]), #first estimate 
                    Sigma0 = np.array([[1,0],[0,1]]), 
                    A = np.asarray([[1,0],[0,1]]), #Observation matrix. We want every state from our state vector.
                    Phi = np.array([[1,1],[0,1]]), #Transition matrix. Displacement is updated with prev disp + curr vel while velocity is updated with prev vel (assuming no acc.) 
                    fullOut = 0, # provide denerated quantities as well as predictions/filtering [0 - False; 1 - True]
                   ):
    N, M = y.shape
    Q = cQ @ (cQ.T) # cholesky decomp: cQ=chol(Q), cR=chol(R)
    R = cR @ (cR.T)
    # Initialize
    xp=[] # xp=x_t^{t-1} (prediction) 
    Pp=[] # Pp=P_t^{t-1} (prediction covariance)
    xf=[] # xf=x_t^t (filtering)
    Pf=[] # Pf=x_t^t (filtering covariance)
    innov=[] # innovations
    sig=[] # innov var-cov matrix
    negLogLike=0
    for t in range(N):
        if t==0:
            xp.append( Phi @ mu0 ) # Eq. (6.18)       
            Pp.append( Phi @ Sigma0 @ (Phi.T) + Q ) # Eq. (6.19) 
        else:
            xp.append( Phi @ xf[t-1] ) # Eq. (6.18)  
            Pp.append( Phi @ Pf[t-1] @ (Phi.T) + Q) # Eq. (6.19)  
        
        sigtemp = A @ Pp[t] @ (A.T) + R # Eq. (6.24)  
        sig.append( (sigtemp.T+sigtemp)/2 ) # innov var - make sure it's symmetric
        siginv = np.linalg.inv( sig[t] )              
        K = Pp[t] @ (A.T) @ siginv # Eq. (6.22)  
        innov.append( y[t,] - A @ xp[t] ) # Eq. (6.23)    
        xf.append( xp[t] + K @ innov[t] ) # Eq. (6.20) 
        Pf.append( Pp[t] - K @ A @ Pp[t] )
        negLogLike += np.log(np.linalg.det(sig[t])) + (innov[t].T) @ siginv @ innov[t] # Eq. (6.60)  
    negLogLike *= 0.5 
    
    if (fullOut == 1):
        return(xp, Pp, xf, Pf, negLogLike, K)
    else:
        return(xp, Pp, xf, Pf, negLogLike)
    
# Kalman Filter and Smoother - Time invariant model without inputs
# See Shumway & Stoffer (2017), ch. 6.
# https://rdrr.io/cran/astsa/src/R/Ksmooth0.R
def myKalmanSmoother0(y, # data np.array
                      cQ, cR,
                      mu0 = np.asarray([30,20]), #first estimate 
                      Sigma0 = np.array([[1,0],[0,1]]), 
                      A = np.asarray([[1,0],[0,1]]), #Observation matrix. We want every state from our state vector.
                      Phi = np.array([[1,1],[0,1]]), #Transition matrix. Displacement is updated with prev disp + curr vel while velocity is updated with prev vel (assuming no acc.) 
                      fullOut = 0, # provide denerated quantities as well as predictions/filtering/smoothing
                     ):
    N, M = y.shape
    # Kalman filering
    if (fullOut == 1):
        xp, Pp, xf, Pf, negLogLike, K = myKalmanFilter0(y, cQ, cR, mu0, Sigma0, A, Phi, fullOut) # Call to kalman filter function
    else:
        xp, Pp, xf, Pf, negLogLike = myKalmanFilter0(y, cQ, cR, mu0, Sigma0, A, Phi) # Call to kalman filter function
    # Kalman smoothing
    xs = [None]*N # xs=x_t^n
    Ps = [None]*N # Ps=P_t^n
    J = [None]*N # J=J_t
    for t in range(N-1,0,-1):
        if t==N-1: # last time step, which is the first iteration
            xs[t] = xf[t] 
            Ps[t] = Pf[t]
        # Kalman smooting equations
        J[t-1] = Pf[t-1] @ (Phi.T) @ np.linalg.inv(Pp[t]) # Eq. (6.47)
        xs[t-1] = xf[t-1] + J[t-1] @ (xs[t] - xp[t]) # Eq. (6.45)
        Ps[t-1] = Pf[t-1] + J[t-1] @ (Ps[t] - Pp[t]) @ (J[t-1].T) # Eq. (6.46)
    
    if (fullOut == 1):
        # First time step
        J0 = Sigma0 @ (Phi.T) @ np.linalg.inv(Pp[0])
        x0n = mu0 + J0 @ (xs[0] - xp[0])
        P0n = Sigma0 + J0 @ (Ps[0] - Pp[0]) @ (J0.T)
        # Output
        return(xp, Pp, xf, Pf, xs, Ps, negLogLike, K, J, J0, x0n, P0n)
    else:
        # Output
        return(xp, Pp, xf, Pf, xs, Ps, negLogLike)
    

#############################################################################################################
# Forward filtering, backward sampling
#############################################################################################################

# Forward filtering, backward samping of state space (displacement, velocity)
# See Shumway & Stoffer (2017), ch. 6. (sec. 6.12; forward filtering backward sampling)
# See also Petris et al (2009), ch. 4. (sec. 4.4; FFBS)
# https://github.com/cran/dlm/blob/master/R/DLM.R
def myFFBS(y, # data np.array
           cQ, cR,
           mu0 = np.asarray([30,20]), #first estimate 
           Sigma0 = np.array([[1,0],[0,1]]), 
           A = np.asarray([[1,0],[0,1]]), #Observation matrix. We want every state from our state vector.
           Phi = np.array([[1,1],[0,1]]), #Transition matrix. Displacement is updated with prev disp + curr vel while velocity is updated with prev vel (assuming no acc.) 
          ):
    N, M = y.shape
    D = mu0.shape[0]
    # Kalman filering
    xp, Pp, xf, Pf, _ = myKalmanFilter0(y, cQ, cR, mu0, Sigma0, A, Phi) # Call to kalman filter function
    # Backward sampling
    xb = [None]*N # xb=x_t^n
    jitter = np.eye(D) * 1e-8
    for t in range(N-1,0,-1):
        if t==N-1: # last time step, which is the first iteration
            xb[t] = xf[t] + np.linalg.cholesky(Pf[t] + jitter) @ np.random.normal(size=(D,1))
        # Backward filtering equations (note that t-1 corr. to t in Eq. 6.218)
        Jt = Pf[t-1] @ (Phi.T) @ np.linalg.inv(Pp[t]) # Eq. (6.47)
        mt = xf[t-1] + Jt @ (xb[t] - xp[t]) # Eq. (6.218 (and 6.45))
        Vt = Pf[t-1] - Jt @ Pp[t] @ (Jt.T) # Eq. (6.218 (and 6.46))  
        xb[t-1] = mt + np.linalg.cholesky(Vt + jitter) @ np.random.normal(size=(D,1))
        
    return(xb)


##############################################################################
# Gibbs sampler for one-parameter base Q-kernel
##############################################################################
def myGibbsQbase(xb, mu0, Phi, Qbase, a0=1, b0=1):
    N, M, _ = xb.shape
    xb_pred = np.concatenate([mu0.T, xb[:-1,:,0]]) # matrix of predictors
    invQbase = np.linalg.inv(Qbase)
    # Sum of squared errors
    errors = (xb[:,:,0] - xb_pred @ Phi.T)
    SStot = np.sum(np.diag((errors[1:,:] @ invQbase.T) @ errors[1:,:].T)) # skip the first prediction (numerical more stable), as m0 is not related to x_1
    # Draw parameter
    para_samp = np.random.gamma( shape=(a0 + (N*M)/2), scale=(1/(b0 + 0.5*SStot)) )
    
    return(para_samp)

def myGibbsQbase_slow(xb, mu0, Phi, Qbase, a0=1, b0=1):
    N, M, _ = xb.shape
    xb_pred = np.concatenate([mu0.T, xb[:-1,:,0]]) # matrix of predictors
    invQbase = np.linalg.inv(Qbase)
    # Sum of squared errors
    SStot = 0
    for t in range(1,N): # skip the first prediction (numerical more stable), as m0 is not related to x_1
        error_t = xb[t,:,0].T - Phi @ xb_pred[t,:].T
        SStot += (error_t.T) @ invQbase @ error_t 
    # Draw parameter
    para_samp = np.random.gamma( shape=(a0 + (N*M)/2), scale=(1/(b0 + 0.5*SStot)) )
    
    return(para_samp)

# Posterior sampling of states (FFBS) and parameters (Gibbs)
def myPosSamplerQbase(y, # data np.array
                      cQ, cR,
                      mu0 = np.asarray([30,20]), #first estimate 
                      Sigma0 = np.array([[1,0],[0,1]]), 
                      A = np.asarray([[1,0],[0,1]]), #Observation matrix. We want every state from our state vector.
                      Phi = np.array([[1,1],[0,1]]), #Transition matrix. Displacement is updated with prev disp + curr vel while velocity is updated with prev vel (assuming no acc.) 
                      Qbase = np.diag([0., 0., 0.1]),
                      a0=1, b0=1, # hyperprior parameters
                      Nsamp = 1000,
                     ):
    
    xb_list = []
    para_list = []
    Qbase += np.eye(Qbase.shape[0])*1e-8 # adding a little jitter
    for i in range(Nsamp):
        if (i>0):
            Q = Qbase * (1/para)
            cQ = np.sqrt(Q)
        # Forward filtering, backward sampling (FFBS)
        xb = myFFBS(y=y, cQ=cQ, cR=cR, mu0=mu0, Sigma0=Sigma0, A=A, Phi=Phi)
        xb = np.array(xb)
        # Gibbs sampling of parameters
        para = myGibbsQbase(xb=xb, mu0=mu0, Phi=Phi, Qbase=Qbase); # print(para)
        # para = myGibbsQbase_slow(xb=xb, mu0=mu0, Phi=Phi, Qbase=Qbase); # print(para)

        # Book keeping
        xb_list.append(xb)
        para_list.append(para)
    
    return(xb_list, para_list)


####################################################################################
# Gibbs sampler for d-inverse gamma (DIG) model
####################################################################################
# See also Petris et al (2009), ch. 4. (sec. 4.4; FFBS)
# https://github.com/cran/dlm/blob/master/R/DLM.R
def myGibbsDIG(xb, mu0, Phi, av=np.array([1,1,1]), bv=np.array([1,1,1])):
    N = xb.shape[0]
    xb_pred = np.concatenate([mu0.T, xb[:-1,:,0]]) # matrix of predictors
    errors = xb[:,:,0] - xb_pred @ Phi.T
    SSs = np.sum( errors**2, axis=0 ) # sum of squared errors
    # note that np.random.gamma uses 'scale' and Petris et al (2009) defines 'rate' (inverse scale)
    para_samp = np.random.gamma( shape=(av + N/2), scale=1/(bv + 0.5*SSs) )
    
    return(para_samp)

def myGibbsDIG_slow(xb, mu0, Phi, av=np.array([1,1,1]), bv=np.array([1,1,1])):
    N, M, _ = xb.shape
    xb_pred = np.concatenate([mu0.T, xb[:-1,:,0]]) # matrix of predictors
    SSs=np.zeros(M)
    for t in range(1,N): # skip the first prediction (numerical more stable), as m0 is not related to x_1
        error_t = xb[t,:,0].T - Phi @ xb_pred[t,:].T # errors
        SSs += error_t**2 # sum of squared errors
    # note that np.random.gamma uses 'scale' and Petris et al (2009) defines 'rate' (inverse scale)
    para_samp = np.random.gamma( shape=(av + N/2), scale=1/(bv + 0.5*SSs) )
    
    return(para_samp)

# Posterior sampling of states (FFBS) and parameters (Gibbs)
def myPosSamplerDIG(y, # data np.array
                    cQ, cR,
                    mu0 = np.asarray([30,20]), #first estimate 
                    Sigma0 = np.array([[1,0],[0,1]]), 
                    A = np.asarray([[1,0],[0,1]]), #Observation matrix. We want every state from our state vector.
                    Phi = np.array([[1,1],[0,1]]), #Transition matrix. Displacement is updated with prev disp + curr vel while velocity is updated with prev vel (assuming no acc.) 
                    av=np.array([1]*3), bv=np.array([1]*3), # hyperprior parameters
                    Nsamp = 1000,
                   ):
    
    xb_list = []
    para_list = []
    for i in range(Nsamp):
        if (i>0):
            Q = np.diag(1/para) + np.eye(cQ.shape[0])*1e-8 # adding a little jitter
            cQ = np.sqrt(Q)
        # Forward filtering, backward sampling (FFBS)
        xb = myFFBS(y=y, cQ=cQ, cR=cR, mu0=mu0, Sigma0=Sigma0, A=A, Phi=Phi)
        xb = np.array(xb)
        # Gibbs sampling of parameters
        para = myGibbsDIG(xb=xb, mu0=mu0, Phi=Phi); # print(para)
        # para = myGibbsDIG_slow(xb=xb, mu0=mu0, Phi=Phi); # print(para)

        # Book keeping
        xb_list.append(xb)
        para_list.append(para)
    
    return(xb_list, para_list)



