#Functions related to classification and
#Pairwise (L)GW computations

import gwb as gwb
import numpy as np
from tqdm import trange
from ot.gromov import cg,gwggrad,gwloss,init_matrix
import sklearn

def gw(X,Y,P_init = None,numItermaxEmd=50000):
    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    #Compute GW coupling with initial coupling set to Wasserstein coupling G0
    constC, hC1, hC2 = init_matrix(X.g, Y.g, X.xi, Y.xi, "square_loss")
    P,log = cg(X.xi, Y.xi, 0, 1, f, df, P_init, armijo=False, C1=X.g, C2=Y.g, constC=constC,log=True,numItermax=numItermaxEmd)
    return P,np.sqrt(f(P))

def pwgw(Xs):
    N = len(Xs)
    gw_mat = np.zeros((N,N))
    Ps_mat = np.zeros((N,N),dtype=object)
    for i in trange(N):
        for j in range(i+1,N):
            P,gw_dist = gw(Xs[i],Xs[j])
            Ps_mat[i,j] = P
            Ps_mat[j,i] = P.T
            gw_mat[i,j] = gw_dist
    gw_mat = gw_mat + gw_mat.T
    #Ps_mat = Ps_mat + Ps_mat.T
    return Ps_mat, gw_mat

def adj_gw_mat(Xs,gw_mat,Ps_mat):
    N = len(gw_mat)
    gw_mat_adj = np.copy(gw_mat)
    Ps_mat_adj = np.copy(Ps_mat)
    count = 0
    for i in range(N):
        for j in range(i+1,N):
            gw_ij = gw_mat[i,j]
            for k in range(0,N):
                diff = gw_ij - (gw_mat[k,j] + gw_mat[i,k])
                if diff > 0:
                    Pik = Ps_mat[i,k] 
                    Pkj = Ps_mat[k,j]

                    idxs,meas,_ = gwb.NWCR([Pik.T,Pkj])
                    P_init = bi_idxs_2_plan(idxs,meas)

                    P_ij_redo, gw_ij_redo = gw(Xs[i],Xs[j],P_init=P_init)
                    if gw_ij_redo < gw_ij:
                        gw_mat_adj[i,j] = gw_ij_redo
                        gw_mat_adj[j,i] = gw_ij_redo
                        Ps_mat_adj[i,j] = P_ij_redo
                        Ps_mat_adj[j,i] = P_ij_redo.T
                        count += 1
                    else:
                        print(i,j,k)
                        print("Something went wrong")
                    break
    return gw_mat_adj,Ps_mat_adj

def LGW_bimarg_via_idxs(X,Y,idxs_X,idxs_Y,meas):
    return np.sqrt(((X.g[idxs_X].T[idxs_X].T - Y.g[idxs_Y].T[idxs_Y].T)**2).dot(meas).dot(meas))

def LGW_via_idxs(Xs,idxs,meas):
    N = len(Xs)
    lgw_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            lgw_mat[i,j] = LGW_bimarg_via_idxs(Xs[i],Xs[j],idxs[:,i],idxs[:,j],meas)
    return (lgw_mat + lgw_mat.T)
    
def check_triangle_ineq(mat):
    N = len(mat)
    for i in range(N):
        for j in range(i):
            if not all(mat[i,j] <= mat[i,:] + mat[:,j]):
                return False
    return True

def bi_idxs_2_plan(idxs,meas):
    xlen = np.max(idxs[:,0]) + 1
    ylen = np.max(idxs[:,1]) + 1
    P = np.zeros((xlen,ylen))
    for i in range(len(idxs)):
        P[idxs[i][0],idxs[i][1]] = meas[i]
    return P

def conf_mat(dists,X,y,n_its = 10000):
    N = len(y)
    classes = np.array(np.unique(y),dtype=int)
    x_pred = []
    x_true = []
    for j in range(n_its):
        l = []
        for i in classes:
            l.append([np.random.choice(X[y == i]),i])
        l = np.array(l)
        for i in range(N):
            tmp1 = np.argmin(dists[i][l[:,0]])
            x_true.append(y[i])
            x_pred.append(tmp1)
    
    conf = sklearn.metrics.confusion_matrix(x_true, x_pred,normalize="true")
    return conf