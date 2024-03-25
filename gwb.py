import numpy as np
import networkx as nx
import ot
from copy import deepcopy
#from gm import GM
from tqdm import trange
from sklearn.decomposition import PCA
from ot.gromov import cg,gwggrad,gwloss,init_matrix
from scipy.sparse import csr_matrix
from GromovWassersteinFramework import gromov_wasserstein_discrepancy as prox_gw

#CLASS

class GM:
    def __init__(self,mode,gauge_mode = None,X = None,g = None,xi = None,Tris=None,Nodes=None,Edges=None,normalize_gauge=False,squared=False):
        if mode == "gauge_only":
            assert type(g) is np.ndarray
            
        elif mode == "euclidean":
            assert gauge_mode == "euclidean" or gauge_mode == "sqeuclidean"
            assert X is not None
                   
        elif mode == "graph" or mode == "weighted_graph":
            assert Edges is not None
        
        elif mode == "surface":
            assert Tris is not None
            assert X is not None
        
        self.X = X
        self.mode = mode
        self.gauge_mode = gauge_mode
        self.normalize_gauge = normalize_gauge
        self.Tris = Tris
        self.Edges = Edges
        self.Nodes = Nodes
        
        #Create Graph if necessary
        if self.mode == "graph" or self.mode == "weighted_graph" or self.mode == "surface":
            if self.mode == "graph":
                self.G = nx.Graph()
                if self.Nodes is not None:
                    self.G.add_nodes_from(self.Nodes)
                self.G.add_edges_from(self.Edges)
            elif self.mode == "weighted_graph":
                self.G = nx.Graph()
                if self.Nodes is not None:
                    self.G.add_nodes_from(self.Nodes)
                self.G.add_weighted_edges_from(self.Edges)
            elif self.mode == "surface":
                self.G = gen_graph_from_surface(self.X,Tris)
            
        self.g = self.set_g(g)
        if squared == True:
            self.g = self.g**2
        self.len = len(self.g)
        self.xi = self.set_xi(xi)
        
    def set_X(self,X):
        return X

    def set_g(self,g):
        if self.mode == "gauge_only":#type(g) == np.ndarray:
            g = g
        elif self.mode == "euclidean":
            g = ot.dist(self.X,metric=self.gauge_mode)
        elif self.mode == "surface" or self.mode == "graph" or self.mode == "weighted_graph":
            if self.gauge_mode == "adjacency":
                g = np.array(nx.adjacency_matrix(self.G).todense(),dtype=float)
            elif self.gauge_mode == "djikstra":
                g = self.djikstra_gauge()
        if self.normalize_gauge:
            g /= np.max(g)
        return g
    
    def set_xi(self,xi):
        if xi is None:
            return ot.unif(self.len)
        elif type(xi) == str and xi == "surface_uniform":
            return surface_uniform_xi(self.X,self.Tris)
        else:
            return xi
        
    def djikstra_gauge(self):
        if self.mode == "weighted_graph" or self.mode == "surface":
            dic = dict(nx.weighted.all_pairs_dijkstra_path_length(self.G))
        elif self.mode == "graph":
            dic = dict(nx.all_pairs_dijkstra_path_length(self.G))
        g = np.zeros((len(self.G.nodes),len(self.G.nodes)))
        for key in dic.keys():
            g[int(key),np.array(list(dic[key].keys()),dtype=int)] = np.array(list(dic[key].values()))
        g = (1/2) * (g + g.T)
        return g
    

    
    
    
    
    
    
    

#FUNCTIONS

## Functions used for class GM

def gen_graph_from_surface(X,Tris):
    G = nx.Graph()
    G.add_nodes_from(range(len(X)))
    for i in range(len(Tris)):
        tri = Tris[i]
        for l1 in range(3):
            for l2 in range(l1+1,3):
                G.add_edge(tri[l1],tri[l2],weight = np.linalg.norm(X[tri[l1]] - X[tri[l2]]))
    return G






## convenience functions for multi-shape interpolation based on results from TB iterations

def interpolate_two(X,Y,idxs,meas,P,N_interpolation):
    """
    Function to interpolate between two gm-spaces based on a GW transport plan.
    X,Y: objects of class GM
    idxs: np.ndarray with shape (n,2) of coupled indices between X and Y
    meas: np.ndarray with length n of coupled masses between indices
    P: GW transport plan from X to Y
    """
    
    #colour X and Y according to P
    cX = np.linalg.norm(X.X - X.X[np.argmin(np.linalg.norm(X.X,axis=1))],axis=1)
    cY = (P.T / np.sum(P,axis=1)).dot(cX)

    #weights
    ws = np.flip((np.stack([np.arange(1,N_interpolation+1,1),np.flip(np.arange(1,N_interpolation+1,1))])/(N_interpolation+1)).T,axis=0)

    #embedding for barycenters
    bary_embs = []
    bary_embs_3d = []
    pca_scores = []
    for w in ws:
        bary_emb = create_surface_embedding([X,Y],idxs,meas,ws=np.sqrt(w))
        bary_embs.append(bary_emb)

        #PCA for dimensionality reduction
        pca = PCA(n_components=3)
        bary_emb_3d = pca.fit_transform(bary_emb)#,**{'weights': [[b]*6 for b in meas]})
        pca_scores.append(pca_score(pca,bary_emb,meas))
        bary_embs_3d.append(bary_emb_3d)

    #colour barycenters
    bary_c = [w[0]*cX[idxs[:,0]] + w[1] * cY[idxs[:,1]] for w in ws]
    
    return cX,cY,bary_embs,bary_embs_3d,bary_c,pca_scores

def interpolate_four(Xs,idxs,meas,Ps,n_grid): 
    """
    Function to fully interpolate between four gm-spaces based on a GW transport plan.
    Xs: List of objects of class GM
    idxs: np.ndarray with shape (n,4) of coupled indices between gm-spaces in Xs
    meas: np.ndarray with length n of coupled masses between indices
    Ps: List of GW transport plans from some reference gm-space to the gm-spaces in Xs
    n_grid: Length of interpolation grid
    """
    N = len(Xs)
    cX = np.linalg.norm(Xs[0].X - Xs[0].X[np.argmin(np.linalg.norm(Xs[0].X,axis=1))],axis=1)
    cXs = [(P.T / np.sum(P,axis=1)).dot(cX) for P in Ps]

    #weights
    w = np.linspace(0, 1, n_grid)
    ws = np.stack([
        (1-w[:, None])*(1-w[None, :]),
        (1-w[:, None])*w[None, :],
        w[:, None]*(1-w[None, :]),
        w[:, None]*w[None, :]
    ], axis=0).transpose((1, 2, 0)).reshape(-1, 4)

    #embedding for barycenters
    bary_tris = np.zeros((n_grid,n_grid),dtype=object)
    bary_embs = np.zeros((n_grid,n_grid),dtype=object)
    bary_embs_3d = np.zeros((n_grid,n_grid),dtype=object)
    bary_cs = np.zeros((n_grid,n_grid),dtype=object)
    pcas = np.zeros((n_grid,n_grid))
    for n,w in enumerate(ws):
        i = n // n_grid
        j = n % n_grid
        
        bary_emb,bary_emb_3d,bary_tri,pca_s,bary_c = single_interpolate_four(Xs,idxs,meas,cXs,w=w)
        bary_embs[i,j] = bary_emb
        bary_embs_3d[i,j] = bary_emb_3d
        bary_tris[i,j] = bary_tri
        pcas[i,j] = pca_s
        bary_cs[i,j] = bary_c

    return cXs,bary_embs,bary_embs_3d,bary_cs,bary_tris,pcas

def single_interpolate_four(Xs,idxs,meas,cXs,w=None):
    """
    Function for a single interpolant between four gm-spaces based on a GW transport plan.
    Xs: List of objects of class GM
    idxs: np.ndarray with shape (n,4) of coupled indices between gm-spaces in Xs
    meas: np.ndarray with length n of coupled masses between indices
    cXs: Colours of the GM-spaces in Xs
    w: Weight of the interpolant
    """
    N = len(Xs)
    if w is None:
        idx = 0
        w = ot.unif(N)
        sqrtw = np.sqrt(w)
    else:
        idx = np.argmax(w)
        sqrtw = np.sqrt(w)
        
    bary_emb = create_surface_embedding(Xs,idxs,meas,ws=sqrtw)

    #PCA for dimensionality reduction
    pca = PCA(n_components=3)
    bary_emb_3d = pca.fit_transform(bary_emb)#,**{'weights': [[b]*6 for b in meas]})
    pca_s = pca_score(pca,bary_emb,meas)

    #triangles
    bary_tri = generate_triangles_from_single(Xs[idx],idxs,idx)
    #colour
    bary_c = np.sum(w.reshape((len(w),1)) * np.array([cXs[k][idxs[:,k]] for k in range(N)]),axis=0)
    
    return bary_emb,bary_emb_3d,bary_tri,pca_s,bary_c

def generate_triangles_from_single(X,idxs,ix):
    """
    Function to generate barycentric triangles based on a reference gm-space.
    X: object of class GM
    idxs: np.ndarray of coupled indices between gm-spaces
    ix: index of X with respect to idxs
    """
    Tris = []
    for tri in X.Tris:
        for i in np.where(idxs[:,ix] ==  tri[0])[0]:
            for j in np.where(idxs[:,ix] ==  tri[1])[0]:
                for k in np.where(idxs[:,ix] ==  tri[2])[0]:
                    Tris.append([i,j,k])
    Tris = np.array(Tris)
    return Tris
    
def create_surface_embedding(Xs,idxs,meas,ws):
    N = len(Xs)
    Nodes = np.concatenate([ws[i] * Xs[i].X[idxs[:,i]] for i in range(N)],axis=1)
    return Nodes

def pca_score(pca,emb,meas):
    return meas.dot([np.linalg.norm(emb[i].dot(pca.components_.T).dot(pca.components_) - emb[i]) for i in range(len(emb))])





    
    
## Functions for TB Iterations.

def tb(Y,Xs,numItermaxEmd = 500000,init_Ps=None,cr="NWCR",method="cg"):
    """
    Main Function which implements a single tangential GW barycenter iteration. 
    Computes GW Transport from Y to all inputs Xs
    and constructs a gluing/melting by the north-west-corner-rule (cr="NWCR")
    or by the maximum rule without replacement (cr="MCR").
    Y: Input gm-space of class GM. Alternatively Y can be an index if the reference is one of the input spaces.
    Xs: List of objects of class GM
    """
    if method == "prox":
        ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
           'ot_method': 'proximal',
           'beta': 0.025,
           'outer_iteration': 2000,
           # outer, inner iteration, error bound of optimal transport
           'iter_bound': 1e-30,
           'inner_iteration': 2,
           'sk_bound': 1e-30,
           'node_prior': 1e3,
           'max_iter': 4,  # iteration and error bound for calcuating barycenter
           'cost_bound': 1e-26,
           'update_p': False,  # optional updates of source distribution
           'lr': 0,
           'alpha': 0}
    #functions needed for GW computation
    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    N = len(Xs)
    if type(Y) is int:
        assert Y < len(Xs)
        piv_index = Y
        Y = Xs[piv_index]
    else:
        piv_index = None
    Ps = []
    for i in range(N):
        if i == piv_index:
            P = np.diag(Xs[piv_index].xi)
        else:
            if init_Ps is None and method == "cg":
                P = ot.gromov.gromov_wasserstein(Y.g,Xs[i].g,Y.xi,Xs[i].xi,numItermaxEmd=numItermaxEmd)
            elif method=="prox":
                P = prox_gw(csr_matrix(Y.g),csr_matrix(Xs[i].g),
                        Y.xi.reshape(-1,1),Xs[i].xi.reshape(-1,1),ot_hyperpara=ot_dict,trans0 = init_Ps[i])[0]
            else:
                constC, hC1, hC2 = init_matrix(Y.g, Xs[i].g, Y.xi, Xs[i].xi, "square_loss")
                P = cg(Y.xi, Xs[i].xi, 0, 1, f, df, init_Ps[i], armijo=False, C1=Y.g, C2=Xs[i].g,
                       constC=constC,log=True,numItermaxEmd=numItermaxEmd)[0]
        Ps.append(P)
    if cr == "NWCR":
        idxs,meas,ref_idx = NWCR(Ps)
    elif cr == "MCR":
        idxs,meas,ref_idx = MCR(Ps)
    return idxs,meas,Ps,ref_idx


def avg_Ms_by_idxs(Ms,idxs,ws):
    """
    Function that averages the gauge based on a multi-marginal GW coupling.
    Xs: List of objects of class GM
    idxs: np.ndarray of coupled indices between gm-spaces with gauges Ms
    ws: weights w.r.t. the gauge functions
    """
    N = len(Ms)
    n = len(idxs)
    M_out = np.zeros((n,n))
    for i in range(N):
        M_out += ws[i] * Ms[i][idxs[:,i]].T[idxs[:,i]].T
    return M_out

    
def gwb_loss(Y,Xs,Ps,ws = None):
    """
    Function that computes the GW Barycenter Loss of gm-space Y with respect to input gm-spaces Xs
    based on precomputed plans Ps.
    Y: Approximate Barycenter, object of class GM
    Xs: List of objects of class GM
    Ps: list of GW Transport plans from Y to gm-spaces in Xs
    ws: weights w.r.t. input spaces Xs
    """
    N = len(Xs)    
    if type(Y) == int:
        Y = Xs[Y]
    if ws is None:
        ws = ot.unif(N)
    bary_loss = 0
    for i in range(N):
        constC, hC1, hC2 = ot.gromov.init_matrix(Y.g,Xs[i].g,Y.xi,Xs[i].xi)
        bary_loss += ws[i] * ot.gromov.gwloss(constC,hC1,hC2,Ps[i])
    return bary_loss


def shrink_bi_idxs(idx1,idx2,meas):
    """
    Function that shrinks bi-marginal idxs by summing the measure when paired indices appear multiple times.
    idx1,idx2: idxs to be shrinked
    meas: np.ndarray with same length as idx1,idx2 consisting of coupled masses between indices
    """
    N = len(idx1)
    assert len(idx2) == N
    outidx1 = []
    outidx2 = []
    outmeas = []
    l = -1
    for i in range(N):
        if idx1[i] == idx1[i-1] and idx2[i] == idx2[i-1]:
            outmeas[l] += meas[i]
        else:
            outidx1.append(idx1[i])
            outidx2.append(idx2[i])
            outmeas.append(meas[i])
            l+=1
    return outidx1,outidx2,outmeas

def create_backwards_initplans(Xs,ref_idx,idxs,meas):
    """
    Function to create a transport plan based on bi-marginal projection of multi-marginal idxs.
    Xs: List of objects of class GM
    ref_idxs: idxs of the reference gm-space of len n
    idxs: np.ndarray with shape (n,4) of coupled indices between gm-spaces in Xs
    meas: np.ndarray with length n of coupled masses between indices
    """
    Ps = []
    N = len(Xs)
    for l in range(N):
        P = np.zeros((len(idxs),Xs[l].len))
        ref_idx_shrink, idxs_l_shrink, meas_shrink = shrink_bi_idxs(ref_idx,idxs[:,l],meas)
        P[ref_idx_shrink, idxs_l_shrink] = meas_shrink
        Ps.append(P)
    return Ps

def bi_idxs_2_multi_idxs(bi_idxs):
    """
    Generates multi-marginal idxs from a list of bi-marginal indices.
    """
    idxs = []
    bi_idxs = np.array([idx[idx[:, 0].argsort()] for idx in bi_idxs])
    for i in bi_idxs[0][:,0]:
        midx = [idx[i,1] for idx in bi_idxs]
        idxs.append(midx)
    return idxs


def node_pair_assignment(trans, p_s, p_t):
    """
    Derives Node pairs based on the maximum rule without replacement.
    Slight modification of the function with the same name in the
    GromovWassersteinGraphToolkit.py from https://github.com/HongtengXu/s-gwl
    """
    pairs_idx = []
    pairs_confidence = []
    if trans.shape[0] >= trans.shape[1]:
        source_idx = list(range(trans.shape[0]))
        for t in range(trans.shape[1]):
            column = trans[:, t] / p_s[:, 0]  # p(t | s)
            idx = np.argsort(column)[::-1]
            for n in range(idx.shape[0]):
                if idx[n] in source_idx:
                    s = idx[n]
                    pairs_idx.append([s, t])
                    pairs_confidence.append(trans[s, t])
                    source_idx.remove(s)
                    break
    else:
        target_idx = list(range(trans.shape[1]))
        for s in range(trans.shape[0]):
            row = trans[s, :] / p_t[:, 0]
            idx = np.argsort(row)[::-1]
            for n in range(idx.shape[0]):
                if idx[n] in target_idx:
                    t = idx[n]
                    pairs_idx.append([s, t])
                    pairs_confidence.append(trans[s, t])
                    target_idx.remove(t)
                    break
    return pairs_idx

def MCR(Ps):
    """
    Constructs a melting from given transport plans Ps based on the Maximum rule without replacement.
    """
    bi_idxs = np.array([np.array(node_pair_assignment(P,np.sum(P,axis=1).reshape(-1,1),np.sum(P,axis=1).reshape(-1,1)))
            for P in Ps])
    m_idxs = np.array(bi_idxs_2_multi_idxs(bi_idxs))
    return m_idxs, np.sum(Ps[0],axis=1), np.arange(len(m_idxs))

def NWCR(Ps):
    """
    Constructs a melting from given transport plans Ps based on the North-West Corner Rule.
    """
    Ps = deepcopy(Ps)
    N = len(Ps)
    n = np.shape(Ps[0])[0]
    nu = np.sum(Ps[0],axis=1)
    assert np.all([nu - np.sum(P,axis=1) < 1e-9 for P in Ps])

    idxs = []
    ref_idx = []
    meas = []
    for i in range(n):
        nz_idxs_per_i = [np.where(P[i,:] != 0)[0] for P in Ps]
        ls = np.zeros(N,dtype=int)
        while nu[i] > 1e-15:
            tup = np.array([nz_idxs_per_i[k][ls[k]] for k in range(N)])
            min_mass = np.min([Ps[k][i,tup[k]] for k in range(N)])

            idxs.append(tup)
            meas.append(min_mass)
            nu[i] -= min_mass
            ref_idx.append(i)
            for k in range(N):
                Ps[k][i,tup[k]] -= min_mass
                if Ps[k][i,tup[k]] == 0:
                    ls[k] += 1
    return np.array(idxs),np.array(meas),np.array(ref_idx)

def bary_from_tb(Xs,idxs,meas,ws = None):
    """
    Constructs a Barycenter based on output from TB iteration.
    """
    if ws is None:
        ws = ot.unif(len(Xs))
    g_avg = avg_Ms_by_idxs([X.g for X in Xs],idxs,ws = ws)
    return GM(mode="gauge_only",g=g_avg,xi=meas)

