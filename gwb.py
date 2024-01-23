import numpy as np
import networkx as nx
import ot
from copy import deepcopy
#from gm import GM
from tqdm import trange
from wpca import WPCA
#from keypoint_guided_optimal_transport.keypoint_guided_OT import KeyPointGuidedOT

#CLASS

class GM:
    def __init__(self,mode,gauge_mode = None,X = None,g = None,xi = None,Tris=None,Nodes=None,Edges=None,normalize_gauge=False,squared=False):#,Nodes=None,Tris = None,Edges=None,metric="euclidean",normalize=False,mu = None):
        if mode == "gauge_only":
            assert type(g) is np.ndarray
            
        elif mode == "euclidean":
            assert gauge_mode == "euclidean" or gauge_mode == "sqeuclidean"
            assert X is not None
                   
        elif mode == "graph" or mode == "weighted_graph":
            assert Edges is not None
            #if gauge_mode == "djikstra":
                #assert np.shape(Edges)[1] == 3
            #if gauge_mode == "euclidean" or gauge_mode == "sqeuclidean":
            #    assert X is not None
        
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

def sample_GM(X,n,mode="gauge_only"):
    sort = np.argsort(X.xi)
    idxs = sort[-n:]
    new_xi = X.xi[idxs]
    new_xi /= np.sum(new_xi)
    new_g = X.g[idxs].T[idxs].T
    return GM(xi = new_xi,g=new_g,mode=mode)

def avg_Ms_by_idxs(Ms,idxs,ws):
    N = len(Ms)
    n = len(idxs)
    M_out = np.zeros((n,n))
    for i in range(N):
        M_out += ws[i] * Ms[i][idxs[:,i]].T[idxs[:,i]].T
    return M_out


def thres_idxs(idxs,meas,thres):
    if type(thres) is int:
        sort = np.argsort(meas)
        return idxs[sort][-thres:], meas[sort][-thres:]
    elif type(thres) is float:
        I = np.where(meas > thres)[0]
        return idxs[I],meas[I]

def tb(Y,Xs,numItermaxEmd = 500000,Ks=None):
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
            if Ks is None:
                P = ot.gromov.gromov_wasserstein(Y.g,Xs[i].g,Y.xi,Xs[i].xi,numItermaxEmd=numItermaxEmd)
            else:
                print("KPGOT NOT IMPLEMENTED!")
                kgot = KeyPointGuidedOT()
                P = np.array(kgot.kpg_rl_gw(Y.xi,Xs[i].xi,Y.g,Xs[i].g,K=Ks[i]),dtype=float)
        Ps.append(P)
    idxs,meas = NWCR(Ps)
    return idxs,meas,Ps

def NWCR(Ps):
    Ps = deepcopy(Ps)
    N = len(Ps)
    n = np.shape(Ps[0])[0]
    nu = np.sum(Ps[0],axis=1)
    assert np.all([nu - np.sum(P,axis=1) < 1e-9 for P in Ps])

    idxs = []
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
            for k in range(N):
                Ps[k][i,tup[k]] -= min_mass
                if Ps[k][i,tup[k]] == 0:
                    ls[k] += 1
    return np.array(idxs),np.array(meas)

def TB(Y,Xs,ws,mode,squared=False,log=False):
    "TODO"
    idxs,meas,Ps = tb(Y,Xs)
    if mode=="avg_gauge_only":
        #compute the gauge
        g_avg = avg_Ms_by_idxs([X.g for X in Xs],idxs,ws = ws)
        Y_out = GM(mode="gauge_only",g=g_avg,xi=meas)
    elif mode=="euclidean":
        print("TODO")
    elif mode=="surface":
        create_embedding(idxs,meas,Ps,ws = ws)
    if log == True:
        return Y_out,{"idxs": idxs, "meas": meas, "Ps": Ps}
    return Y_out

def LGW_bimarg_via_idxs(X,Y,idxs_X,idxs_Y,meas):
    return ((X.g[idxs_X].T[idxs_X].T - Y.g[idxs_Y].T[idxs_Y].T)**2).dot(meas).dot(meas)

def LGW_via_idxs(Xs,idxs,meas):
    N = len(Xs)
    lgw_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            lgw_mat[i,j] = LGW_bimarg_via_idxs(Xs[i],Xs[j],idxs[:,i],idxs[:,j],meas)
    return 1/2 * (lgw_mat + lgw_mat.T)

def pairwise_GW(Xs):
    N = len(Xs)
    gw_mat = np.zeros((N,N))
    for i in trange(N):
        for j in range(i+1,N):
            gw_mat[i,j] = ot.gromov_wasserstein(Xs[i].g,Xs[j].g,Xs[i].xi,Xs[j].xi,log=True)[1]["gw_dist"]
    return 1/2 * (gw_mat + gw_mat.T)


    
def area_of_tri(tri):
    p1,p2,p3 = tri
    a = np.linalg.norm(p1-p2)
    b = np.linalg.norm(p2-p3)
    c = np.linalg.norm(p3-p1)
    s = (a+b+c)/2
    return np.sqrt(s*(s-a)*(s-b)*(s-c))

def surface_uniform_mu(Nodes,Tris):
    mu = []
    for i in range(len(Nodes)):
        tris_at_i = Tris[np.where(Tris == i)[0]]
        area = np.sum([area_of_tri(Nodes[tri]) for tri in tris_at_i])
        mu.append(np.sqrt(area))
    mu = np.array(mu)
    return mu / np.sum(mu)

def surface_uniform_xi(X,Tris):
    mu = np.zeros(len(X))
    for tri in Tris:
        mu[tri] += 1/3 * area_of_tri(X[tri])
    mu = np.sqrt(mu)
    mu /= np.sum(mu)
    return mu

def gen_graph_from_surface(X,Tris):
    G = nx.Graph()
    G.add_nodes_from(range(len(X)))
    for i in range(len(Tris)):
        tri = Tris[i]
        for l1 in range(3):
            for l2 in range(l1+1,3):
                G.add_edge(tri[l1],tri[l2],weight = np.linalg.norm(X[tri[l1]] - X[tri[l2]]))
    return G



def generate_triangles_from_single(X,idxs,ix):
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
    return Nodes##return GM(mode="euclidean",gauge_mode="")
