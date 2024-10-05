import numpy as np
import time
from sklearn.neighbors import KernelDensity
import math
from sklearn.neighbors import RadiusNeighborsTransformer
import heapq

class FACE:
    def __init__(
            self,
            clf,
            data: np.ndarray,
            radius: float,
            epsilon: float,
            tp: float,
            td: float,
            ):
        self.data = data
        self.clf = clf
        self.radius = radius
        self.epsilon = epsilon
        self.tp = tp
        self.td = td
        self.kernel = KernelDensity()
        self.kernel.fit(self.data)
        self.radius_graph = RadiusNeighborsTransformer(mode='distance', radius=self.radius, algorithm='auto',metric='minkowski', p=2)
        self.radius_graph.fit(self.data)
        self.dist_list = self.radius_graph.radius_neighbors(X=self.data, radius=self.epsilon)
        self.wN = self.make_graph_adjList()


    def means(self, xi, xj):
        return 0.5*(xi + xj)

    def kernelKDE(self, means):
        density_at_mean = self.kernel.score(means)
        return np.exp(density_at_mean)

    def density_function(self, density_at_mean):
        return -np.log(density_at_mean)

    def computeDistance(self, xi, xj):
        dist = np.linalg.norm(xi - xj, 2)
        return dist

    def make_graph_adjList(self):
        N = len(self.data)
        G = [[] for i in range(N)]
        for i in range(0, N):
            for k in range(len(self.dist_list[0][i])):
                j = self.dist_list[1][i][k]
                xi = self.data[i]
                xj = self.data[j]
                dist = self.dist_list[0][i][k]
                wij =dist*self.density_function(self.kernelKDE(self.means(xi, xj).reshape(1,-1)))
                G[i].append([j,wij])
                G[j].append([i,wij])
        return G

    def get_candidates(self):
        candidates = {}
        for x_id, x in enumerate(self.data):
            if (self.clf.predict_proba(x.reshape(1,-1))[0][1] >= self.tp):
                candidates[x_id] = x
        return candidates

    def is_connected(self, sourse_id, target_id):
        n = len(self.wN)
        visited = [False]*n
        def dfs(v):
            visited[v]=True
            for k in self.wN[v]:
                w = k[0]
                if visited[w] is False:
                    dfs(w)
        dfs(sourse_id)
        if visited[target_id] is False:
            return False
        return True

    def shortestPath(self, sourse_id, target_id):
        n = len(self.wN)
        infty = math.inf
        dist = [infty]*n
        dist[sourse_id] = 0
        L = [[0,sourse_id]]
        heapq.heapify(L)

        pre = [-1]*n
        selected = [False]*n

        while len(L)>0:
            [d, u] = heapq.heappop(L) #ヒープを使って起点を選ぶ
            if selected[u] is False:
                selected[u]=True
                for w, l in self.wN[u]:
                    if dist[w] > dist[u]+l:
                        pre[w] = u
                        dist[w] = dist[u]+l
                        heapq.heappush(L,[dist[w],w]) #訪問予定ヒープに[dist[w],w]を追加.
        k=target_id
        Rlist=[]
        while k!=sourse_id:
            Rlist.append(self.data[pre[k]])
            if pre[k] == -1:
                return False,False
            else:
                k=pre[k]
        Rlist.pop()
        Rlist.reverse
        return Rlist , dist[target_id]

    def compute_recourse(self, source_id):
        start_time = time.time()

        candidates = self.get_candidates()
        min_path_cost = float('inf')
        min_target = -1
        min_path = None
        for candidate_id in candidates:
            candidate	= candidates[candidate_id]
            closest_target_path, path_cost = self.shortestPath(source_id, candidate_id)
            if closest_target_path is False:
                continue
            if (path_cost < min_path_cost):
                min_target = candidate
                min_path_cost = path_cost
                min_path = closest_target_path
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processing time of FACE: {elapsed_time} second")
        return min_target, min_path_cost, min_path


