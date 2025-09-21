import itertools
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
dados=pd.read_csv('data.csv')
M=5
atribuicoes=(100*np.random.rand(M, M)).astype(int).tolist()
x=dados['x'].values
y=dados['y'].values
plt.figure(figsize=(4,4))
plt.title('Entregas')
plt.xlabel('Coordenadas x')
plt.ylabel('Coordenadas y')
plt.scatter(x, y)
plt.grid();
class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        """
        Args:
            distances (2D numpy.array): Matriz de distâncias entre as cidades.
            n_ants (int): Número de formigas.
            n_best (int): Número de melhores formigas a escolher.
            n_iterations (int): Número de iterações.
            decay (float): Taxa de evaporação de feromônios.
            alpha (int or float): Peso do feromônio.
            beta (int or float): Peso da distância.
        """
        self.distances  = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone * self.decay
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(i)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)

        return path

    def pick_move(self, pheromone, dists, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0.1

        row = (pheromone ** self.alpha) * (( 1.0 / dists) ** self.beta)

        points = [(i,p) for i,p in zip(self.all_inds, row) if i not in visited]
        row_aux = np.array([p[1] for p in points])
        points = [p[0] for p in points]

        norm_row = row_aux / row_aux.sum()
        norm_row = [(p if not np.isnan(p) else 0) for p in norm_row]
        if all([p==0 for p in norm_row]):
          norm_row = np.ones_like(norm_row) / len(norm_row)

        # move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        move = np.random.choice(points, 1, p=norm_row)[0]
        return move
pontos=dados.values
dist=0
distancia=np.zeros((len(pontos), len(pontos)))
for i in range(len(pontos)):
  for j in range(i+1, len(pontos)):
    dist=np.linalg.norm(pontos[i]-pontos[j])
    distancia[i, j]=dist
    distancia[j, i]=dist
ant_colony = AntColony(distancia, 3, 3, 100, 0.95, alpha=1, beta=2)
seq, shortest_path = ant_colony.run()
x=dados.x.values
y=dados.y.values
distancias=[]
for r, s in seq:
  distancias.append(((r, s), np.linalg.norm([x[r] - x[s], y[r] - y[s]])))

M=10
for m in range(M):
  max_dist=max(distancias, key=lambda i: i[1])
  distancias.remove(max_dist)

percurso=[]
for idx, d in distancias:
  r, s = idx
  percurso.append(s)
percurso=np.array(percurso)

x=dados.x.values[percurso]
y=dados.y.values[percurso]

plt.figure(figsize=(6,6))
plt.title('Entregas')
plt.xlabel('Coordenadas x')
plt.ylabel('Coordenadas y')
plt.scatter(x, y)
plt.plot(x, y)
plt.grid();
