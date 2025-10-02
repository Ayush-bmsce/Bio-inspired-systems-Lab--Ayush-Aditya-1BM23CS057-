import numpy as np
import random

class ACO_TSP:
    def __init__(self, dist_matrix, n_ants=10, n_iterations=100, alpha=1, beta=2, rho=0.5, Q=100):
        """
        dist_matrix: distance matrix (2D numpy array)
        n_ants: number of ants
        n_iterations: number of iterations
        alpha: influence of pheromone
        beta: influence of heuristic (1/distance)
        rho: pheromone evaporation rate
        Q: pheromone deposit factor
        """
        self.dist_matrix = dist_matrix
        self.n_cities = dist_matrix.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        # Initialize pheromone matrix
        self.pheromone = np.ones((self.n_cities, self.n_cities))

        # Best solution
        self.best_length = float("inf")
        self.best_path = None

    def run(self):
        for iteration in range(self.n_iterations):
            all_paths = self.construct_solutions()
            self.update_pheromones(all_paths)
            print(f"Iteration {iteration+1}/{self.n_iterations}, Best length so far: {self.best_length}")
        return self.best_path, self.best_length

    def construct_solutions(self):
        all_paths = []
        for ant in range(self.n_ants):
            path = self.build_path()
            length = self.path_length(path)
            all_paths.append((path, length))

            # Update global best
            if length < self.best_length:
                self.best_length = length
                self.best_path = path
        return all_paths

    def build_path(self):
        path = []
        visited = set()
        start = random.randint(0, self.n_cities - 1)
        path.append(start)
        visited.add(start)

        for _ in range(self.n_cities - 1):
            current_city = path[-1]
            next_city = self.choose_next_city(current_city, visited)
            path.append(next_city)
            visited.add(next_city)

        return path

    def choose_next_city(self, current_city, visited):
        pheromone = np.copy(self.pheromone[current_city])
        heuristic = 1 / (self.dist_matrix[current_city] + 1e-10)  # avoid div by 0

        # Zero out visited cities
        for city in visited:
            pheromone[city] = 0
            heuristic[city] = 0

        probabilities = (pheromone ** self.alpha) * (heuristic ** self.beta)
        probabilities = probabilities / probabilities.sum()

        return np.random.choice(range(self.n_cities), p=probabilities)

    def path_length(self, path):
        length = 0
        for i in range(len(path)):
            length += self.dist_matrix[path[i]][path[(i+1) % self.n_cities]]
        return length

    def update_pheromones(self, all_paths):
        # Evaporation
        self.pheromone *= (1 - self.rho)

        # Deposit new pheromone
        for path, length in all_paths:
            deposit_amount = self.Q / length
            for i in range(len(path)):
                a, b = path[i], path[(i+1) % self.n_cities]
                self.pheromone[a][b] += deposit_amount
                self.pheromone[b][a] += deposit_amount


# ------------------------
# Example Usage
# ------------------------
if __name__ == "__main__":
    # Distance matrix (symmetric for TSP)
    dist_matrix = np.array([
        [0, 2, 9, 10],
        [1, 0, 6, 4],
        [15, 7, 0, 8],
        [6, 3, 12, 0]
    ])

    aco = ACO_TSP(dist_matrix, n_ants=5, n_iterations=50, alpha=1, beta=2, rho=0.5, Q=100)
    best_path, best_length = aco.run()
    print("\nBest Path:", best_path)
    print("Best Path Length:", best_length)
