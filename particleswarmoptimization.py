import numpy as np
import random

# Objective function
def objective_function(x):
    return -x**2 + 5*x + 20   # maximize this

class PSO:
    def __init__(self, n_particles=30, n_iterations=50, bounds=(-10, 10),
                 w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.bounds = bounds
        self.w = w        # inertia
        self.c1 = c1      # personal influence
        self.c2 = c2      # social influence

        # Initialize particles
        self.positions = np.random.uniform(bounds[0], bounds[1], n_particles)
        self.velocities = np.zeros(n_particles)

        # Personal bests
        self.pbest_positions = np.copy(self.positions)
        self.pbest_values = np.array([objective_function(x) for x in self.positions])

        # Global best
        best_idx = np.argmax(self.pbest_values)
        self.gbest_position = self.pbest_positions[best_idx]
        self.gbest_value = self.pbest_values[best_idx]

    def optimize(self):
        for t in range(self.n_iterations):
            for i in range(self.n_particles):
                r1, r2 = random.random(), random.random()

                # Update velocity
                inertia = self.w * self.velocities[i]
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])
                self.velocities[i] = inertia + cognitive + social

                # Update position
                self.positions[i] += self.velocities[i]

                # Clamp position inside bounds
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])

                # Evaluate
                value = objective_function(self.positions[i])

                # Update personal best
                if value > self.pbest_values[i]:
                    self.pbest_positions[i] = self.positions[i]
                    self.pbest_values[i] = value

                    # Update global best
                    if value > self.gbest_value:
                        self.gbest_position = self.positions[i]
                        self.gbest_value = value

            print(f"Iteration {t+1}/{self.n_iterations}, Best = {self.gbest_value:.4f} at x = {self.gbest_position:.4f}")

        return self.gbest_position, self.gbest_value


# ------------------------
# Example Usage
# ------------------------
if __name__ == "__main__":
    pso = PSO(n_particles=20, n_iterations=50, bounds=(-10, 10), w=0.7, c1=1.5, c2=1.5)
    best_x, best_val = pso.optimize()
    print("\nBest Solution Found:")
    print(f"x = {best_x:.4f}, f(x) = {best_val:.4f}")
