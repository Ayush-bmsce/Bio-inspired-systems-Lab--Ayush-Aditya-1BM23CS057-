import numpy as np
import random

class CuckooSearch:
    def __init__(self, objective_func, bounds, n_nests=25, pa=0.25, max_iter=1000):
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_nests = n_nests
        self.pa = pa
        self.max_iter = max_iter
        self.dim = len(bounds)
        
        # Initialize nests
        self.nests = np.zeros((n_nests, self.dim))
        self.fitness = np.zeros(n_nests)
        
        # Best solution tracking
        self.best_nest = None
        self.best_fitness = float('inf')
        
    def initialize_nests(self):
        """Initialize nests with random positions within bounds"""
        for i in range(self.n_nests):
            for j in range(self.dim):
                lower, upper = self.bounds[j]
                self.nests[i, j] = lower + (upper - lower) * random.random()
            self.fitness[i] = self.objective_func(self.nests[i])
            
        # Find initial best
        best_idx = np.argmin(self.fitness)
        self.best_nest = self.nests[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
    
    def levy_flight(self, beta=1.5):
        """
        Generate step size using Levy flight
        """
        # Generate random direction
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        
        # Calculate step size using Levy distribution
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        
        step = 0.01 * (u / (np.abs(v) ** (1 / beta))) * sigma
        
        return step
    
    def generate_new_solution(self, nest_idx):
        """Generate new solution using Levy flight"""
        step = self.levy_flight()
        new_nest = self.nests[nest_idx].copy()
        
        for j in range(self.dim):
            lower, upper = self.bounds[j]
            new_nest[j] += step[j]
            # Boundary check
            new_nest[j] = np.clip(new_nest[j], lower, upper)
            
        return new_nest
    
    def abandon_worst_nests(self):
        """Abandon worst nests and build new ones"""
        # Sort nests by fitness
        sorted_indices = np.argsort(self.fitness)
        num_abandon = int(self.n_nests * self.pa)
        
        # Abandon worst nests
        for i in range(num_abandon):
            idx = sorted_indices[-(i + 1)]
            for j in range(self.dim):
                lower, upper = self.bounds[j]
                self.nests[idx, j] = lower + (upper - lower) * random.random()
            self.fitness[idx] = self.objective_func(self.nests[idx])
    
    def optimize(self):
        """Main optimization loop"""
        self.initialize_nests()
        
        print(f"Initial best fitness: {self.best_fitness}")
        
        for iteration in range(self.max_iter):
            # Generate new solutions using Levy flight
            for i in range(self.n_nests):
                # Get a cuckoo randomly by Levy flight
                new_solution = self.generate_new_solution(i)
                new_fitness = self.objective_func(new_solution)
                
                # Choose a random nest
                j = random.randint(0, self.n_nests - 1)
                
                # If new solution is better, replace it
                if new_fitness < self.fitness[j]:
                    self.nests[j] = new_solution
                    self.fitness[j] = new_fitness
                    
                    # Update global best
                    if new_fitness < self.best_fitness:
                        self.best_nest = new_solution.copy()
                        self.best_fitness = new_fitness
            
            # Abandon worst nests
            self.abandon_worst_nests()
            
            # Keep the best solution
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_nest = self.nests[current_best_idx].copy()
                self.best_fitness = self.fitness[current_best_idx]
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Best fitness: {self.best_fitness}")
        
        print(f"Final best fitness: {self.best_fitness}")
        print(f"Best solution: {self.best_nest}")
        
        return self.best_nest, self.best_fitness

# Example usage: Minimize f(x) = x^2 (as shown in the PDF)
def sphere_function(x):
    """Example objective function: f(x) = x^2"""
    return np.sum(x**2)

# Test with the example from the PDF (1D problem)
if __name__ == "__main__":
    # Define bounds for 1D problem
    bounds = [(-10, 10)]
    
    # Create Cuckoo Search instance
    cs = CuckooSearch(sphere_function, bounds, n_nests=15, pa=0.25, max_iter=500)
    
    # Run optimization
    best_solution, best_fitness = cs.optimize()
    
    print("\n" + "="*50)
    print("CUCKOO SEARCH ALGORITHM RESULTS")
    print("="*50)
    print(f"Global minimum found at: x = {best_solution[0]:.6f}")
    print(f"Function value: f(x) = {best_fitness:.6f}")
    print(f"Expected: x = 0.0, f(x) = 0.0")

# Additional example: 2D Rosenbrock function
def rosenbrock_function(x):
    """Rosenbrock function - common test function for optimization"""
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Test with 2D problem
print("\n" + "="*50)
print("TESTING WITH ROSENBROCK FUNCTION (2D)")
print("="*50)

bounds_2d = [(-2, 2), (-2, 2)]
cs_2d = CuckooSearch(rosenbrock_function, bounds_2d, n_nests=20, pa=0.25, max_iter=1000)
best_solution_2d, best_fitness_2d = cs_2d.optimize()

print(f"Best solution: {best_solution_2d}")
print(f"Best fitness: {best_fitness_2d:.6f}")
