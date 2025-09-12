import numpy as np
import random

# Objective Function: f(x) = 2x - sin(x)
def objective_function(x):
    return 2 * x - np.sin(x)

# Parameters
population_size = 10  # Population size (updated to 10)
mutation_rate = 0.15  # Mutation rate (updated to 0.15)
crossover_rate = 0.15  # Crossover rate (updated to 0.15)
num_generations = 5  # Number of generations (updated to 5)
lower_bound = -5  # Lower bound of the search space
upper_bound = 5   # Upper bound of the search space

# Initialize Population (Random sequences for x in the given range)
def initialize_population(pop_size, lower_bound, upper_bound):
    population = []
    for _ in range(pop_size):
        individual = random.uniform(lower_bound, upper_bound)  # Random float in the range
        population.append(individual)
    return population

# Evaluate Fitness
def evaluate_fitness(population):
    fitness_values = []
    for individual in population:
        fitness = objective_function(individual)
        fitness_values.append(fitness)
    return fitness_values

# Selection (Roulette Wheel Selection)
def select_parents(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]
    
    selected_parents = np.random.choice(population, size=2, p=probabilities)
    return selected_parents

# Crossover (Single-point crossover)
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        # For one gene, just take the average as the crossover
        child1 = (parent1 + parent2) / 2
        child2 = (parent1 + parent2) / 2
        return child1, child2
    return parent1, parent2

# Mutation (Uniform mutation)
def mutate(individual, mutation_rate, lower_bound, upper_bound):
    if random.random() < mutation_rate:
        mutation_value = random.uniform(lower_bound, upper_bound)
        individual = mutation_value
    return individual

# Main Genetic Algorithm
def genetic_algorithm():
    population = initialize_population(population_size, lower_bound, upper_bound)
    best_solution = None
    best_fitness = -float('inf')

    for generation in range(num_generations):
        fitness_values = evaluate_fitness(population)
        
        # Track the best solution
        max_fitness_idx = np.argmax(fitness_values)
        if fitness_values[max_fitness_idx] > best_fitness:
            best_fitness = fitness_values[max_fitness_idx]
            best_solution = population[max_fitness_idx]

        # Selection
        selected_parents = select_parents(population, fitness_values)
        
        # Crossover and Mutation
        new_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = selected_parents
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, lower_bound, upper_bound)
            child2 = mutate(child2, mutation_rate, lower_bound, upper_bound)
            new_population.extend([child1, child2])
        
        # Update population
        population = new_population

    return best_solution, best_fitness

# Run the algorithm
best_solution, best_fitness = genetic_algorithm()

# Output the best solution found
print("Best solution (x value):", best_solution)
print("Maximized fitness (f(x) value):", best_fitness)

