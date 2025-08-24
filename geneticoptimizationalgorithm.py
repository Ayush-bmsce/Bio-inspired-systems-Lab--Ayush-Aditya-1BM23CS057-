import random

# --- Problem Setup ---
def fitness_function(individual, w=5):
    """Objective function f(x) = x^2 - w*x + 3
       We take 'height' as x just for demo.
    """
    x = individual["height"]
    return -(x**2 - w*x + 3)  # Negative for maximization

# --- GA Setup ---
POP_SIZE = 20
N_GEN = 10
MUTATION_RATE = 0.1

TRAITS = ["skin_tone", "height", "face", "immune_system", "allergies"]

def random_individual():
    """Create random individual with 5 traits between [0, 10]."""
    return {trait: random.uniform(0, 10) for trait in TRAITS}

def crossover(parent1, parent2):
    """Single-point crossover on traits."""
    child = {}
    for trait in TRAITS:
        child[trait] = random.choice([parent1[trait], parent2[trait]])
    return child

def mutate(individual):
    """Random mutation on traits."""
    for trait in TRAITS:
        if random.random() < MUTATION_RATE:
            individual[trait] = random.uniform(0, 10)
    return individual

def evaluate_population(pop):
    """Return list of (fitness, individual)."""
    return [(fitness_function(ind), ind) for ind in pop]

# --- Genetic Algorithm ---
population = [random_individual() for _ in range(POP_SIZE)]

for gen in range(N_GEN):
    evaluated = evaluate_population(population)
    evaluated.sort(reverse=True, key=lambda x: x[0])

    # Select top 10 parents
    parents = [ind for _, ind in evaluated[:10]]

    # Reproduce offspring only
    offspring = []
    for i in range(len(parents)//2):
        p1, p2 = parents[2*i], parents[2*i+1]
        child1 = mutate(crossover(p1, p2))
        child2 = mutate(crossover(p2, p1))
        offspring.extend([child1, child2])

    # Replace old population fully with new offspring
    population = offspring
    while len(population) < POP_SIZE:
        population.append(random_individual())

    # Display ONLY offspring population
    print(f"\n=== Generation {gen+1} Offspring Population ===")
    for ind in population:
        fit = fitness_function(ind)
        print(f"Fitness={fit:.2f} | {ind}")
