import numpy as np 
import random 
import concurrent.futures 
 
def rastrigin(x): 
    A = 10 
    return A * len(x) + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in 
x]) 
 
GRID_SIZE = (10, 10) 
DIM = 2 
RADIUS = 1 
ITER = 100 
BEST = None 
 
def init_grid(size, dim): 
    return [[np.random.uniform(-5.12, 5.12, size=(dim,)) for _ in 
range(size[1])] for _ in range(size[0])] 
 
def fitness(cell): 
    return rastrigin(cell) 
 
def update_state(grid, i, j, radius): 
    curr = grid[i][j] 
    fitness_curr = fitness(curr) 
    neighbors = [grid[ni][nj] for dx in range(-radius, radius+1) for dy in 
range(-radius, radius+1) 
                 if 0 <= (ni := i+dx) < len(grid) and 0 <= (nj := j+dy) < 
len(grid[0]) and (dx or dy)] 
    if neighbors: 
        best_neigh = min(neighbors, key=fitness) 
        return curr + 0.1 * (best_neigh - curr) 
    return curr 
 
def run_iteration(grid, radius): 
    new_grid = [[None for _ in range(len(grid[0]))] for _ in 
range(len(grid))] 
    with concurrent.futures.ThreadPoolExecutor() as ex: 
        futures = [ex.submit(update_state, grid, i, j, radius) for i in 
range(len(grid)) for j in range(len(grid[0]))] 
        for idx, future in enumerate(futures): 
            i, j = divmod(idx, len(grid[0])) 
            new_grid[i][j] = future.result() 
    return new_grid



def track_best(grid): 
    global BEST 
    best_cell, best_fitness = None, float('inf') 
    for row in grid: 
        for cell in row: 
            f = fitness(cell) 
            if f < best_fitness: 
                best_fitness = f 
                best_cell = cell 
    if BEST is None or best_fitness < fitness(BEST): 
        BEST = best_cell 
 
def parallel_cellular_algorithm(): 
    global BEST 
    grid = init_grid(GRID_SIZE, DIM) 
    for _ in range(ITER): 
        grid = run_iteration(grid, RADIUS) 
        track_best(grid) 
        print(f"Best Fitness: {fitness(BEST)}") 
    print("Best Solution:", BEST) 
    print("Best Fitness:", fitness(BEST)) 
 
parallel_cellular_algorithm() 



