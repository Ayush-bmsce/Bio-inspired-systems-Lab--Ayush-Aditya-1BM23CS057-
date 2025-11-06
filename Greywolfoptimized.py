import numpy as np 
 
# Objective function (example: Sphere function) 
def objective_function(x): 
    return np.sum(x**2) 
N, dim, T = 30, 10, 100  # Number of wolves, dimensions, iterations 
lower_bound, upper_bound = -10, 10 
 
wolves = np.random.uniform(lower_bound, upper_bound, (N, dim)) 
 
alpha_pos, beta_pos, delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim) 
alpha_score, beta_score, delta_score = float('inf'), float('inf'), 
float('inf') 
for t in range(T): 
    for i in range(N): 
        fitness = objective_function(wolves[i])  # Evaluate fitness 
        if fitness < alpha_score: 
            delta_score, delta_pos = beta_score, beta_pos.copy() 
            beta_score, beta_pos = alpha_score, alpha_pos.copy() 
            alpha_score, alpha_pos = fitness, wolves[i].copy() 
        elif fitness < beta_score: 
            delta_score, delta_pos = beta_score, beta_pos.copy() 
            beta_score, beta_pos = fitness, wolves[i].copy() 
        elif fitness < delta_score: 
            delta_score, delta_pos = fitness, wolves[i].copy() 
    a = 2 - t * (2 / T) 
    for i in range(N): 
        r1, r2 = np.random.rand(dim), np.random.rand(dim) 
        A, C = 2 * a * r1 - a, 2 * r2 
        wolves[i] += A * (abs(C * alpha_pos - wolves[i]) + 
                          abs(C * beta_pos - wolves[i]) + 
                          abs(C * delta_pos - wolves[i])) 
 
        wolves[i] = np.clip(wolves[i], lower_bound, upper_bound) 
print("Best Solution:", alpha_pos) 
print("Best Score:", alpha_score) 
