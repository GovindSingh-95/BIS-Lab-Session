import numpy as np

# Objective (fitness) function - Example: Sphere function (minimize sum of squares)
def objective_function(x):
    return np.sum(x**2)

# Levy flight step generation
def levy_flight(Lambda, dimension):
    sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (np.math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
    u = np.random.randn(dimension) * sigma
    v = np.random.randn(dimension)
    step = u / abs(v)**(1 / Lambda)
    return step

# Cuckoo Search Algorithm
def cuckoo_search(obj_func, n=25, d=2, lb=-5, ub=5, pa=0.25, alpha=0.01, max_iter=1000):
    """
    n  : number of nests (population size)
    d  : number of dimensions
    lb : lower bound
    ub : upper bound
    pa : probability of discovering a worse nest
    alpha : step size scaling factor
    max_iter : maximum number of iterations
    """

    # Initialize nests randomly
    nests = np.random.uniform(lb, ub, size=(n, d))
    fitness = np.array([obj_func(x) for x in nests])

    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]

    for t in range(max_iter):
        # Generate new solutions (cuckoos) via Levy flights
        for i in range(n):
            step = levy_flight(1.5, d)
            new_nest = nests[i] + alpha * step * (nests[i] - best_nest)
            new_nest = np.clip(new_nest, lb, ub)
            new_fitness = obj_func(new_nest)

            # Greedy selection: Replace if better
            if new_fitness < fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness

                # Update global best
                if new_fitness < best_fitness:
                    best_nest = new_nest.copy()
                    best_fitness = new_fitness

        # Abandon a fraction of worse nests and replace with new ones
        abandon_mask = np.random.rand(n, d) > pa
        step_size = np.random.rand(n, d) * (nests[np.random.permutation(n)] - nests[np.random.permutation(n)])
        new_nests = nests + step_size * abandon_mask
        new_nests = np.clip(new_nests, lb, ub)

        new_fitness = np.array([obj_func(x) for x in new_nests])
        improve_idx = new_fitness < fitness
        nests[improve_idx] = new_nests[improve_idx]
        fitness[improve_idx] = new_fitness[improve_idx]

        # Update best solution again
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_nest = nests[current_best_idx].copy()
            best_fitness = fitness[current_best_idx]

    return best_nest, best_fitness


if __name__ == "__main__":
    # Example usage: minimize Sphere function
    best_solution, best_value = cuckoo_search(objective_function, n=25, d=5, lb=-10, ub=10, max_iter=500)
    print("Best solution:", best_solution)
    print("Best fitness value:", best_value)
