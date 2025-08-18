import random
import math

# Function to optimize
def fitness_function(x):
    return x * math.sin(10 * math.pi * x) + 1.0

# Parameters
POP_SIZE = 10
GENS = 20
CROSS_RATE = 0.8
MUTATE_RATE = 0.1
X_BOUND = [0, 1]

# Generate random initial population
population = [random.uniform(*X_BOUND) for _ in range(POP_SIZE)]

# Main GA loop
for gen in range(GENS):
    # Evaluate fitness for each individual
    fitness = [fitness_function(ind) for ind in population]

    # Track the best solution
    best_idx = fitness.index(max(fitness))
    best_x = population[best_idx]
    best_score = fitness[best_idx]
    print(f"Gen {gen+1}: Best x = {best_x:.4f}, f(x) = {best_score:.4f}")

    # Selection (Roulette Wheel)
    total_fitness = sum(fitness)
    probs = [f / total_fitness for f in fitness]
    selected = random.choices(population, weights=probs, k=POP_SIZE)

    # Crossover
    children = []
    for i in range(0, POP_SIZE, 2):
        p1, p2 = selected[i], selected[(i+1) % POP_SIZE]
        if random.random() < CROSS_RATE:
            alpha = random.random()
            c1 = alpha * p1 + (1 - alpha) * p2
            c2 = alpha * p2 + (1 - alpha) * p1
        else:
            c1, c2 = p1, p2
        children.extend([c1, c2])

    # Mutation
    for i in range(len(children)):
        if random.random() < MUTATE_RATE:
            children[i] += random.uniform(-0.1, 0.1)
            children[i] = max(min(children[i], X_BOUND[1]), X_BOUND[0])  # keep in bounds

    # Update population
    population = children

print(f"\nBest solution found: x = {best_x:.4f}, f(x) = {best_score:.4f}")
