import numpy as np
import random
import math

# Define available functions and terminals for GEP
FUNCTIONS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b if b != 0 else 1,  # Safe division
    'sin': lambda a: math.sin(a),
    'cos': lambda a: math.cos(a)
}
TERMINALS = ['x']

# Parameters
HEAD_LENGTH = 5
POP_SIZE = 30
GENERATIONS = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.4

# Target function for symbolic regression: f(x) = x^2 + x + 1
def target_function(x):
    return x**2 + x + 1

# Generate random gene
def random_gene():
    gene = []
    for i in range(HEAD_LENGTH):
        if random.random() < 0.5:
            gene.append(random.choice(list(FUNCTIONS.keys())))
        else:
            gene.append(random.choice(TERMINALS))
    return gene

# Decode gene into expression tree (simple left-to-right evaluation)
def evaluate_gene(gene, x):
    stack = []
    for symbol in gene:
        if symbol in FUNCTIONS:
            func = FUNCTIONS[symbol]
            if symbol in ['+', '-', '*', '/']:
                if len(stack) < 2:
                    stack.append(x)  # fill missing operands
                b, a = stack.pop(), stack.pop()
                stack.append(func(a, b))
            else:  # unary functions
                if len(stack) < 1:
                    stack.append(x)
                a = stack.pop()
                stack.append(func(a))
        else:
            stack.append(x)
    return stack[-1] if stack else x

# Fitness function (MSE)
def fitness(gene):
    xs = np.linspace(-5, 5, 50)
    predicted = [evaluate_gene(gene, x) for x in xs]
    actual = [target_function(x) for x in xs]
    mse = np.mean((np.array(predicted) - np.array(actual))**2)
    return -mse  # Maximize fitness (minimize error)

# Mutation
def mutate(gene):
    new_gene = gene.copy()
    for i in range(len(new_gene)):
        if random.random() < MUTATION_RATE:
            if random.random() < 0.5:
                new_gene[i] = random.choice(list(FUNCTIONS.keys()))
            else:
                new_gene[i] = random.choice(TERMINALS)
    return new_gene

# Crossover
def crossover(g1, g2):
    if random.random() > CROSSOVER_RATE:
        return g1.copy(), g2.copy()
    point = random.randint(1, len(g1)-1)
    return g1[:point] + g2[point:], g2[:point] + g1[point:]

# GEP Algorithm
def gep():
    population = [random_gene() for _ in range(POP_SIZE)]
    for gen in range(GENERATIONS):
        fitness_scores = [fitness(g) for g in population]
        best_idx = np.argmax(fitness_scores)
        best_gene = population[best_idx]
        print(f"Gen {gen+1}: Best Fitness = {fitness_scores[best_idx]:.6f}, Gene = {best_gene}")

        # Selection (tournament)
        new_pop = []
        while len(new_pop) < POP_SIZE:
            i, j = random.sample(range(POP_SIZE), 2)
            winner = population[i] if fitness_scores[i] > fitness_scores[j] else population[j]
            new_pop.append(winner)

        # Apply crossover and mutation
        next_gen = []
        for i in range(0, POP_SIZE, 2):
            g1, g2 = crossover(new_pop[i], new_pop[min(i+1, POP_SIZE-1)])
            next_gen.append(mutate(g1))
            next_gen.append(mutate(g2))
        population = next_gen[:POP_SIZE]

    return best_gene

if __name__ == "__main__":
    best_gene = gep()
    print("Best gene found:", best_gene)
