import random
import math

# Objective function (to minimize)
def objective_function(position):
    x, y = position
    return x**2 + y**2

# Particle class
class Particle:
    def __init__(self, num_dimensions, bounds):
        self.position = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(num_dimensions)]
        self.velocity = [random.uniform(-1, 1) for _ in range(num_dimensions)]
        self.pbest = self.position[:]  # personal best position
        self.pbest_value = objective_function(self.position)

# PSO algorithm
def PSO(num_particles=30, num_dimensions=2, max_iterations=100, bounds=[(-10,10), (-10,10)]):
    w = 0.7   # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social

    # Initialize swarm
    swarm = [Particle(num_dimensions, bounds) for _ in range(num_particles)]
    gbest = swarm[0].position[:]
    gbest_value = objective_function(gbest)

    # Find initial global best
    for particle in swarm:
        if particle.pbest_value < gbest_value:
            gbest = particle.pbest[:]
            gbest_value = particle.pbest_value

    # Main loop
    for iteration in range(max_iterations):
        for particle in swarm:
            # Evaluate fitness
            fitness = objective_function(particle.position)

            # Update personal best
            if fitness < particle.pbest_value:
                particle.pbest = particle.position[:]
                particle.pbest_value = fitness

            # Update global best
            if fitness < gbest_value:
                gbest = particle.position[:]
                gbest_value = fitness

        # Update velocity and position
        for particle in swarm:
            for d in range(num_dimensions):
                r1 = random.random()
                r2 = random.random()

                # velocity update
                particle.velocity[d] = (w * particle.velocity[d] +
                                        c1 * r1 * (particle.pbest[d] - particle.position[d]) +
                                        c2 * r2 * (gbest[d] - particle.position[d]))
                # position update
                particle.position[d] += particle.velocity[d]

                # keep inside bounds
                particle.position[d] = max(bounds[d][0], min(bounds[d][1], particle.position[d]))

        # Print progress every 10 iterations
        if iteration % 10 == 0 or iteration == max_iterations - 1:
            print(f"Iteration {iteration+1}/{max_iterations}, Best Value: {gbest_value:.6f}")

    return gbest, gbest_value

# Run PSO
best_position, best_value = PSO()
print("\nBest solution found:", best_position)
print("Objective value at best solution:", best_value)
