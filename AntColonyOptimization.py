import random
import math

# ----------------------------
# Distance function
def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# ----------------------------
# ACO for TSP
def ACO_TSP(cities, num_ants=10, alpha=1.0, beta=5.0, rho=0.5, Q=100, max_iterations=100):
    num_cities = len(cities)

    # Distance matrix
    dist_matrix = [[distance(cities[i], cities[j]) for j in range(num_cities)] for i in range(num_cities)]

    # Initialize pheromones
    pheromone = [[0.1 for j in range(num_cities)] for i in range(num_cities)]

    best_tour = None
    best_length = float("inf")

    for iteration in range(max_iterations):
        all_tours = []
        all_lengths = []

        for ant in range(num_ants):
            # Start from a random city
            start = random.randint(0, num_cities - 1)
            tour = [start]
            unvisited = set(range(num_cities)) - {start}

            # Build tour
            while unvisited:
                current = tour[-1]
                probabilities = []
                total_prob = 0.0

                for next_city in unvisited:
                    tau = pheromone[current][next_city] ** alpha
                    eta = (1.0 / dist_matrix[current][next_city]) ** beta
                    prob = tau * eta
                    probabilities.append((next_city, prob))
                    total_prob += prob

                # Roulette wheel selection
                r = random.random() * total_prob
                cumulative = 0.0
                for city, prob in probabilities:
                    cumulative += prob
                    if cumulative >= r:
                        next_city = city
                        break

                tour.append(next_city)
                unvisited.remove(next_city)

            # Return to start
            tour.append(start)

            # Calculate length
            length = sum(dist_matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1))

            all_tours.append(tour)
            all_lengths.append(length)

            # Update best
            if length < best_length:
                best_length = length
                best_tour = tour

        # Pheromone evaporation
        for i in range(num_cities):
            for j in range(num_cities):
                pheromone[i][j] *= (1 - rho)

        # Pheromone deposit
        for k in range(num_ants):
            tour = all_tours[k]
            length = all_lengths[k]
            deposit = Q / length
            for i in range(len(tour)-1):
                a, b = tour[i], tour[i+1]
                pheromone[a][b] += deposit
                pheromone[b][a] += deposit  # symmetric TSP

        # Print progress
        if iteration % 10 == 0 or iteration == max_iterations - 1:
            print(f"Iteration {iteration+1}/{max_iterations}, Best Length: {best_length:.4f}")

    return best_tour, best_length

# ----------------------------
# Example usage
if __name__ == "__main__":
    # Example cities (coordinates)
    cities = [(0,0), (1,5), (5,2), (6,6), (8,3), (7,9), (2,7)]
    best_tour, best_length = ACO_TSP(cities, num_ants=20, max_iterations=50)

    print("\nBest tour found:", best_tour)
    print("Best length:", best_length)
