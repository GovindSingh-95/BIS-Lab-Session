import numpy as np
import matplotlib.pyplot as plt

# Grey Wolf Optimizer (GWO)
class GWO:
    def __init__(self, func, dim, lb, ub, n_wolves=20, max_iter=100):
        self.func = func      # objective function
        self.dim = dim        # problem dimensions
        self.lb = lb          # lower bound
        self.ub = ub          # upper bound
        self.n_wolves = n_wolves
        self.max_iter = max_iter

    def optimize(self):
        # Initialize wolves randomly
        wolves = np.random.uniform(self.lb, self.ub, (self.n_wolves, self.dim))
        fitness = np.apply_along_axis(self.func, 1, wolves)

        # Initialize alpha, beta, delta
        alpha, beta, delta = np.zeros(self.dim), np.zeros(self.dim), np.zeros(self.dim)
        f_alpha, f_beta, f_delta = float("inf"), float("inf"), float("inf")

        convergence_curve = []

        for t in range(self.max_iter):
            for i in range(self.n_wolves):
                score = fitness[i]

                # Update alpha, beta, delta
                if score < f_alpha:
                    f_alpha, alpha = score, wolves[i].copy()
                elif score < f_beta:
                    f_beta, beta = score, wolves[i].copy()
                elif score < f_delta:
                    f_delta, delta = score, wolves[i].copy()

            # coefficient decreasing from 2 -> 0
            a = 2 - t * (2 / self.max_iter)

            # Update position of wolves
            for i in range(self.n_wolves):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()

                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[j] - wolves[i][j])
                    X1 = alpha[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[j] - wolves[i][j])
                    X2 = beta[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[j] - wolves[i][j])
                    X3 = delta[j] - A3 * D_delta

                    # Average the 3 influences
                    wolves[i][j] = (X1 + X2 + X3) / 3

                # Keep wolves inside bounds
                wolves[i] = np.clip(wolves[i], self.lb, self.ub)
                fitness[i] = self.func(wolves[i])

            convergence_curve.append(f_alpha)

        return alpha, f_alpha, convergence_curve


# Test on Sphere function (minimum at [0,0,...,0])
def sphere(x):
    return sum(x**2)

# Run GWO
gwo = GWO(func=sphere, dim=2, lb=-10, ub=10, n_wolves=30, max_iter=50)
best_pos, best_score, curve = gwo.optimize()

print("Best Position Found:", best_pos)
print("Best Score (Fitness):", best_score)

# Plot convergence
plt.plot(curve)
plt.title("GWO Convergence Curve")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.show()
