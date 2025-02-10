import numpy as np
import matplotlib.pyplot as plt

# Constants (set to 1 for simplicity)
hbar = 1  # Reduced Planck's constant
m = 1     # Mass of particle
L = 1     # Width of the well
N = 100   # Number of points for discretization
dx = L / (N + 1)  # Grid spacing

# Define the potential (Infinite Square Well -> V=0 inside, V=infinity outside)
V = np.zeros(N)  # V(x) = 0 inside the well

# Create the Hamiltonian matrix
H = np.zeros((N, N))
for i in range(N):
    H[i, i] = 2.0
    if i > 0:
        H[i, i - 1] = -1.0
    if i < N - 1:
        H[i, i + 1] = -1.0

# Apply finite difference approximation: H = - (hbar^2 / 2m) * d^2/dx^2
H *= - (hbar**2 / (2 * m * dx**2))

# Solve the eigenvalue problem
E, psi = np.linalg.eigh(H)  # Eigenvalues (E) and Eigenvectors (psi)

# Normalize wavefunctions
psi /= np.sqrt(dx)

# Plot the first few wavefunctions
x = np.linspace(0, L, N + 2)  # Extend x to include boundaries at 0 and L
psi_full = np.zeros((N + 2, N))
psi_full[1:-1, :] = psi

# Plot the first few wavefunctions with boundaries
num_states = 4  # Number of states to plot

plt.figure(figsize=(10, 8))
for i in range(num_states):
    plt.plot(x, psi_full[:, i], label=f'n={i+1}, E={E[i]:.3f}')

plt.xlabel("Position (x)")
plt.ylabel("Wavefunction")
plt.title("Wavefunctions for Infinite Square Well")
plt.legend()
plt.grid(True)
plt.show()
num_states = 4  # Number of states to plot

plt.figure(figsize=(8, 6))
for i in range(num_states):
    plt.plot(x, psi_full[:, i], label=f'n={i+1}, E={E[i]:.3f}')
plt.xlabel("Position (x)")
plt.ylabel("Wavefunction")
plt.title("Wavefunctions for Infinite Square Well")
plt.legend()
plt.grid()
plt.show()

# Print first few energy eigenvalues
print("First few energy eigenvalues:")
for i in range(num_states):
    print(f'n={i+1}, E={E[i]:.5f}')
    # Print the first 10 energy eigenvalues
    num_states = 10  # Number of states to print

    print("First 10 energy eigenvalues:")
    for i in range(num_states):
        print(f'n={i+1}, E={E[i]:.5f}')