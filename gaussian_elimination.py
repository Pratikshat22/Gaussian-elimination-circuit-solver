import numpy as np

# ============================================
# PART 1: Implement the augmented matrix C
# ============================================

# Define the coefficient matrix A (resistance matrix)
A = np.array([[10, -2, -3],
              [-2, 12, -5],
              [-3, -5, 15]])

# Define the voltage source vector b
b = np.array([20, 0, -10])

# Get the size of the system
n = len(b)

# PART 1: Pre-allocate an n × (n+1) matrix of zeros
C = np.zeros((n, n + 1))

# Fill the augmented matrix [A|b]
C[:, 0:n] = A      # First n columns are matrix A
C[:, n] = b        # Last column is vector b

print("=" * 60)
print("TASK 6.1: KIRCHHOFF'S CIRCUIT ANALYSIS")
print("=" * 60)
print("\nAugmented Matrix [A|b]:")
print(C)


# ============================================
# PART 2: Gaussian Elimination Function
# ============================================

def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting
    """
    n = len(b)
    
    # Create augmented matrix
    Ab = np.zeros((n, n + 1))
    Ab[:, 0:n] = A
    Ab[:, n] = b
    
    # Forward elimination with partial pivoting
    for k in range(n - 1):
        # Find pivot row (largest element in column k)
        pivot_row = k
        max_val = abs(Ab[k, k])
        for i in range(k + 1, n):
            if abs(Ab[i, k]) > max_val:
                max_val = abs(Ab[i, k])
                pivot_row = i
        
        # Swap rows if necessary
        if pivot_row != k:
            Ab[[k, pivot_row]] = Ab[[pivot_row, k]]
            print(f"Swapped row {k} with row {pivot_row}")
        
        # Eliminate below
        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] = Ab[i, k:] - factor * Ab[k, k:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x

# Solve the circuit using my function
print("\n" + "-" * 60)
print("SOLVING FOR CURRENTS:")
print("-" * 60)

currents = gaussian_elimination(A, b)

print(f"\nI₁ = {currents[0]:.4f} Amperes")
print(f"I₂ = {currents[1]:.4f} Amperes")
print(f"I₃ = {currents[2]:.4f} Amperes")


# ============================================
# PART 3: Verify using NumPy's @ operator
# ============================================

print("\n" + "-" * 60)
print("VERIFICATION (Using @ operator):")
print("-" * 60)

# Calculate A @ x
Ax = A @ currents

print("\nA @ I =", Ax)
print("b      =", b)

# Check if they are equal (within floating point error)
verification = np.allclose(Ax, b)
print(f"\nDoes A·I = b? {verification}")

if verification:
    print("Solution is correct!")
else:
    print("Something went wrong!")

# Calculate the error
error = Ax - b
print(f"\nError (A·I - b) = {error}")


# 6.2 : Stability Analysis

def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting
    """
    n = len(b)

    # Create augmented matrix
    Ab = np.zeros((n, n + 1))
    Ab[:, 0:n] = A
    Ab[:, n] = b

    # Forward elimination with partial pivoting
    for k in range(n - 1):
        # Find pivot row (largest element in column k)
        pivot_row = k
        max_val = abs(Ab[k, k])
        for i in range(k + 1, n):
            if abs(Ab[i, k]) > max_val:
                max_val = abs(Ab[i, k])
                pivot_row = i

        # Swap rows if necessary
        if pivot_row != k:
            Ab[[k, pivot_row]] = Ab[[pivot_row, k]]
            # print(f"Swapped row {k} with row {pivot_row}") # Commented out to reduce output

        # Eliminate below
        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] = Ab[i, k:] - factor * Ab[k, k:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

    return x


# ============================================
# TASK 6.2: STABILITY ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("TASK 6.2: STABILITY ANALYSIS")
print("=" * 60)

# Define the near-singular system
A_stable = np.array([[1.0001, 1.0000],
                      [1.0000, 1.0000]])

print("\nSystem:")
print("1.0001x₁ + 1.0000x₂ = b₁")
print("1.0000x₁ + 1.0000x₂ = b₂")
print("\nMatrix A:")
print(A_stable)

# Calculate condition number (shows how ill-conditioned the system is)
cond_number = np.linalg.cond(A_stable)
print(f"\nCondition Number of A: {cond_number:.2f}")
print("(Large condition number means system is ill-conditioned)")

# ============================================
# Case 1: Original b vector
# ============================================
print("\n" + "-" * 40)
print("CASE 1: Original System")
print("-" * 40)

b1 = np.array([2.0001, 2.0000])
print(f"b = [{b1[0]:.4f}, {b1[1]:.4f}]")

# Solve using my function
x1 = gaussian_elimination(A_stable, b1)
print(f"Solution: x₁ = {x1[0]:.4f}, x₂ = {x1[1]:.4f}")

# Verify
print(f"Verification A·x = {A_stable @ x1}")
print(f"Original b      = {b1}")

# ============================================
# Case 2: Slightly changed b vector
# ============================================
print("\n" + "-" * 40)
print("CASE 2: Changed System (b₁ changed by 0.0001)")
print("-" * 40)

b2 = np.array([2.0002, 2.0000])
print(f"b = [{b2[0]:.4f}, {b2[1]:.4f}]")
print(f"Change in b₁: +0.0001 (0.005% change)")

# Solve using my function
x2 = gaussian_elimination(A_stable, b2)
print(f"Solution: x₁ = {x2[0]:.4f}, x₂ = {x2[1]:.4f}")

# Verify
print(f"Verification A·x = {A_stable @ x2}")
print(f"Original b      = {b2}")

# ============================================
# Compare the results
# ============================================
print("\n" + "-" * 40)
print("COMPARISON")
print("-" * 40)

print(f"Case 1 solution: x₁ = {x1[0]:.4f}, x₂ = {x1[1]:.4f}")
print(f"Case 2 solution: x₁ = {x2[0]:.4f}, x₂ = {x2[1]:.4f}")
print(f"\nChange in x₁: {x2[0]-x1[0]:.2f} ({abs(x2[0]-x1[0])/abs(x1[0])*100:.1f}% change)")
print(f"Change in x₂: {x2[1]-x1[1]:.2f} ({abs(x2[1]-x1[1])/abs(x1[1])*100:.1f}% change)")

# ============================================
# DISCUSSION
# ============================================
print("\n" + "-" * 40)
print("DISCUSSION")
print("-" * 40)

print("""
Why does the result change so drastically?

1. The two equations are almost identical:
   Eq1: 1.0001x₁ + 1.0000x₂ = b₁
   Eq2: 1.0000x₁ + 1.0000x₂ = b₂

   The coefficients differ by only 0.0001 in the first term.

2. Geometrically, these represent two nearly parallel lines.
   A tiny shift in one line completely changes their intersection point.

3. The condition number (~40,000) tells us that errors are amplified
   by a factor of 40,000. So a 0.005% input change becomes a 200%
   output change.

Relation to experimental uncertainty in physics:

In real experiments, we never measure quantities exactly. If our
voltmeters have even 0.1% uncertainty, and our system is this
ill-conditioned, our calculated currents could be completely wrong.

This is why engineers design circuits to avoid near-singular matrices
- they want systems where small measurement errors don't blow up into
huge solution errors.
""")
