import numpy as np

def pseudoinverse_svd(A):
    print("Original Matrix A:\n", A)

    # Step 1: SVD decomposition
    U, S, VT = np.linalg.svd(A)

    print("\nU Matrix:\n", U)
    print("\nSingular Values:\n", S)
    print("\nV^T Matrix:\n", VT)

    # Step 2: Create Sigma+
    Sigma_plus = np.zeros((VT.shape[0], U.shape[0]))

    # Threshold to avoid division by zero
    threshold = 1e-10

    for i in range(len(S)):
        if S[i] > threshold:
            Sigma_plus[i, i] = 1 / S[i]

    print("\nSigma+ (Pseudo-inverse of Sigma):\n", Sigma_plus)

    # Step 3: Compute A+
    A_plus = VT.T @ Sigma_plus @ U.T
    print("\nPseudoinverse A+:\n", A_plus)

    # Step 4: Verify Moore-Penrose Conditions
    print("\n--- Verification ---")

    cond1 = np.allclose(A @ A_plus @ A, A)
    cond2 = np.allclose(A_plus @ A @ A_plus, A_plus)
    cond3 = np.allclose((A @ A_plus).T, A @ A_plus)
    cond4 = np.allclose((A_plus @ A).T, A_plus @ A)

    print("Condition 1 (A A+ A = A):", cond1)
    print("Condition 2 (A+ A A+ = A+):", cond2)
    print("Condition 3 (A A+ symmetric):", cond3)
    print("Condition 4 (A+ A symmetric):", cond4)

    return A_plus


# ===== TEST MATRICES =====

# Square full rank (3x3)
A1 = np.array([[1, 2, 3],
               [0, 1, 4],
               [5, 6, 0]])

# Over-determined (4x2)
A2 = np.array([[1, 2],
               [3, 4],
               [5, 6],
               [7, 8]])

# Under-determined (2x4)
A3 = np.array([[1, 2, 3, 4],
               [5, 6, 7, 8]])

# Rank-deficient (singular matrix)
A4 = np.array([[1, 2, 3],
               [2, 4, 6],
               [1, 1, 1]])

print("===== TEST 1 =====")
pseudoinverse_svd(A1)

print("\n===== TEST 2 =====")
pseudoinverse_svd(A2)

print("\n===== TEST 3 =====")
pseudoinverse_svd(A3)

print("\n===== TEST 4 (Rank Deficient) =====")
pseudoinverse_svd(A4)