import numpy as np

def diagonalize_matrix(A):
    print("Original Matrix A:\n", A)

    # Step 1: Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("\nEigenvalues:\n", eigenvalues)
    print("\nEigenvectors (P):\n", eigenvectors)

    # Step 2: Construct D (Diagonal Matrix)
    D = np.diag(eigenvalues)
    print("\nDiagonal Matrix D:\n", D)

    # Step 3: Compute P inverse
    try:
        P_inv = np.linalg.inv(eigenvectors)
    except np.linalg.LinAlgError:
        print("\nMatrix is NOT diagonalizable (P is not invertible)")
        return

    print("\nP Inverse:\n", P_inv)

    # Step 4: Verify A = P D P⁻¹
    A_reconstructed = eigenvectors @ D @ P_inv
    print("\nReconstructed A (P D P⁻¹):\n", A_reconstructed)

    # Step 5: Error check
    error = np.linalg.norm(A - A_reconstructed)
    print("\nReconstruction Error:", error)

    # Step 6: Example: Compute A^3 using diagonalization
    D_power = np.linalg.matrix_power(D, 3)
    A_power = eigenvectors @ D_power @ P_inv

    print("\nA^3 using diagonalization:\n", A_power)


# ===== TEST MATRICES =====

# 3x3 matrix
A1 = np.array([[5, 1, 2],
               [1, 4, 1],
               [2, 1, 3]])

# 2x2 matrix
A2 = np.array([[2, 1],
               [1, 2]])

# Defective matrix (not diagonalizable)
A3 = np.array([[1, 1, 0, 0],
               [0, 1, 1, 0],
               [0, 0, 1, 1],
               [0, 0, 0, 1]])

print("===== TEST 1 =====")
diagonalize_matrix(A1)

print("\n===== TEST 2 =====")
diagonalize_matrix(A2)

print("\n===== TEST 3 (Defective Matrix) =====")
diagonalize_matrix(A3)