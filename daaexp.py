import sys

def matrix_chain_order(p):
    n = len(p) - 1  # Number of matrices
    dp = [[0] * n for _ in range(n)]

    # L is chain length
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            dp[i][j] = sys.maxsize
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
    
    return dp[0][n - 1]

# Example usage
if __name__ == "__main__":
    arr = [40, 20, 30, 10, 30]
    print("Minimum number of multiplications is", matrix_chain_order(arr))
