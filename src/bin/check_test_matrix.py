import numpy as np

""" 
=== Hardcode of values in A

row/col 0  1  2  3  4  5  6  7 
0     | 1           1  1       |
1     |    1     1  1          |
2     |       1  1           1 | 
3     |    1  1  1             |
4     | 1  1        1          |
5     | 1              1       |
6     |                   1  1 | 
7     |    1  1           1  1 |
"""

rows = 8; cols = 8; shape = (rows, cols)
A = np.array(
    [
        #  0   1   2   3   4   5   6   7
        [ 10,  0,  0,  0,  1,  1,  0,  0],
        [  0,  10,  0,  1,  1,  0,  0,  1],
        [  0,  0,  10,  1,  0,  0,  0,  1],
        [  0,  1,  1,  10,  0,  0,  0,  0],
        [  1,  1,  0,  0,  10,  0,  0,  0],
        [  1,  0,  0,  0,  0,  10,  0,  0],
        [  0,  0,  0,  0,  0,  0,  10,  1],
        [  0,  1,  1,  0,  0,  0,  1,  10]
        
    ])

# Check symmetry
for i in range(0,rows):
    for j in range(0,i):
        if A[i,j] != A[j,i]:
            print(f"Non symmetry: A[{i},{j}]={A[i,j]}!={A[j,i]}=A[{j},{i}]")
            print("Not going forward.")
            exit(0)

eigvals = np.linalg.eigvals(A)
if np.all(eigvals>0.0):
    print("A is SPD")
else:
    print("A is not SPD. Not computing L")
    print("Eigvals of A are: ", eigvals)
    exit(0)

L = np.linalg.cholesky(A)
print("Cholesky factor L is:")
print(L)

b = np.array([1,1,1,1,1,1,1,1]).T
x = np.linalg.solve(A, b)
print("Solution of Ax=b is: ")
print(x)