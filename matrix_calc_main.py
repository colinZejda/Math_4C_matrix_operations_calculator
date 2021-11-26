###!user/bin/python

import numpy as np


class Matrix:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.A = np.zeros((rows, columns))       #we have our custom matrix in the constructor

    def setArray(self):
        for i in range(self.rows):
            input_row = input("enter the entries of row {i} for your matrix: ").split(" ")
            for x in range(len(input_row)):
                self.A[i][int(x)] = int(input_row[x])

    def get_determinant(self):
        return self.A.linalg.det(self.A)

    def get_rank(self):                          #remember, rank is the # of linearly independent rows or columns in a matrix(tells us )
        return np.linalg.matrix_rank(self.A)

    def get_eigenvalues(self):                   # matrix must be square
        if (self.rows == self.columns):
            return np.linalg.eigvals(self.A)         # return format is an array of doubles?
        else:
            return 0                             # not sure what to do if matrix isn't square

    def get_eigenvectors(self):
        if (self.rows == self.columns):
            eigenvalues, eigenvectors = np.linalg.eig(self.A)      #note: these values aren't integers, they're decimals
            return eigenvectors
        else:
            return 0

    def Transpose(self):
        return self.A.transpose

    def Inverse(self):   #use inv method
        if ((self.rows == self.columns) and (self.A.get_determinant != 0)):
            return np.linagl.inv(self.A)

    def Add(self, B):           # B is another matrix
        return self.A + B
    
    def Multiply(self, B):      # B is another matrix
        return self.A * B

    def solve_system(self, b):          #solve the system Ax = b(must add safety checks)
        return np.linalg.solve(self.A, b)

    def random_generator(self, range):   # range is the range for the random number generator
        return np.random.randint(range, size=(self.rows, self.columns))
    



def main():
    rows = int(input("enter the # of rows for your matrix: "))
    columns = int(input("enter the # of columns for your matrix: "))
    A = Matrix(rows, columns)
    #A.setArray()
    A.random_generator(columns)
    print(str(A.A))



if __name__ == "__main__":
    main()