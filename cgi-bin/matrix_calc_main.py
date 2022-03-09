#####!python3
print('Content-Type: text/html')
print('')                           # require blank line between CGI header and data

import sys
import os
import numpy as np
from numpy import linalg
import cgi
from fractions import Fraction

"""      a very useful tool during this project: the try-except
try:
    import numpy as np
    from numpy import linalg
except Exception as e:
    print(e, flush=True)
"""

#Plese note: most comments in this file were used for testing/labeling code, which is why I didn't clean them out

#my matrix calculator relies upon the numpy library in python. 
#in the large if-else block below, the general steps for each operation have been typed out by me(to display understanding)
class Matrix:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        #print(f"<p>rows = {rows}, columns = {columns}<p>")
        self.A = np.zeros((rows, columns))       #we have our custom matrix in the constructor
    def Add(self, B):           # B is another matrix
        return self.A + B.A
        
    def Multiply(self, B):      # B is another matrix
        return np.matmul(self.A, B.A)

    def get_determinant(self):
        if self.rows == self.columns:
            return Fraction(np.linalg.det(self.A)).limit_denominator()
        else:
            return 0              #represents singular matrices

    def get_rank(self):                          #remember, rank is the # of linearly independent rows or columns in a matrix(tells us )
        return np.linalg.matrix_rank(self.A)

    def Transpose(self):
        return self.A.transpose()

    def Inverse(self):   #use inv method
        try:
            return np.linalg.inv(self.A)
        except Exception as e:
            print(e, flush=True)
            return -1

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
    
    def QR_Factorization(self):
        return np.linalg.qr(self.A)    # deal with the 2 matrices that are returned in the if-else block below
    
    def SVD_Factorization(self):
        return np.linalg.svd(self.A)   #similar tactic to QR_Factorization

    def pseudo_inverse_MP(self):
        return np.linalg.pinv(self.A)

    def solve_system(self, b):          #solve the system Ax = b(must add safety checks)
        return np.linalg.solve(self.A, b)

    def random_generator(self, range):   # range is the range for the random number generator
        self.A = np.random.randint(range, size=(self.rows, self.columns))
    



def main():
    #code below uses cgi to communicate with html file(specifically the forms for user input)
    formData = cgi.FieldStorage()
    #formData = cgi.FieldStorage(None, None, [cgi.MiniFieldStorage('userOption', '1'), cgi.MiniFieldStorage('rows1', '3'), cgi.MiniFieldStorage('columns1', '3'), cgi.MiniFieldStorage('matrix1_inputs', '1 0 0 \r\n0 1 0 \r\n0 0 1'), cgi.MiniFieldStorage('rows2', '3'), cgi.MiniFieldStorage('columns2', '3'), cgi.MiniFieldStorage('matrix2_inputs', '1 0 0 \r\n0 1 0 \r\n0 0 1')])
    
    #print(f"<p>formData = {formData}<p>", flush = True)
    try:
        userOption = int(formData.getvalue('userOption')) if 'userOption' in formData else -1
    except Exception as e:
        userOption = -100000
    with open('../index.html') as fh:
        htmlText = fh.read()
    print(htmlText.replace('</body>', ''))
    print('<div style="float:right;line-height:30px; margin: 100px;" class="list" id="results"> ')
    print("<p>Results(hit calculate to update)<p>")
    #collect user inputs from html website
    if True:
        rows1 = int(formData.getvalue('rows1')) if 'rows1' in formData else -1
        columns1 = int(formData.getvalue('columns1')) if 'columns1' in formData else -1
        rows2 = int(formData.getvalue('rows2')) if 'rows2' in formData else -1
        columns2 = int(formData.getvalue('columns2')) if 'columns2' in formData else -1
        matrix1_str = formData.getvalue('matrix1_inputs') if 'matrix1_inputs' in formData else 'rip'
        matrix2_str = formData.getvalue('matrix2_inputs') if 'matrix2_inputs' in formData else 'rip'
    else:         #code used for testing when formData was corrupted/incorrect
        userOption = 1
        rows1=3
        columns1=3
        rows2=3
        columns2=3
        matrix1_str = "1 0 0\n0 1 0\n0 0 1\n"
        matrix2_str = "1 0 0 0 1 0 0 0 1 6 6 8"


    #print function for matricies(with formatting)
    def print_matrix(m1, rows, columns):       #m1 is a matrix object, we must access it's matrix, which is m1.A
        print("<table style='border: 1px solid black'>")
        for i in range(rows):
            print("<tr>")
            for j in range(columns):
                temp = str(Fraction(round(m1[i][j], 3)).limit_denominator())
                print(f"<td style='border: 1px solid black'>{temp}</td>")
            print("</tr>")
        print("</table>")


    #create matrix objects for user
    userMatrix1 = Matrix(rows1, columns1)
    userMatrix2 = Matrix(rows2, columns2)

    #split the matrix strings correctly to extract the numerical data
    matrix1_data = matrix1_str.split()     #reshape((rows1, columns1))
    matrix2_data = matrix2_str.split()
    
    #populate the matrices in my userMatrix objects
    for i in range(rows1):
        for j in range(columns1):
            userMatrix1.A[i][j] = matrix1_data[i * columns1 + j]
    for i in range(rows2):
        for j in range(columns2):
            userMatrix2.A[i][j] = matrix2_data[i * columns2 + j]
    
    

    #big if-else block for our menu of matrix operations
    if userOption in range(1, 11):
        #print out our matrix info(for user's viewing pleasure)
        print(f'<p>\nMatrix #1 is a {rows1} by {columns1} matrix:<p>\n', flush=True)
        print_matrix(userMatrix1.A, rows1, columns1)
        print(f'<p>\nMatrix #2 is a {rows2} by {columns2} matrix:<p>\n', flush=True)
        print_matrix(userMatrix2.A, rows2, columns2)

    if userOption == 1:              #add
        if rows1 == rows2 & columns1 == columns2:
            print(f'<p>\nThe matrix sum is:\n<p>\n', flush=True)
            matrixSum = userMatrix1.Add(userMatrix2)
            print_matrix(matrixSum, rows1, columns1)     #must pass in matrix object, not the matrix itself
        else:
            print('<p>\nInvalid inputs for matrix addition.\n<p>', flush=True)
        print("<p><p>")
        print(f'<p>\nGeneral steps for matrix addition by hand:\n<p>\n', flush=True)
        print(f'<p>\n1. Make sure both matrices have the same dimensions. This means both of your matrices are m by n.\n<p>\n', flush=True)
        print(f'<p>\n2. Add each matrix entry A(i, j) in matrix1 with the corresponding entry A(i, j) with matrix2. The resulting number goes in entry spot A(i, j) for the matrix sum.\n<p>\n', flush=True)
        print(f'<p>\n2. Repeat for each entry, (1,1) to (m, n) until you have a full m by n matrix.\n<p>\n', flush=True)


    elif userOption == 2:            #multiply
        if columns1 == rows2:
            print(f'<p>\nThe matrix product is:\n<p>\n', flush=True)
            matrixProduct = userMatrix1.Multiply(userMatrix2)             #no return value, this line is simply an operation
            print_matrix(matrixProduct, rows1, columns1)    #pass matrix object userMatrix1 into my print function
        else:
            print('<p>\nSorry, your inputs were invalid for matrix multiplication.\n<p>', flush=True)
        print("<p><p>")
        print(f'<p>\nGeneral steps for matrix multiplication by hand:\n<p>\n', flush=True)
        print(f'<p>\n1. Make sure that matrix1 has as many columns as matrix2 has rows. If not, the matrices are not compatible for matrix multiplication.\n<p>\n', flush=True)
        print(f'<p>\n1. Find the dot product between row #1 of matrix1 and each column n2 of matrix2. The resulting number becomes entry A(1, n2) of the matrix product.\n<p>\n', flush=True)
        print(f'<p>\n2. Repeat step one for each row of matrix1 and the columns of matrix2. It will be a lot of work, but stay strong!\n<p>\n', flush=True)
            
    elif userOption == 3:            #determinant
        detM = userMatrix1.get_determinant()
        print(f'<p>\nMatrix1 has a determinant of {detM}\n<p>\n', flush=True)
        print("<p><p>")
        print(f'<p>\nGeneral steps for finding the determinant by hand:\n<p>\n', flush=True)
        print(f'<p>\n1. Make sure your matrix is square. This means it has as many rows and it does columns.\n<p>\n', flush=True)
        print(f'<p>\n2. For 2x2 matricies, det(matrix A) = ad - bc, where elements a and b are in row #1, and elements c and d are in row #2.\n<p>\n', flush=True)
        print(f'<p>\n3. For 3x3 matricies, det(matrix A) = the cofactor expansion across any row or down any column of the matrix. The general form down row 1 is det(A) = a11C11 + a12C12 + ... + a1nC1n, where C = (-1)^(i+j) * det(Aij).\n<p>\n', flush=True)
        print(f'<p>\n4. For 4x4 matrices and up, the cofactor expansion works, but the amount of mathematical computations by hand can become tedious, so a calculator would be recommended.\n<p>\n', flush=True)

    elif userOption == 4:            #rank
        rankM = userMatrix1.get_rank()
        print("<p><p>")
        print(f'<p>\nMatrix1 has a rank of {rankM}\n<p>\n', flush=True)
        print("<p><p>")
        print(f'<p>\nGeneral steps for matrix rank by hand:\n<p>\n', flush=True)
        print(f'<p>\n1. Row reduce your matrix A to reduced echelon form.\n<p>\n', flush=True)
        print(f'<p>\n2. Count how many pivot columns there are in the reduced matrix. This number is your rank.\n<p>\n', flush=True)

    elif userOption == 5:            #transpose
        print(f'<p>\nThe transpose of matrix1 is:\n<p>\n', flush=True)
        t = userMatrix1.Transpose()
        print_matrix(t, rows1, columns1)
        print("<p><p>")
        print(f'<p>\nGeneral steps for finding a matrix transpose by hand:\n<p>\n', flush=True)
        print(f'<p>\n1. From your m by n matrix, take each element in row #1, and place these elements into column #1 of the resulting matrix.\n<p>\n', flush=True)
        print(f'<p>\n2. Repeat step #1 until each row has been converted, and the resulting matrix is an n by m matrix.\n<p>\n', flush=True)

    elif userOption == 6:            #inverse
        if (rows1 == columns1) & (userMatrix1.get_determinant() != 0):          #if matrix is square and non-singular, we can compute it's inverse
            print(f'<p>\nThe inverse of matrix1 is:\n<p>\n', flush=True)
            i = userMatrix1.Inverse()
            print_matrix(i, rows1, columns1)
        elif userMatrix1.get_determinant() == 0:
            print('<p>\nSorry, det(matrix1) = 0, and the inverse cannot be performed on singular matrices.\n<p>', flush=True)
        else:
            print('<p>\nInvalid inputs for a inverse matrix.\n<p>', flush=True)
        print("<p><p>")
        print(f'<p>\nGeneral steps for finding a matrix inverse by hand:\n<p>\n', flush=True)
        print(f'<p>\n1. Make sure the matrix is square. If it is not, then it the matrix is not invertible.\n<p>\n', flush=True)
        print(f'<p>\n2. Find the determinant and make sure it is nonzero. If det(A) = 0, then the matrix is singular, and will be non-invertible.\n<p>\n', flush=True)
        print(f'<p>\n3. Create an augmented matrix from matrix A and the identity matrix I.\n<p>\n', flush=True)
        print(f"<p>\n4. Perform row-reduction operations on [A I] until the augmented matrix is [I A^-1]. A^-1 is your inverse.\n<p>\n")

    elif userOption == 7:            #eigenvalues
        eigenvalues = userMatrix1.get_eigenvalues()
        try:
            print(f'<p>\nThe eigenvalues of matrix1 are:\n<p>\n', flush=True)
            print("<p>")
            for i in eigenvalues:
                print(f'{round(i, 2)}&nbsp&nbsp&nbsp')
            print("<p>")
        except Exception as E:
            print(E, flush=True)
            print(f'<p>\nThe eigenvalues are {eigenvalues}\n<p>\n', flush=True)
        print("<p><p>")
        print(f'<p>\nGeneral steps for finding eigenvalues by hand:\n<p>\n', flush=True)
        print(f'<p>\n1. Find the matrix [A-lambda*I]. Here, I is the identity matrix and lambda is a variable representing our eigenvalues.\n<p>\n', flush=True)
        print(f'<p>\n2. Find the determinant of the matrix [A-lambda*I]. The result is your characteristic equation, which should resemble a polynomial.\n<p>\n', flush=True)
        print(f'<p>\n3. Solve the characteristic equation by factoring, or with the quadratic formula. The roots are your desired eigenvalues.\n<p>\n', flush=True)

    elif userOption == 8:            #eigenvectors
        eigenvalues = userMatrix1.get_eigenvalues()
        eigenvectors = userMatrix1.get_eigenvectors()
        print(f'<p>\nThe eigenvectors are:\n<p>\n', flush=True)
        try:
            print("<p>")
            for i in eigenvectors:
                print('[')
                for j in i:
                    print(f'{round(j, 2)}&nbsp&nbsp')   #the &nbsp is a hard-space in html, end=' ' doesn't work here b/c we're in the <p> brackets
                print(']')
            print("<p>")
        except Exception as e:
            print(e, flush=True)
        print("<p>Their respective eigenvalues are:<p>")
        print("<p>")
        for i in eigenvalues:
            print(f'{round(i, 2)}&nbsp&nbsp&nbsp')   #the &nbsp is a hard-space in html, end=' ' doesn't work here b/c we're in the <p> brackets
        print("<p>")
        print("<p><p>")
        print(f'<p>\nGeneral steps for finding eigenvectors by hand:\n<p>\n', flush=True)
        print(f'<p>\n1. Find the eigenvalues of your matrix\n<p>\n', flush=True)
        print(f'<p>\n2. For each eigenvalue, recreate the [A-lambda*I] matrix. Plug in one of your eigenvalues for lambda\n<p>\n', flush=True)
        print(f'<p>\n3. Take the elements of row #1 of [A-lambda*I] and create its corresponding equation. Ex: x1 - 3*x2 = 0\n<p>\n', flush=True)
        print(f'<p>\n4. Plug in a convenient value for the free variable x2 in your equation, and you will find that x1 = some value. Craft your eigenvector as [x1 x2]\n<p>\n', flush=True)
        print(f'<p>\n5. Repeat steps 2-4 for each eigenvalue. If your eigenvalue has a multiplicity greater than 1, say n, then you will have n eigenvectors for that eigenvalue\n<p>\n', flush=True)
    
    elif userOption == 9:       # QR factorization
        print(f'<p>\nThe QR factorization of matrix1 is:\n<p>\n', flush=True)
        matrixQ, matrixR = userMatrix1.QR_Factorization()
        print("<p>Matrix Q:<p>")
        print_matrix(matrixQ, len(matrixQ), len(matrixQ[0]))
        print("<p>Matrix R:<p>")
        print_matrix(matrixR, len(matrixR), len(matrixR[0]))
        print("<p><p>")
        print(f'<p>\nGeneral steps for factoring a matrix into QR by hand:\n<p>\n', flush=True)
        print(f'<p>\n1. For a matrix A, use the Gram-Schmidt Process to find an orthogonal basis \n<p>\n', flush=True)
        print(f'<p>\n1a. Gram Schmidt involves calculating the difference between a vector x(a column of matrix A) and its projection onto the subspace W(spanned by vectors x1 and v1)\n<p>\n', flush=True)
        print(f'<p>\n2. Normalize each vector in the orthogonal basis. These vectors will form the columns of matrix Q\n<p>\n', flush=True)
        print(f'<p>\n3. Calculate the transpose of Q. The transpose of Q is equivalent to the inverse of Q because Q has orthonormal columns\n<p>\n', flush=True)
        print(f'<p>\n4. Left-multiply both sides of the equation A = QR by the transpose of Q. The resulting matrix is our R matrix\n<p>\n', flush=True)
        print(f'<p>\n5. Finally, check that Q is an m by n matrix whose columns are an orthonormal basis for Col(A), and that R is an n by n upper-triangular matrix\n<p>\n', flush=True)

    elif userOption == 10:          #SVD factorization
        try:
            print(f'<p>\nThe SVD factorization of matrix1 is:\n<p>\n', flush=True)
            matrixU, matrixSigma, matrixVT = userMatrix1.SVD_Factorization()         # matrix A = U * sigma * the transpose of V
            print("<p>Matrix U:<p>")
            print_matrix(matrixU, len(matrixU), len(matrixU[0]))
            print("<p>Matrix Sigma has values:<p>")
            for i in matrixSigma:
                print(f'{round(i, 2)}&nbsp&nbsp')   #the &nbsp is a hard-space in html, end=' ' doesn't work here b/c we're in the <p> brackets
            print("<p>Matrix Transpose of V:<p>")
            #print(f'<p>\n{matrixVT}\n<p>\n', flush=True)
            print_matrix(matrixVT, len(matrixVT), len(matrixVT[0]))
        except Exception as E:
            print(E, flush=True)
            print("<P>Sorry, the SVD computation doesn't converge<p>")
        print("<p><p>")
        print(f'<p>\nGeneral steps for finding the SVD of a matrix by hand:\n<p>\n', flush=True)
        print(f'<p>\n1. Compute the transpose of A, A^T\n<p>\n', flush=True)
        print(f'<p>\n2. Left multiply A by A^T, for a resulting matrix [ATA]\n<p>\n', flush=True)
        print(f'<p>\n3. Find the eigenvectors of ATA, then normalize them. These vectors will be used as the columns of matrix V\n<p>\n', flush=True)
        print(f'<p>\n4. Find the eigenvalues of ATA in descending order, then take their square roots. The resulting numbers are the singular values of ATA. \n<p>\n', flush=True)
        print(f'<p>\n5. Along the main diagonal of matrix Sigma, place the nonzero singular values(from step 4). Populate the rest of matrix Sigma with zeroes. The dimensions of matrix Sigma are the same as the original matrix A\n<p>\n', flush=True)
        print(f'<p>\n6. For each eigenvector v, calculate (matrix A) * (vector v) = Av \n<p>\n', flush=True)
        print(f'<p>\n7. Then, multiply each Av by (1 / the corresponding singular value of eigenvector v), obtaining our columns for matrix U \n<p>\n', flush=True)
        print(f'<p>\n8. Finally, assemble your factorization as A = U * Sigma * V^T, where V^T is the transpose of matrix V\n<p>\n', flush=True)

    elif userOption == 11:          #Moore-Penrose, aka the pseudo-inverse of a matrix
        try:
            print(f'<p>\nThe pseudo-inverse(Moore-Penrose) of matrix1 is:\n<p>\n', flush=True)
            matrixPINV = userMatrix1.pseudo_inverse_MP()
            print_matrix(matrixPINV, len(matrixPINV), len(matrixPINV[0]))
        except Exception as E:
            print(E, flush=True)
        print("<p><p>")
        print(f'<p>\nGeneral steps for finding the pseudo-inverse(A^+) by hand:\n<p>\n', flush=True)
        print(f'<p>\n1. There are two cases, when the columns of matrix A are linearly independent, or when the rows are LI\n<p>\n', flush=True)
        print(f'<p>\n2. If the columns are LI, A^+ = (the inverse of (A^T * A)) * A\n<p>\n', flush=True)
        print(f'<p>\n3. If the rows are LI, A^+ = A^T * (the inverse of (A * A^T))\n<p>\n', flush=True)
        print(f'<p>\n4. Case #3: if both the rows and columns of A are linearly independent, then the matrix is invertible, and the pseudo inverse equals the inverse: A^+ = A^-1\n<p>\n', flush=True)
    else:                            #invalid option
        print("<p>Sorry, invalid option selected. Please try again.<p>")
    print('</div>')
    print('</body>')
    exit(0)


if __name__ == "__main__":
    main()
