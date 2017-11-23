import numpy as np

def convolve (matrix1, matrix2):
    mshape1 = matrix1.shape
    mshape2 = matrix2.shape
    if mshape1[0] < mshape2[0] or mshape1[1] < mshape2[1]:
        Exception()
    result = np.zeros([ mshape1[0] -  mshape2[0] + 1 , mshape1[1] -  mshape2[1] + 1 ])
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i][j] = np.sum(np.multiply(matrix1[i:i+mshape2[0], j:j+mshape2[1]],matrix2))
    return result


def generate_filtered_image(bnwimage, filter):
    convolved_image = convolve(bnwimage,filter)
    return convolved_image
