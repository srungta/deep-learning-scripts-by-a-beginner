'''
Python Basics with Numpy
This is a brief introduction to Python. 
The script uses Python 3.

The script below is to get familiar with:
1. Be able to use iPython Notebooks
2. Be able to use numpy functions and numpy matrix/vector operations
3. Understand the concept of "broadcasting"
4. Be able to vectorize code
Let's get started!
'''

#==============================================================================================================
#1 - Building basic functions with numpy
#==============================================================================================================

'''
Numpy is the main package for scientific computing in Python. It is maintained by a large community (www.numpy.org).Several key numpy functions such as np.exp, np.log, and np.reshape are used everyday in ML and DL.
'''
#------------------------------------------
# 1.1 - sigmoid function, np.exp()
#------------------------------------------
'''Before using np.exp(), we will use math.exp() to implement the sigmoid function. we will then see why np.exp() is preferable to math.exp().
Exercise: Build a function that returns the sigmoid of a real number x. Use math.exp(x) for the exponential function.
Reminder: sigmoid(x)=11+e−xsigmoid(x)=11+e−x is sometimes also known as the logistic function. It is a non-linear function used not only in Machine Learning (Logistic Regression), but also in Deep Learning.

To refer to a function belonging to a specific package we could call it using package_name.function(). 
'''
​
import math
​
def basic_sigmoid(x):
    """
    Compute sigmoid of x.
​
    Arguments:
    x -- A scalar
​
    Return:
    s -- sigmoid(x)
    """
    
    s = 1 / (1 + math.exp(-x))
    return s

#basic_sigmoid(3)	#0.9525741268224334
'''
Actually, we rarely use the "math" library in deep learning because the inputs of the functions are real numbers. In deep learning we mostly use matrices and vectors. This is why numpy is more useful.
### One reason why we use "numpy" instead of "math" in Deep Learning ###

'''
#Uncomment and run this line to see the error
#x = [1, 2, 3]
#basic_sigmoid(x)
'''
In fact, if x=(x1,x2,...,xn)x=(x1,x2,...,xn) is a row vector then np.exp(x)np.exp(x) will apply the exponential function to every element of x. The output will thus be: np.exp(x)=(ex1,ex2,...,exn)np.exp(x)=(ex1,ex2,...,exn)
'''
import numpy as np
​
# example of np.exp
x = np.array([1, 2, 3])
print(np.exp(x)) # result is (exp(1), exp(2), exp(3))

'''
Furthermore, if x is a vector, then a Python operation such as s=x+3s=x+3 or s=1xs=1x will output s as a vector of the same size as x.
'''
# example of vector operation
x = np.array([1, 2, 3])
print (x + 3)
#[4 5 6]
'''Any time we need more info on a numpy function, look at the official documentation or write np.exp? (for example) to get quick access to the documentation.
'''​
import numpy as np # this means we can access numpy functions by writing np.function() instead of numpy.function()

def sigmoid(x):
    """
    Compute the sigmoid of x
​
    Arguments:
    x -- A scalar or numpy array of any size
​
    Return:
    s -- sigmoid(x)
    """
    
    s = 1 / (1 + np.exp(-x) ) 
    
    return s

#x = np.array([1, 2, 3])
#sigmoid(x)

#------------------------------------------
# 1.2 - Sigmoid gradient
#------------------------------------------
'''
we will need to compute gradients to optimize loss functions using backpropagation. Let's code our first gradient function.
Exercise: Implement the function sigmoid_grad() to compute the gradient of the sigmoid function with respect to its input x. 
The formula is:

sigmoid_derivative(x)=σ′(x)=σ(x)(1−σ(x))
'''
​
def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    we can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array
​
    Return:
    ds -- wer computed gradient.
    """
    
    s = sigmoid(x)
    ds = s*(1-s)
    
    return ds

#x = np.array([1, 2, 3])
#print ("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

#------------------------------------------
# 1.3 - Reshaping arrays
#------------------------------------------
'''
Two common numpy functions used in deep learning are np.shape and np.reshape().
X.shape is used to get the shape (dimension) of a matrix/vector X.
X.reshape(...) is used to reshape X into some other dimension.
For example, in computer science, an image is represented by a 3D array of shape (length,height,depth=3)(length,height,depth=3). However, when we read an image as the input of an algorithm we convert it to a vector of shape (length∗height∗3,1)(length∗height∗3,1). In other words, we "unroll", or reshape, the 3D array into a 1D vector.

Let us implement image2vector() that takes an input of shape (length, height, 3) and returns a vector of shape (length*height*3, 1). For example, if we would like to reshape an array v of shape (a, b, c) into a vector of shape (a*b,c) we would do:
v = v.reshape((v.shape[0]*v.shape[1], v.shape[2])) # v.shape[0] = a ; v.shape[1] = b ; v.shape[2] = c
'''

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    
    return v

# This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],
​
       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],
​
       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
​
print ("image2vector(image) = " + str(image2vector(image)))


#------------------------------------------
# 1.4 - Normalizing rows
#------------------------------------------
'''
Another common technique we use in Machine Learning and Deep Learning is to normalize our data. It often leads to a better performance because gradient descent converges faster after normalization. Here, by normalization we mean changing x to x/∥x∥ (dividing each row vector of x by its norm).

we can divide matrices of different sizes and it works fine: this is called broadcasting 
Let us implement normalizeRows() to normalize the rows of a matrix. After applying this function to an input matrix x, each row of x should be a vector of unit length (meaning length 1).
'''
​
def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. we are allowed to modify x.
    """
    
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x,axis=1,keepdims=True)
    
    # Divide x by its norm.
    x = x / x_norm
​
    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))

'''
Note: In normalizeRows(), we can try to print the shapes of x_norm and x, and then rerun the assessment. we'll find out that they have different shapes. This is normal given that x_norm takes the norm of each row of x. So x_norm has the same number of rows but only 1 column. So how did it work when we divided x by x_norm? This is called broadcasting and we'll talk about it now!
'''
#------------------------------------------
# 1.5 - Broadcasting and the softmax function
#------------------------------------------
'''
A very important concept to understand in numpy is "broadcasting". It is very useful for performing mathematical operations between arrays of different shapes. For the full details on broadcasting, we can read the official broadcasting documentation.
Let us implement a softmax function using numpy. we can think of softmax as a normalizing function used when wer algorithm needs to classify two or more classes. 
Read more about this at https://en.wikipedia.org/wiki/Softmax_function 
'''

def softmax(x):
    """Calculates the softmax for each row of the input x.
​
    wer code should work for a row vector and also for matrices of shape (n, m).
​
    Argument:
    x -- A numpy matrix of shape (n,m)
​
    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)
​
    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp,axis=1,keepdims=True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp/x_sum
​
    
    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))
'''
If we print the shapes of x_exp, x_sum and s above and rerun the assessment cell, we will see that x_sum is of shape (2,1) while x_exp and s are of shape (2,5). x_exp/x_sum works due to python broadcasting.
'''

#==============================================================================================================
# 2) Vectorization
#==============================================================================================================
'''
In deep learning, we deal with very large datasets. Hence, a non-computationally-optimal function can become a huge bottleneck in wer algorithm and can result in a model that takes ages to run. To make sure that wer code is computationally efficient, we will use vectorization. Below are a few examples that demonstarte this
'''
import time
​
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
​
### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot+= x1[i]*x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
​
### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
tic = time.process_time()
outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i]*x2[j]
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
​
### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
​
### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j]*x1[j]
toc = time.process_time()
print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")


x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
​
### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
​
### VECTORIZED OUTER PRODUCT ###
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
​
### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
​
### VECTORIZED GENERAL DOT PRODUCT ###
tic = time.process_time()
dot = np.dot(W,x1)
toc = time.process_time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")


'''
As you may have noticed, the vectorized implementation is much cleaner and more efficient. For bigger vectors/matrices, the differences in running time become even bigger.
Note that np.dot() performs a matrix-matrix or matrix-vector multiplication. This is different from np.multiply() and the * operator (which is equivalent to  .* in Matlab/Octave), which performs an element-wise multiplication.
'''
#------------------------------------------
# 2.1 Implement the L1 and L2 loss functions
#------------------------------------------
'''
Let us implement the numpy vectorized version of the L1 loss. we may find the function abs(x) (absolute value of x) useful.
The loss is used to evaluate the performance of wer model. The bigger wer loss is, the more different wer predictions (ŷ y^) are from the true values (y). In deep learning, we use optimization algorithms like Gradient Descent to train our model and to minimize the cost.
L1 loss is defined as:
L1(ŷ ,y)=∑i=0m|y(i)−ŷ (i)
'''
​
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    loss = np.sum(abs(yhat-y))
    
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))

'''
Let us also implement the numpy vectorized version of the L2 loss. There are several way of implementing the L2 loss but we may find the function np.dot() useful. As a reminder, if x=[x1,x2,...,xn]x=[x1,x2,...,xn], then np.dot(x,x) = ∑nj=0x2j∑j=0nxj2.
L2 loss is defined as
L2(ŷ ,y)=∑i=0m(y(i)−ŷ (i))2
'''
​
def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    loss = np.sum(np.dot(yhat-y,yhat-y))
    #Also possible loss = np.sum((yhat-y)**2)
    
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))

# BONUS :  A quick reshape hack is to use -1 as the parameter
a = np.random.rand(3,2,2,3)
b = a.reshape(3,-1).T
'''
The shape of b is converted to (3,12). The number of columns is calculated automatically.
'''