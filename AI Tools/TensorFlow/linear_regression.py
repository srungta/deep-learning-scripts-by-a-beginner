import tensorflow as tf

# Create variables for weight and bias
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

#Hyperparameters
learning_rate =0.01

#Model
squared_deltas  = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

optimiser = tf.train.GradientDescentOptimizer(learning_rate)
train = optimiser.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

#Initialize global variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(W))

#Train
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

sess.close()