


import tensorflow as tf

### initials in tensorflow


# x = tf.constant(5)
# x = tf.constant([[1,2,3],[4,5,6]] , dtype=tf.float32)

# x = tf.ones(shape=(2,3), dtype=tf.float32)
# x = tf.zeros_like(x)


# x = tf.random.normal(shape=(2,3), mean= 0, stddev=1)
# x = tf.random.uniform(shape=(2,3), minval=1, maxval=2 )

# x = tf.range(9)
# x = tf.cast(x, dtype=tf.float32)
# print(x)



# n = tf.Variable('ali', dtype=tf.string)
# com = tf.Variable(2+4j , dtype=tf.complex64)
# vec = tf.Variable([1,2,3], dtype=tf.float16)
# mat = tf.Variable([[1,2,3],[4,5,6]]) 




### math

# x = tf.constant([1,2,3])
# y = tf.constant([9,8,7])

# z = tf.add(x, y)
# z = x + y

# z = tf.subtract(x, y)
# z = x - y

# z = tf.divide(x, y)
# z  = x / y 

# #x = x**2

# z = x * y
# z = tf.multiply(x, y)
# z = tf.reduce_sum(z, axis=0)
# z = tf.tensordot(x, y, axes=1)


# x = tf.random.normal(shape=(2,3))
# y = tf.random.normal(shape=(3,3))

# z = tf.matmul(x, y)
# z = x @ y

# print(z)



# Indexing
#x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
# print(x[:])
# print(x[1:])
# print(x[1:3])
# print(x[::2])
# print(x[::-1])

# indices = tf.constant([0, 3])
# x_ind = tf.gather(x, indices)
# print(x_ind)


# x = tf.constant([[1, 2], [3, 4], [5, 6]])

# print(x[0, :])
# print(x[0:2, :])



# x = tf.range(9)
# x = tf.reshape(x, (3, 3))
# x = tf.transpose(x, perm=[1, 0])



# x = tf.Variable([[1,2,3],[1,2,3]])
# print(x)
# x = tf.reshape(x, shape=(-1,6))
# print(x)


'''
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequantial

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train.reshape(-1,28*28).astype('float32') / 255.0
# x_test = x_test.reshape(-1,28*28).astype('float32') / 255.0

# print(x_train.shape)
# print(x_test.shape)



#1

model = tensorflow.keras.Sequantial(

    [
      keras.Input(28*28),
      layers.Dense(512, activation='relu'),
      layers.Dense(256, activation='relu'),
      layers.Dense(10),      
    ]
)

print(model.Summary())


#2 
model = tensorflow.keras.Sequantial()
model.add(keras.Input(28*28))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#3
input_ = keras.Input(shape=(28*28))
x = layers.Dense(512, activation='relu')(input_)
x = layers.Dense(256, activation='relu')(x)
output = layers.Dense(10, activation='softmax')(x)
model = keras.Model(input=input_, output=output)




model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)


model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

'''
