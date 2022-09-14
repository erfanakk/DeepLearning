


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




'''
#multi output

# Define model layers.
input_layer = Input(shape=(len(train .columns),))
first_dense = Dense(units='128', activation='relu')(input_layer)
second_dense = Dense(units='128', activation='relu')(first_dense)

# Y1 output will be fed directly from the second dense
y1_output = Dense(units='1', name='y1_output')(second_dense)
third_dense = Dense(units='64', activation='relu')(second_dense)

# Y2 output will come via the third dense
y2_output = Dense(units='1', name='y2_output')(third_dense)

# Define the model with the input layer and a list of output layers
model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

print(model.summary())


# Specify the optimizer, and compile the model with loss functions for both outputs
optimizer = tf.keras.optimizers.SGD(lr=0.001)
model.compile(optimizer=optimizer,
              loss={'y1_output': 'mse', 'y2_output': 'mse'},
              metrics={'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                       'y2_output': tf.keras.metrics.RootMeanSquaredError()})'''




'''

#def loss function without param

def huber_loss(y_true , y_pred):
    thresold = 1
    error = y_true - y_pred
    return_type = tf.abs(error) <= thresold
    r1 = 0.5 * tf.square(error)
    r2 = thresold * (tf.abs(error) - (0.5*thresold))
    return tf.where(return_type , r1 , r2)




#def loss function with param
def huber_loss_wrapper(thresold):
    def huber_loss(y_true , y_pred):
        error = y_true - y_pred
        return_type = tf.abs(error) <= thresold
        r1 = 0.5 * tf.square(error)
        r2 = thresold * (tf.abs(error) - (0.5*thresold))
        return tf.where(return_type , r1 , r2)
    return huber_loss


#class loss with param

class Huber(Loss):
    thresold = 1
    def __init__(self , thresold):
        super().__init__()
        self.thresold = thresold
    def call(self , y_true , y_pred): #callllll 
        error = y_true - y_pred
        return_type = tf.abs(error) <= self.thresold
        r1 = 0.5 * tf.square(error)
        r2 = self.thresold * (tf.abs(error) - (0.5*self.thresold))
        return tf.where(return_type , r1 , r2)


'''





# x = False
# z = 'this is z'
# y = 'this is y'

# print(tf.where(x,z,y))



'''
#siamese network 



def initialize_base_network():
    input = Input(shape=(28,28,), name="base_input")
    x = Flatten(name="flatten_input")(input)
    x = Dense(128, activation='relu', name="first_base_dense")(x)
    x = Dropout(0.1, name="first_dropout")(x)
    x = Dense(128, activation='relu', name="second_base_dense")(x)
    x = Dropout(0.1, name="second_dropout")(x)
    x = Dense(128, activation='relu', name="third_base_dense")(x)

    return Model(inputs=input, outputs=x)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)



# create the left input and point to the base network
input_a = Input(shape=(28,28,), name="left_input")
vect_output_a = base_network(input_a)

# create the right input and point to the base network
input_b = Input(shape=(28,28,), name="right_input")
vect_output_b = base_network(input_b)

# measure the similarity of the two vector outputs
output = Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])#***

# specify the inputs and output of the model
model = Model([input_a, input_b], output)
'''





'''#creat dense layer whitout activation function


class SimpleDense(tf.keras.layers.Layer):
    def __init__(self , units):
        super(SimpleDense , self).__init__()
        self.units = units
    def build(self , input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init((input_shape[-1] , self.units)) , trainable=True , dtype=tf.float32 , name="kernal")
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init((self.units)) , trainable=True , dtype=tf.float32 , name="bias")
    def call(self , inputs):
        return tf.multiply(inputs , self.w) + self.b
        
        
def my_Reu(x):
  return k.max(0,x)

model_simpledense = Sequential([
    Flatten(input_shape=(28,28)),
    MyDenseLayer(128),
    Lambda(my_Reu), 
    Dense(10 , activation = softmax)
])        
        
        
        
        
        
'''



'''
#class model 

class MyOwnModel(Model):
    def __init__(self,units = 30 , activation = "relu" , **kwargs):
        super().__init__()
        self.hidden1 = Dense(units , activation=activation , name="hidden1")
        self.hidden2 = Dense(units , activation=activation , name="hidden2")
        self.main_output = Dense(1)
        self.aux_output = Dense(1)
    def call(self , inputs):
        input_l , input_r = inputs
        hidden1 = self.hidden1(input_r)
        hidden2 = self.hidden2(hidden1)
        concat = concatenate([input_l , hidden2])
        main_output = self.main_output(concat)
        aux_output  = self.aux_output(hidden2)
        return main_output , aux_output

model = MyOwnModel()

'''




