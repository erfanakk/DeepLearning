import tensorflow as tf




class IdentityBlock(tf.keras.Model):
    def __init__(self, filters, kernal_size):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernal_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, kernal_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.add = tf.keras.layers.Add()
        self.relu = tf.keras.layers.Activation('relu')

    def call(self, input_tensor):
        Iden = input_tensor
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.add([x, Iden])
        x = self.relu(x)

        return x




class ResNet(tf.keras.Model):
    def __init__(self, num_class):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(64, 7, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.maxpool = tf.keras.layers.MaxPool2D((3,3))
        self.relu = tf.keras.layers.Activation('relu')
        self.iden1a = IdentityBlock(filters=64, kernal_size=3)
        self.iden1b = IdentityBlock(filters=64, kernal_size=3)
        self.globalpool= tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_class, activation='softma')
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.maxpool(x)
        
        x = self.iden1a(x)
        x = self.iden1b(x)
        
        x = self.globalpool(x)
        x = self.classifier(x)
        return x



if __name__ == '__main__':

    model = ResNet(num_class=10)
    input_shape = (1, 28, 28, 1) #NHWC
    model.build(input_shape)
    print(model.summary())




