import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 


# test_images = np.load('mnist_train_images.npy')
# print(test_images.shape)
# new = np.reshape(test_images,(-1,28,28))
# print(new.shape)
# plt.imshow(new[0])
# plt.show()

INPUT = 784
HIDDEN = 512
BATCH_SIZE = 100

class Encoder (tf.keras.layers.Layer):
    def __init__ (self):
        super(Encoder, self).__init__()

        # self.flattenlayer = tf.keras.layers.flatten() 
        self.dense1 = tf.keras.layers.Dense(200, activation = 'tanh') 
        self.dropout = tf.keras.layers.Dropout(0.2) 

        self.mean = tf.keras.layers.Dense(16, activation = 'tanh') 
        self.logvariance = tf.keras.layers.Dense(16, activation = 'tanh') 


    def forward (self, X):
        
        # X = self.flattenlayer(X)
        X = self.dense1(X)
        X = self.dropout(X)

        mean = self.mean(X)
        logvariance = self.logvariance(X)
        # print(mean.shape)
        # print(logvariance.shape)

        return mean,logvariance


class Decoder (tf.keras.layers.Layer):
    def __init__ (self):
        super(Decoder, self).__init__()

        self.dense1 = tf.keras.layers.Dense(200, activation = 'tanh') 
        self.dense2 = tf.keras.layers.Dense(INPUT, activation = 'tanh') 

    def forward (self, Z):
        Z = self.dense1(Z)
        generatedoutput = self.dense2(Z)

        return generatedoutput


class VAE (tf.keras.Model):
    def __init__ (self):
        super(VAE, self).__init__()
        self.decoder = Decoder()
        self.encoder = Encoder()

    # Computes the hidden representation given the pre-sampled values eps
    def z (self, x, eps):
        code = self.encoder
        # TODO extract mu and sigma2 from the code, and then use them and eps
        # to compute z.
        mu,logsigma2 = code.forward(x)
        z = mu+ tf.exp(0.5*logsigma2)*eps

        return z, mu, tf.exp(logsigma2)

    def x (self, z):
        x = self.decoder(z)
        return x


def train (vae, X):
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)  # worked for me

    def computeLoss (x):
        # TODO implement your custom loss
        model = VAE()
        encoder = Encoder()
        mean,variance  = encoder.forward(x)
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(mean)[0], tf.shape(variance)[1]))

        z, mean, variance = model.z(x,epsilon)
        output = model.x(z)
        reconstruction_loss = INPUT*tf.keras.losses.binary_crossentropy(x,x)
        
        kl_loss = - 0.5* tf.keras.backend.sum((1 + variance - tf.square(mean) - tf.exp(variance)), axis = -1)
        return tf.keras.backend.mean(reconstruction_loss+kl_loss),z

        

    for e in range(150):  # epochs
        print("Epoch {}".format(e))
        for i in np.arange(0, len(X), BATCH_SIZE):
            # Call computeLoss on each minibatch
            # loss, z = computeLoss(X[i:i+BATCH_SIZE,:])
            # loss.backward()
            optimizer.compile(optimizer, loss=computeLoss(X[i:i+BATCH_SIZE,:])
            optimizer.fit(X[i:i+BATCH_SIZE,:], X[i:i+BATCH_SIZE,:])

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # vae = VAE().to(device)
    vae = VAE()
    
    X = np.load('mnist_train_images.npy')
    train(vae, X)

    # epsilon = tf.random_normal(shape=(tf.shape(mean)[0], tf.shape(variance)[1]))
