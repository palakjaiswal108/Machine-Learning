# palak jaiswal (0827CI201126) 
import tensorflow as tf
from tenserflow.keras.datasets import mnist
#Load the MNIST dataset
(x train,V train),(x test,V test) = mnist.load data()
#preprocessing data
x train = x train / 255.0
x test = x test /255.0
#Reshape the input data to have a depth of 1
x train = x train.reshape(x train.shape[0],28,28,1)
x test = x test.reshape(x test.shape[0],28,28,1)
#define cnn architecture
# palak jaiswal (0827CI201126) 
from tensorflow.keras.models import Sequential from
tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten,Dense
#define cnn architecture
# palak jaiswal (0827CI201126) 
from tensorflow.keras.models import Sequential from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  
# Define the CNN architecture 
model = Sequential([
Convo2D(32,(3,3),activation='relu',input shape=(28,28,1)),
MaxPooling2D((2,2)),
Conv2D(64,(3,3),activation='relu'),
MaxPooling2D((2,2));
Flatten(),
Dense(64, activation='relu'),
Dense(10,activatiob='softmax')
])
#Compile the model
model.compile(optimizer='adam',loss='categorical crossentropy',metrics=['accuracy']
#Train the model
model.fit(x train, tf.keras.utils.to categorical(v train), epochs=5,batch size=32)
# palak jaiswal (0827CI201126)   # Evaluate the model on the test 
              set test_loss, test_acc = model.evaluate(x_test, tf.keras.utils.to_categorical(y_test)) print('Test accuracy:', test_acc) 
