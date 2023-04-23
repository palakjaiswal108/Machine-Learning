# palak jaiswal (0827CI201126)  # Import Libraries import numpy as np import pandas 
as pd from sklearn.datasets import load_iris from sklearn.model_selection import 
train_test_split import matplotlib.pyplot as plt 
# palak jaiswal (0827CI201126) 
# Load dataset data = load_iris() 

# Get features and target X=data.data y=data.target 
# palak jaiswal (0827CI201126)   # Get dummy variable  y = pd.get_dummies(y).values  y[:3]  
# palak jaiswal (0827CI201126) 
#Split data into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=4) 
# palak jaiswal (0827CI201126)    # Initialize variables learning_rate = 0.1 iterations = 5000 
N = y_train.size 

# number of input features input_size = 4 

# number of hidden layers neurons hidden_size = 2  

# number of neurons at the output layer output_size = 3   
 results = pd.DataFrame(columns=["mse", "accuracy"])  
# palak jaiswal (0827CI201126) 
for itr in range(iterations):     
    # feedforward propagation on hidden layer 
    Z1 = np.dot(x_train, W1) 
    A1 = sigmoid(Z1) 

    # on output layer 
    Z2 = np.dot(A1, W2) 
    A2 = sigmoid(Z2) 

    # Calculating error     mse = mean_squared_error(A2, y_train) 
    acc = accuracy(A2, y_train)     results=results.append({"mse":mse, "accuracy":acc},ignore_index=True ) 

    # backpropagation     E1 = A2 - y_train     dW1 = E1 * A2 * (1 - A2) 

    E2 = np.dot(dW1, W2.T)     dW2 = E2 * A1 * (1 - A1) 

    # weight updates 
    W2_update = np.dot(A1.T, dW1) / N 
    W1_update = np.dot(x_train.T, dW2) / N  
    W2 = W2 - learning_rate * W2_update 
    W1 = W1 - learning_rate * W1_update  
# palak jaiswal (0827CI201126) 
results.mse.plot(title="Mean Squared Error") 
# palak jaiswal (0827CI201126) 
import numpy as np import tensorflow as tf 
from tensorflow.keras.datasets import cifar100 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization from tensorflow.keras.callbacks import EarlyStopping 
# Load the CIFAR-100 dataset 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()  
# Normalize the input images to have pixel values between 0 and 1 x_train = x_train.astype('float32') / 255.0 x_test = x_test.astype('float32') / 255.0 

# Convert labels to one-hot encoded vectors num_classes = 100 y_train = tf.keras.utils.to_categorical(y_train, num_classes) y_test = tf.keras.utils.to_categorical(y_test, num_classes)  
# Define the model architecture model = Sequential([     # Input layer     tf.keras.layers.Flatten(input_shape=(32, 32, 3)), 

    # Hidden layer 1     tf.keras.layers.Dense(512, activation='relu'),     tf.keras.layers.BatchNormalization(),     tf.keras.layers.Dropout(0.2), 

    # Hidden layer 2     tf.keras.layers.Dense(256, activation='relu'),     tf.keras.layers.BatchNormalization(),     tf.keras.layers.Dropout(0.2), 

    # Output layer     tf.keras.layers.Dense(num_classes, activation='softmax') ]) 

# Compile the model model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

# Define early stopping callback early_stop = EarlyStopping(monitor='val_loss', patience=5)  
# Train the model 
history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stop])  
# Evaluate the model on test set 
test_loss, test_acc = model.evaluate(x_test, y_test)  
# Print the test accuracy 
print("Test accuracy:", test_acc)  
