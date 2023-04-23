
# palak jaiswal (0827CI201126) 
# Import Libraries import numpy as np import pandas 
as pd from sklearn.datasets import load_iris from sklearn.model_selection import 
train_test_split import matplotlib.pyplot as plt 
# palak jaiswal (0827CI201126) 
# Load dataset data = load_iris() 

# Get features and target X=data.data y=data.target 
# palak jaiswal (0827CI201126)   # Get dummy variable  y = pd.get_dummies(y).values 
 y[:3] 
# palak jaiswal (0827CI201126)   
#Split data into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=4) 
learning_rate = 0.1 iterations = 5000 
N = y_train.size 

# number of input features input_size = 4 

# number of hidden layers neurons hidden_size = 2  

# number of neurons at the output layer output_size = 3   

results = pd.DataFrame(columns=["mse", "accuracy"])  
# palak jaiswal (0827CI201126) 
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) def mean_squared_error(y_pred, y_true): 
    return ((y_pred - y_true)**2).sum() / (2*y_pred.size) def accuracy(y_pred, y_true): 
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)     return acc.mean() 
# palak jaiswal (0827CI201126) 
for itr in range(iterations):     
    # feedforward propagation on hidden layer 
    Z1 = np.dot(x_train, W1) 
    A1 = sigmoid(Z1) 

    # on output layer 
    Z2 = np.dot(A1, W2) 
    A2 = sigmoid(Z2) 

    # Calculating error    
    mse = mean_squared_error(A2, y_train)     
    acc = accuracy(A2, y_train)     
    results=results.append({"mse":mse, "accuracy":acc},ignore_index=True ) 

    # backpropagation    
    E1 = A2 - y_train     
    dW1 = E1 * A2 * (1 - A2) 

    E2 = np.dot(dW1, W2.T)     
    dW2 = E2 * A1 * (1 - A1) 

    # weight updates 
    W2_update = np.dot(A1.T, dW1) / N 
    W1_update = np.dot(x_train.T, dW2) / N  
    W2 = W2 - learning_rate * W2_update 
    W1 = W1 - learning_rate * W1_update  
# palak jaiswal (0827CI201126) 
results.mse.plot(title="Mean Squared Error") 
# palak jaiswal (0827CI201126) 
results.accuracy.plot(title="Accuracy") 
# palak jaiswal (0827CI201126) 
Z1 = np.dot(x_test, W1) 
A1 = sigmoid(Z1) 

Z2 = np.dot(A1, W2) 
A2 = sigmoid(Z2)  acc = accuracy(A2, y_test) print("Accuracy: {}".format(acc)) 
