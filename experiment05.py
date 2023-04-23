# palak jaiswal (0827CI201126) 
import pandas as pd 

# Load data from CSV file data = pd.read_csv('data.csv') 

# Print first 5 rows of the data print(data.head()) 
# palak jaiswal (0827CI201126)  
from sklearn.model_selection import train_test_split  
# Split the data into features (X) and target variable (y) X = data.iloc[:, :-1].values y = data.iloc[:, -1].values 

# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
# palak jaiswal (0827CI201126) 
class LinearRegression: 
         def __init__(self): 
        self.coefficients = None 
         def fit(self, X, y): 
        # Add a column of ones to X for the bias term 
        X = np.hstack((np.ones((X.shape[0], 1)), X))          
        # Calculate the coefficients using the normal equation         
        self.coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) 
         def predict(self, X): 
        # Add a column of ones to X for the bias term 
        X = np.hstack((np.ones((X.shape[0], 1)), X)) 

        # Make predictions using the coefficients        
        y_pred = X.dot(self.coefficients) 
                 return y_pred 
# palak jaiswal (0827CI201126)  
import numpy as np 

# Create an instance of the Linear Regression model 
lr = LinearRegression() 

# Train the model on the training data lr.fit(X_train, y_train) 
# palak jaiswal (0827CI201126)   # Make predictions on the testing data y_pred = lr.predict(X_test)  
# Print the predictions 
print(y_pred) 
# palak jaiswal (0827CI201126)   
import pandas as pd import numpy as np from sklearn.model_selection import train_test_split 
 class LinearRegression: 
         def __init__(self): 
        self.coefficients = None 
         def fit(self, X, y): 
        # Add a column of ones to X for the bias term 
        X = np.hstack((np.ones((X.shape[0], 1)), X))          
        # Calculate the coefficients using the normal equation         
        self.coefficients = np.linalg.inv(X.T)
