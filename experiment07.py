# palak jaiswal (0827CI201126) 
from sklearn.model_selection import train_test_split 
   xtrain, xtest, ytrain, ytest = train_test_split(     x, y, test_size=0.25, random_state=0) 
# palak jaiswal (0827CI201126) 
from sklearn.preprocessing import StandardScaler 

sc_x = StandardScaler() xtrain = sc_x.fit_transform(xtrain) xtest = sc_x.transform(xtest) 

print (xtrain[0:10, :]) 
# palak jaiswal (0827CI201126) 
from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression(random_state = 0) classifier.fit(xtrain, ytrain) 
# palak jaiswal (0827CI201126) 
from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(ytest, y_pred) print ("Confusion Matrix : \n", cm) 
# palak jaiswal (0827CI201126) 
from matplotlib.colors import ListedColormap 

X_set, y_set = xtest, ytest 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,                                
                               stop = X_set[:, 0].max() + 1, step = 0.01),                      
                     np.arange(start = X_set[:, 1].min() - 1,  
                               stop = X_set[:, 1].max() + 1, step = 0.01)) 

plt.contourf(X1, X2, classifier.predict(  np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))    
plt.xlim(X1.min(), X1.max()) plt.ylim(X2.min(), X2.max()) 
   for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],                
                c = ListedColormap(('red', 'green'))(i), label = j) 

plt.title('Classifier (Test set)') plt.xlabel('Age') plt.ylabel('Estimated Salary') plt.legend() 
plt.show() 
