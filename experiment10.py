# palak jaiswal (0827CI201126) 
magic, n = struct.unpack('>II', lbpath.read(8)) labels = np.fromfile(lbpath, dtype=np.uint8) 
# palak jaiswal (0827CI201126) 
import numpy as np import pandas as pd import matplotlib.pyplot as plt my_list = [1, 2, 3, 4, 5] my_tuple = (6, 7, 8, 9, 10) my_string = "Hello, world!" my_dict = {'name': 'John', 'age': 30, 'city': 'New York'} multiply = lambda x, y: x * y class Person:     def _init_(self, name, age): 
        self.name = name        
  self.age = age     
  def say_hello(self): 
        print("Hello, my name is", self.name) person1 = Person("John", 30) print(person1.name) person1.say_hello() my_array = np.array([1, 2, 3, 4, 5]) print(my_array) my_dataframe = pd.DataFrame({'name': ['John', 'Mary', 'Adam'], 'age': [30, 25, 
40]}) 
      print(my_dataframe) x = np.linspace(0, 10, 100) y = np.sin(x) plt.plot(x, y) plt.show() 
from tensorflow import keras from tensorflow.keras import layers 

 model = keras.Sequential([ layers.Dense(1024, activation='relu', input_shape=[11]),     layers.Dropout(0.3),     layers.BatchNormalization(),     layers.Dense(1024, activation='relu'), layers.Dropout(0.3),     layers.BatchNormalization(),     layers.Dense(1024, activation='relu'),     layers.Dropout(0.3),     layers.BatchNormalization(),     layers.Dense(1), 
]) model.compile(     optimizer='adam',     loss='mae')  history = model.fit(     X_train, y_train,     validation_data=(X_valid, y_valid),     batch_size=256,     epochs=100,     verbose=0) 
