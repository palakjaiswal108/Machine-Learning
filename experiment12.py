# palak jaiswal (0827CI201126) 
import tensorflow as tf from tensorflow 
import keras from tensorflow.keras 
import layers 
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data() x_train = x_train.astype('float32') / 255.0 x_test = x_test.astype('float32') / 255.0 model = keras.Sequential([     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),     layers.MaxPooling2D((2, 2)),     layers.Conv2D(64, (3, 3), activation='relu'),     layers.MaxPooling2D((2, 2)),     layers.Conv2D(128, (3, 3), activation='relu'),     layers.MaxPooling2D((2, 2)),     layers.Flatten(),     layers.Dense(128, activation='relu'),     layers.Dense(100) 
]) model.compile(optimizer='adam',  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']) model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))  
