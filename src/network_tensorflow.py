import tensorflow as tf
from keras.datasets import mnist
from keras.utils import plot_model



if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = X_train / 255.0
    X_test = X_train / 255.0
    

    
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(30,activation="sigmoid"),
        tf.keras.layers.Dense(10,activation="sigmoid")
    ])

    model.compile(optimizer="sgd",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    # plot_model(model, to_file='model.png')
    model.summary()
    model.fit(X_train,y_train,epochs=20,batch_size=30)
    
    

