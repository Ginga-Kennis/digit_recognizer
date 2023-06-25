"""
function to visualize digit data
"""
import mnist_loader
import matplotlib.pyplot as plt
import numpy as np

def visualize(data):
    fig = plt.figure
    plt.imshow(data,cmap="gray")
    plt.show()
    
if __name__ == "__main__":
    training_data, validation_data, test_data = mnist_loader.load_data()
    data = np.reshape(training_data[0][1000],(28,28))
    visualize(data)