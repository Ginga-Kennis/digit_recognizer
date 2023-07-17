import sys
import numpy as np
import random
from tqdm import tqdm

print(np.__file__)

class Network:
    def __init__(self,sizes): # sizes : number of neurons in the respective layers
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])] # Wjk(connection between the Kth neuron and Jth neuron)
        
    def feedforward(self,a): # a : input 
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a
    
    # stochastic gradient descent
    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        # if test_data is provided, the network will be evaluated against the test data after each epoch
        if test_data:
            n_test = len(test_data)
            
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            
            # for each mini_batch apply gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {0} completed")
                
    #compute gradients for every training data in mini_batch and update biases and weights            
    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x,y in mini_batch:
            # backpropagation
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
            
        self.weights = [w - (eta/len(mini_batch)) * nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b,nb in zip(self.biases, nabla_b)]
        
    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x]
        zs = []
        
        # feedforward
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # Eq➀ : error in the output layer 
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # Eq③ : rate of the change if the cost with respect to biases
        nabla_b[-1] = delta
        # Eq④ : rate of the change if the cost with respect to weights
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # backpropagate the error
        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            # Eq② : error delta(l) in terms of the error in the next layer delata(l+1)
            delta = np.dot(self.weights[-l+1].transpose(),delta) * sp
            # Eq③ : rate of the change if the cost with respect to biases
            nabla_b[-l] = delta
            # Eq④ : rate of the change if the cost with respect to weights
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
        return nabla_b, nabla_w
    
    def cost_derivative(self,output_activations,y):
        return output_activations - y
    
    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
                
        
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

    