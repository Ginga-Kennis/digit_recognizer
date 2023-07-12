from mnist_loader import load_data,load_data_wrapper
from network_numpy import Network

if __name__ == "__main__":
    tr_d,va_d,te_d = load_data_wrapper()
    net = Network([784, 30, 10])
    net.SGD(tr_d,30,10,3.0,te_d)