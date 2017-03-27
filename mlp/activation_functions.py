from scipy.special import expit


class ReLU:
    @staticmethod
    def activation(net):
        net[net < 0] = 0
        return net

    @staticmethod
    def f_prime(net):
        n = net.copy()
        n[n <= 0] = 0
        n[n > 0] = 1
        return n


class Sigmoid:
    @staticmethod
    def activation(net):
        return expit(net, net)

    @staticmethod
    def f_prime(net):
        return net * (1. - net)
