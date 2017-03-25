from scipy.special import expit


class ReLU:
    @staticmethod
    def activation(net):
        net[net < 0] = 0
        return net

    @staticmethod
    def f_prime(net):
        net[net <= 0] = 0
        net[net > 0] = 1
        return net


class Sigmoid:
    @staticmethod
    def activation(net):
        return expit(net, net)

    @staticmethod
    def f_prime(net):
        net *= (1. - net)
