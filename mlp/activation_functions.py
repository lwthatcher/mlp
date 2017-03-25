
class ReLU:

    @staticmethod
    def activation(net):
        net[net < 0] = 0

    @staticmethod
    def f_prime(net):
        net[net <= 0] = 0
        net[net > 0] = 1
