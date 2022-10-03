import random
import numpy as np

class AlgoritmoPerceptronSimple:
    def __init__(self, stimulus, expected_output, learningRate):
        self.stimulus = stimulus
        self.expected_output = expected_output
        self.learningRate = learningRate

    def calculateActivation(self, w, u):
        aux = self.stimulus[u]
        h = 0
        for i in range(0, len(aux)):
            h += w[i] * aux[i]
        if h < 0:
            return -1
        else:
            return 1
    
    def calculateError(self, w):
        error = 0
        for i in range(0, len(self.stimulus)):
            o = self.calculateActivation(w, i)
            error += np.abs(self.expected_output[i] - o)
        return error
        
    
    def run(self):
        i = 0
        w = np.zeros(len(self.stimulus[0]))
        error = 0
        error_min = 10000000000
        cota = 100000
        while error_min > 0 and i < cota:
            u = random.randint(0, len(self.stimulus) - 1)
            o = self.calculateActivation(w, u)
            deltaW = ((self.expected_output[u] - o) * self.learningRate) * self.stimulus[u]
            w = np.add(w , deltaW)
            error = self.calculateError(w)
            if error < error_min:
                error_min = error
                w_min = w
            i += 1
        if(i >= cota):
            print("Corté por cota")
        return w

stimulus = np.array([[1,-1, 1], [1,1, -1], [1,-1, -1], [1,1, 1]])
expected_output = np.array([1, 1, -1, -1])
A = AlgoritmoPerceptronSimple(stimulus, expected_output, 0.5)
print(A.run())
#print(np.multiply(2,[1,1]))
