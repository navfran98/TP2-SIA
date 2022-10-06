import random
import numpy as np

class NonLinearPerceptron:

    def __init__(self, stimulus, expected_output, learningRate,beta):
        self.stimulus = stimulus
        self.expected_output = expected_output
        self.learningRate = learningRate
        self.beta = beta
        self.w = np.zeros(len(self.stimulus[0]))
    
    def getH(self, w, u): #o
        aux = self.stimulus[u]
        h = 0
        for i in range(0, len(aux)):
            h += float(w[i]) * float(aux[i])
        return h
    
    def calculateActivation(self,w,u):
        h = self.getH(w,u)
        return np.tanh(self.beta*h)
    
    def calculateActivationDerivative(self,w,u):
        h = self.getH(w,u)
        return self.beta * (1 - ((np.tanh(self.beta * h)**2)))
    
    def calculateError(self, w):
        error = 0
        for u in range(0, len(self.stimulus)):
            o = self.calculateActivation(w, u)
            error += (self.expected_output[u] - o)**2
        return error/2
    
    def calculateDeltaW(self,w,u):
        ret = np.array([0,0,0,0])
        for u in range(0,len(self.stimulus)):
            aux = ((self.expected_output[u] - self.calculateActivation(w, u)) * self.calculateActivationDerivative(w, u)) * self.stimulus[u]
            ret = np.add(ret, aux)
        return self.learningRate * ret
    
    def run(self):
        i = 0
        error = 0
        error_min = 10000000000
        cota = 100000
        while error_min > 0.001 and i < cota:
            u = random.randint(0, len(self.stimulus) - 1)
            deltaW = self.calculateDeltaW(self.w,u)
            self.w = np.add(self.w , deltaW)
            error = self.calculateError(self.w)
            if error < error_min:
                error_min = error
                w_min = self.w
            i += 1
        if(i >= cota):
            print("Corté por cota")
        return w_min
    

