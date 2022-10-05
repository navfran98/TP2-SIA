import random
import numpy as np
import csv


class LinearPerceptron:
    def __init__(self, stimulus, expected_output, learningRate):
        self.stimulus = stimulus
        self.expected_output = expected_output
        self.learningRate = learningRate

    def calculateActivation(self, w, u):
        aux = self.stimulus[u]
        h = 0
        for i in range(0, len(aux)):
            h += w[i] * aux[i]
        return h
    
    def calculateError(self, w):
        error = 0
        for i in range(0, len(self.stimulus)):
            o = self.calculateActivation(w, i)
            error += (self.expected_output[i] - o)**2
        return error/2
    
    def calculateDeltaW(self,i,o):
        ret = 0
        for u in range(0,len(self.stimulus)):
            ret += (self.expected_output[u] - o)*self.stimulus[u]
        
        return self.learningRate * ret

def parseCSV(csvname):
    file = open(csvname)
    csvreader = csv.reader(file)
    next(csvreader) #header
    stimulus = []
    expected_output = []
    for row in csvreader:
        stimulus.append([1,row[0],row[1],row[2]])
        expected_output.append(row[3])
    stimulus = np.array(stimulus)
    expected_output = np.array(expected_output)
    
    return stimulus, expected_output
   
        


#Sacar de csv los stimuli ([x1,x2,x3]) y los expected_outputs (y)
stimulus, expected_output = parseCSV("./Ejercicio2/TP2-ej2-conjunto.csv")


A = SimplePerceptronAlgorithm(stimulus, expected_output, 0.5)
linear_perceptron = LinearPerceptron(stimulus,expected_output,0.5)
print(A.run(linear_perceptron.stimulus,linear_perceptron.calculateError,linear_perceptron.calculateActivation,linear_perceptron.calculateDeltaW))

#print(np.multiply(2,[1,1]))
