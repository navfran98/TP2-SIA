import csv
from NonLinearPerceptron import NonLinearPerceptron
from LinearPerceptron import LinearPerceptron
from SimplePerceptron import SimplePerceptron
import numpy as np

def parseCSV(csvname):
    file = open(csvname)
    csvreader = csv.reader(file)
    next(csvreader) #header
    stimulus = []
    expected_output = []
    for row in csvreader:
        stimulus.append([1,float(row[0]),float(row[1]),float(row[2])])
        expected_output.append(float(row[3]))
    stimulus = np.array(stimulus)
    expected_output = np.array(expected_output)
    file.close()
    return stimulus, expected_output

def runEj1And():
    stimulus = np.array([[1,-1, 1], [1,1, -1], [1,-1, -1], [1,1, 1]])
    expected_output = np.array([-1, -1, -1, 1])
    threshold = 0.5 #Umbral
    learningRate = 0.5
    A = SimplePerceptron(stimulus, expected_output, learningRate, threshold)
    print(A.run())

def runEj1ExclusiveOr():
    stimulus = np.array([[1,-1, 1], [1,1, -1], [1,-1, -1], [1,1, 1]])
    expected_output = np.array([1, 1, -1, -1])
    threshold = 0.5 #Umbral
    learningRate = 0.5
    A = SimplePerceptron(stimulus, expected_output, learningRate, threshold)
    print(A.run())
    
def runEj2Linear():
    #Sacar de csv los stimuli ([x1,x2,x3]) y los expected_outputs (y)
    stimulus, expected_output = parseCSV("./Ejercicio1y2/TP2-ej2-conjunto.csv")
    learningRate = 0.001
    A = LinearPerceptron(stimulus, expected_output, learningRate)
    print(A.run())

def runEj2NonLinear():
    #Sacar de csv los stimuli ([x1,x2,x3]) y los expected_outputs (y)
    stimulus, expected_output = parseCSV("./Ejercicio1y2/TP2-ej2-conjunto.csv")
    learningRate = 0.001
    beta = 0.003
    A = NonLinearPerceptron(stimulus, expected_output, learningRate, beta)
    print(A.run())

# ---------------------------------------------------------------------------- #

runEj2NonLinear()