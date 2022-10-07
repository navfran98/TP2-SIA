import csv
import math
from NonLinearPerceptron import NonLinearPerceptron
from LinearPerceptron import LinearPerceptron
from SimplePerceptron import SimplePerceptron
import numpy as np

def parseCSV(csvname):
    file = open(csvname)
    csvreader = csv.reader(file)
    next(csvreader) #header
    stimulus = []
    col0 = []
    col1 = []
    col2 = []
    expected_output = []
    for row in csvreader:
        stimulus.append([1,float(row[0]),float(row[1]),float(row[2])])
        #col0.append(float(row[0]))
        #col1.append(float(row[1]))
        #col2.append(float(row[2]))
        expected_output.append(float(row[3]))
    #col0 = normalize(col0)
    #col1 = normalize(col1)
    #col2 = normalize(col2)
    #for i in range(0,len(col0)):
    #   stimulus.append([1,col0[i],col1[i],col2[i]])
    #expected_output = normalize(expected_output)
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
    stimulus, expected_output = parseCSV("./Ejercicio1y2/aprendizaje.csv")
    learningRate = 0.001
    A = LinearPerceptron(stimulus, expected_output, learningRate)
    print(A.run())

def runEj2NonLinear():
    #Sacar de csv los stimuli ([x1,x2,x3]) y los expected_outputs (y)
    stimulus, expected_output = parseCSV("./Ejercicio1y2/aprendizaje.csv")
    test_stimulus, test_expected = parseCSV("./Ejercicio1y2/testeo.csv")
    learningRate = 0.001
    beta = 0.4
    A = NonLinearPerceptron(stimulus, expected_output, learningRate, beta)
    w = A.run()

    print(A.test(test_stimulus[0])- A.escalate_value(test_expected[0]))
    
    print(w)

# ---------------------------------------------------------------------------- #

runEj2NonLinear()