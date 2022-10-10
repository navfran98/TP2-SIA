import csv
import math
from NonLinearPerceptron import NonLinearPerceptron
from LinearPerceptron import LinearPerceptron
from SimplePerceptron import SimplePerceptron
import numpy as np
import imageio

import matplotlib.pyplot as plt

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

def makeGIF(hist):
    chart_size = 3
    dot_size = 25
    dot_color_true = 'green'
    dot_color_false = 'red'
    plt.xlim(-chart_size, chart_size)
    plt.ylim(-chart_size, chart_size)
    plt.scatter(1, 1, s=dot_size, c=dot_color_true)
    plt.scatter(1, -1, s=dot_size, c=dot_color_false)
    plt.scatter(-1, 1, s=dot_size, c=dot_color_false)
    plt.scatter(-1, -1, s=dot_size, c=dot_color_false)
    plt.title("Iteracion - 0")
    plt.plot()
    plt.savefig(f'PNGs/line-{0}.png')
    plt.close()
    for i in range(0,len(hist)):
        plt.xlim(-chart_size, chart_size)
        plt.ylim(-chart_size, chart_size)
        plt.scatter(1, 1, s=dot_size, c=dot_color_true)
        plt.scatter(1, -1, s=dot_size, c=dot_color_false)
        plt.scatter(-1, 1, s=dot_size, c=dot_color_false)
        plt.scatter(-1, -1, s=dot_size, c=dot_color_false)
        p1, p2 = getLinePointsEj1(hist[i])
        plt.title("Iteracion - " + str(i+1))
        plt.plot(p1, p2, marker = 'o')
        plt.savefig(f'PNGs/line-{i+1}.png')
        plt.close()
    with imageio.get_writer('line.gif', mode='i') as writer:
        for i in range(0, len(hist)+1):
            image = imageio.imread(f'PNGs/line-{i}.png')
            # Para que el gif no vaya tan rapido cada imagen 
            # la agregamos 5 veces
            for i in range(0,5):
                writer.append_data(image)def makeGIF(hist):
    chart_size = 3
    dot_size = 25
    dot_color_true = 'green'
    dot_color_false = 'red'
    plt.xlim(-chart_size, chart_size)
    plt.ylim(-chart_size, chart_size)
    plt.scatter(1, 1, s=dot_size, c=dot_color_true)
    plt.scatter(1, -1, s=dot_size, c=dot_color_false)
    plt.scatter(-1, 1, s=dot_size, c=dot_color_false)
    plt.scatter(-1, -1, s=dot_size, c=dot_color_false)
    plt.title("Iteracion - 0")
    plt.plot()
    plt.savefig(f'PNGs/line-{0}.png')
    plt.close()
    for i in range(0,len(hist)):
        plt.xlim(-chart_size, chart_size)
        plt.ylim(-chart_size, chart_size)
        plt.scatter(1, 1, s=dot_size, c=dot_color_true)
        plt.scatter(1, -1, s=dot_size, c=dot_color_false)
        plt.scatter(-1, 1, s=dot_size, c=dot_color_false)
        plt.scatter(-1, -1, s=dot_size, c=dot_color_false)
        p1, p2 = getLinePointsEj1(hist[i])
        plt.title("Iteracion - " + str(i+1))
        plt.plot(p1, p2, marker = 'o')
        plt.savefig(f'PNGs/line-{i+1}.png')
        plt.close()
    with imageio.get_writer('line.gif', mode='i') as writer:
        for i in range(0, len(hist)+1):
            image = imageio.imread(f'PNGs/line-{i}.png')
            # Para que el gif no vaya tan rapido cada imagen 
            # la agregamos 5 veces
            for i in range(0,5):
                writer.append_data(image)



def runEj2NonLinear():
    betas =  np.arange(0.05, 2.0, 0.05)
    #Sacar de csv los stimuli ([x1,x2,x3]) y los expected_outputs (y)
    stimulus, expected_output = parseCSV("./Ejercicio1y2/aprendizaje.csv")
    test_stimulus, test_expected = parseCSV("./Ejercicio1y2/testeo.csv")
    learningRate = 0.001
    for beta in betas:
        A = NonLinearPerceptron(stimulus, expected_output, learningRate, beta)
        w,e = A.run()

        print(A.test(test_stimulus[0])- A.escalate_value(test_expected[0]))
        
        print(w)

# ---------------------------------------------------------------------------- #

runEj2NonLinear()