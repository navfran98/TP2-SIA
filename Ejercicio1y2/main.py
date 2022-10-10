import csv
import math
from NonLinearPerceptron import NonLinearPerceptron
from LinearPerceptron import LinearPerceptron
from SimplePerceptron import SimplePerceptron
import numpy as np
import matplotlib.pyplot as plt
import imageio

import matplotlib.pyplot as plt

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

def getLinePointsEj1(w):
    evaluate = lambda y, m, x, b: (m/y) * x + (b/y)
    x1 = 5
    x2 = -5
    y1 = evaluate(w[2],-w[1],x1,-w[0])
    y2 = evaluate(w[2],-w[1],x2,-w[0])
    return [x1, x2], [y1, y2]

def makeGIF(errors, it, beta):
    for i in range(0,len(errors)):
        parameters = {'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.labelsize': 14}
        plt.rcParams.update(parameters)
        plt.figure(figsize=(7,5))

        plt.ylabel("Error")
        plt.xlabel("Iterations")

        plt.title("Beta - " + str(beta[i]))
        # plt.ylim(0, 100000)
        plt.semilogy()
        plt.plot(it[1:], errors[i][1:])
        plt.savefig(f'Ejercicio1y2/PNGs/line-{i}.png')
        plt.close()
    with imageio.get_writer('Ejercicio1y2/PNGs/line.gif', mode='i') as writer:
        for i in range(0, len(errors)):
            image = imageio.imread(f'Ejercicio1y2/PNGs/line-{i}.png')
            # Para que el gif no vaya tan rapido cada imagen 
            # la agregamos 5 veces
            for i in range(0,5):
                writer.append_data(image)

def runEj1And(threshold, learningRate):
    stimulus = np.array([[1,-1, 1], [1,1, -1], [1,-1, -1], [1,1, 1]])
    expected_output = np.array([-1, -1, -1, 1])
    # threshold = 0.5 #Umbral
    # learningRate = 0.5
    A = SimplePerceptron(stimulus, expected_output, learningRate, threshold)
    wmin, e, it = A.run()
    ret = []
    for st in stimulus:
        ret.append(A.test(wmin, st))
    return wmin, e, it, ret

def runEj1ExclusiveOr(threshold, learningRate):
    stimulus = np.array([[1,-1, 1], [1,1, -1], [1,-1, -1], [1,1, 1]])
    expected_output = np.array([1, 1, -1, -1])
    # threshold = 0.5 #Umbral
    # learningRate = 0.5
    A = SimplePerceptron(stimulus, expected_output, learningRate, threshold)
    return A.run()
    
def runEj2Linear(learningRate):
    #Sacar de csv los stimuli ([x1,x2,x3]) y los expected_outputs (y)
    stimulus, expected_output = parseCSV("./Ejercicio1y2/aprendizaje.csv")
    # learningRate = 0.001
    A = LinearPerceptron(stimulus, expected_output, learningRate)
    return A.run()

def runEj2NonLinear(beta, learningRate):
    #Sacar de csv los stimuli ([x1,x2,x3]) y los expected_outputs (y)
    stimulus, expected_output = parseCSV("./Ejercicio1y2/aprendizaje_2.csv")
    # learningRate = 0.001
    # beta = 0.5
    A = NonLinearPerceptron(stimulus, expected_output, learningRate, beta)
    wmin, e, it = A.run()

    test_stimulus, test_expected = parseCSV("./Ejercicio1y2/testeo_2.csv")
    ret = {"x":[], "y":[]}
    for i in range(0, len(test_expected)):
        ret["x"].append(i+1)
        ret["y"].append(abs(A.test(test_stimulus[i])- A.escalate_value(test_expected[i])))
    return wmin, e, it, ret


# def makeGIF(hist):
#     chart_size = 3
#     dot_size = 25
#     dot_color_true = 'green'
#     dot_color_false = 'red'
#     plt.xlim(-chart_size, chart_size)
#     plt.ylim(-chart_size, chart_size)
#     plt.scatter(1, 1, s=dot_size, c=dot_color_true)
#     plt.scatter(1, -1, s=dot_size, c=dot_color_false)
#     plt.scatter(-1, 1, s=dot_size, c=dot_color_false)
#     plt.scatter(-1, -1, s=dot_size, c=dot_color_false)
#     plt.title("Iteracion - 0")
#     plt.plot()
#     plt.savefig(f'PNGs/line-{0}.png')
#     plt.close()
#     for i in range(0,len(hist)):
#         plt.xlim(-chart_size, chart_size)
#         plt.ylim(-chart_size, chart_size)
#         plt.scatter(1, 1, s=dot_size, c=dot_color_true)
#         plt.scatter(1, -1, s=dot_size, c=dot_color_false)
#         plt.scatter(-1, 1, s=dot_size, c=dot_color_false)
#         plt.scatter(-1, -1, s=dot_size, c=dot_color_false)
#         p1, p2 = getLinePointsEj1(hist[i])
#         plt.title("Iteracion - " + str(i+1))
#         plt.plot(p1, p2, marker = 'o')
#         plt.savefig(f'PNGs/line-{i+1}.png')
#         plt.close()
#     with imageio.get_writer('line.gif', mode='i') as writer:
#         for i in range(0, len(hist)+1):
#             image = imageio.imread(f'PNGs/line-{i}.png')
#             # Para que el gif no vaya tan rapido cada imagen 
#             # la agregamos 5 veces
#             for i in range(0,5):
#                 writer.append_data(image)

# ---------------------------------------------------------------------------- #
def graph(x, y):
    parameters = {'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.labelsize': 14}
    plt.rcParams.update(parameters)
    plt.figure(figsize=(7,5))

    plt.plot(x, y)
    # plt.semilogy()
    plt.ylabel("Error")
    plt.xlabel("Iterations")
    plt.show()

def graphScatter(x, y):
    parameters = {'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.labelsize': 14}
    plt.rcParams.update(parameters)
    plt.figure(figsize=(7,5))
    plt.title("Error = | Expected Value - Output |")
    
    plt.scatter(x, y)
    # plt.ylabel("Error")
    plt.xlabel("Test Stimuli")
    plt.show()


# learningRate = 0.5
# threshold = 0.5
# wmin, e, it, ret = runEj1And(threshold, learningRate)
# print(f'WMin -> {wmin}')
# print(f'Test output -> {ret}')
# p1, p2 = getLinePointsEj1(wmin)
# plt.scatter(1,1, color="red")
# plt.scatter(-1,1, color="blue")
# plt.scatter(1,-1, color="blue")
# plt.scatter(-1,-1, color="blue")
# plt.plot(p1, p2)
# plt.show()
# graph(it, e)

# learningRate = 0.5
# threshold = 0.5
# wmin, e, it = runEj1ExclusiveOr(threshold, learningRate)
# print(f'WMin -> {wmin}')
# graph(it, e)

# learningRate = 0.0001
# wmin, e, it = runEj2Linear(learningRate)
# print(f'WMin -> {wmin}')
# graph(it[1:], e[1:])

learningRate = 0.0001
# betas = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2]
# errors = []
# for b in betas:
#     wmin, e, it = runEj2NonLinear(b, learningRate)
#     errors.append(e)
# makeGIF(errors, it, betas)
wmin, e, it, ret = runEj2NonLinear(0.5 ,learningRate)
print(f'WMin -> {wmin}')
graphScatter(ret["x"], ret["y"])


