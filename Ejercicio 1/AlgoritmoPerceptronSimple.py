import random
import numpy as np

class AlgoritmoPerceptronSimple:
    def __init__(self, entrada, salida, tasaAprendizaje):
        entrada = entrada
        salida = salida
        tasaAprendizaje = tasaAprendizaje

    def calcularExcitacion(self, w, u):
        aux = self.entrada[u]
        h = 0
        for i in range(0, len(aux)):
            h += w[i] + aux[i]
        return h
    
    def calcularError(self, w)):
        for i in range(0, len(self.entrada)):
            h = self.calcularExcitacion(w, i)
            if h < 0:
                O = -1
            else:
                O = 1
            np.abs(self.salida[i] - O)
    
    def run(self):
        i = 0
        w = np.zeros(len(self.salida), 1)
        error = 0
        error_min = 100000000
        cota = 1000
        while error_min > 0 and i < 1000:
            u = random.randint(0, len(self.entrada) - 1)
            h = self.calcularExcitacion(w, u)
            if h < 0:
                O = -1
            else:
                O = 1
            deltaW = self.tasaAprendizaje * (self.salida[u] - O) * self.entrada[u]
            w = w + deltaW
            error = self.calcularError(w)
            if error < error_min:
                error_min = error
                w_min = w
            i += 1

entrada = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
salida = [−1, −1, −1, 1]

A = AlgoritmoPerceptronSimple(entrada, salida, 0.5)
A.run()