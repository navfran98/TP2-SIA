import numpy as np
import random

# --- Variables Indexs ---
# i = indice de la neurona de salida
# j = indice de la neurona de capa intermedia
# k = indice de entrada o de capa anterior
# m = indice de capa intermedia
# p = cantidad de datos

# --- Estimulos y outputs ---
stimulus = [[1,-1, 1],[1,1, -1],[1,-1, -1],[1,1, 1]]
eOutput = [1,1,-1,-1]

class MultiLayerPerceptronA:
    def __init__(self, st, out, lr):
        self.st = st
        self.out = out
        self.n = lr
        self.beta = 1
        # Matriz con los pesos
        self.ws =[
            [1, -1], #W
            np.random.uniform(size=3, low=-1, high=1), #w1
            np.random.uniform(size=3, low=-1, high=1), #w2
        ]

    def g(self,x):
        return np.tanh(self.beta * x)
    
    def gDerivative(self, x):
        return self.beta * (1 - self.g(x))

    def calculateError(self):
        error = 0
        # Por cada estimulo
        for index,u in enumerate(self.st):
            # No tenemos este ciclo for porque solo hay una O (salida)
            eo = self.out[index]
            for idx,W in enumerate(self.ws[0]):
                a = 0
                for index1,w1 in enumerate(self.ws[idx+1]):
                    a += w1 * u[index1]
                # Como estamos usando un perceptron lineal
                # G es la identidad.
                error += (eo - self.g(W*self.g(a)))**2
                
        return 0.5 * error

    # Lo revise mil veces creo que esta perfecta
    # esta funcion, si alguien mas la revisa god
    def dWOut(self):
        dW = 0
        # Por cada entrada
        for index,u in enumerate(self.st):
            for idx,W in enumerate(self.ws[0]):
                a = 0
                for index1,w1 in enumerate(self.ws[idx+1]):
                    a += w1 * u[index1]
                # La derivada esta implicita porque es 1
                dW += (self.out[index] - self.g(W*self.g(a)))*self.g(a)
        return self.n * dW

    #vas a querer armar un [,,] 
    def dWMid(self):
        dW = [[0,0,0], [0,0,0]]
        # Por cada entrada
        for j in range(0, 2):
            for index,u in enumerate(self.st):
                dw_aux = 0
                for idx,W in enumerate(self.ws[0]):
                    h = 0
                    for index1,w1 in enumerate(self.ws[idx+1]):
                        h += w1 * u[index1]
                dw_aux += (self.out[index] - self.g(h)) * self.ws[0][j] * self.st[index]
            self.ws[j+1] = self.n * dw_aux
        

    def run(self):
        # V00 v01 v02 serian los 3 valores del stimulo
        # propago para arriba y calculo V11 V12 y dsp V21 que seria el de salida
        # calculo el delta para la capa de salida (delta21)
        # retropropago el delta y obtengo el delta11 y delta12
        # actualizo todos los pesos wij usando el deltaWij
        # calculo el error
        i = 0
        error = 0
        error_min = 10000000000
        cota = 100000
        while error_min > 0.0001 and i < cota:
            # Ovtengo un random self.st[u]
            u = random.randint(0, len(self.st) - 1)
            # Obtengo los V0 V1 y V2 al propagar hacia arriba
            V0 = [self.st[u][0], self.st[u][1], self.st[u][2]]
            V1, h1 = self.getV1(V0)
            V2, h2 = self.getV2(V1)
            # Obtengo el delta de la salida
            delta_2_1 = self.getDeltaOutput(V2, u, h2)
            delta_1 = self.getDeltaMid(delta_2_1, h1)
    
            self.updateW(delta_2_1, delta_1, V1, V0)
            error = self.calculateError()
            if error < error_min:
                error_min = error
                w_min = self.ws
            i += 1
        if(i >= cota):
            print("CortÃ© por cota")
            print(error)
            print(self.ws)
            self.w = w_min
        return w_min
    
    def getV1(self, v0s):
        vs = [0, 0]
        hs = [0, 0]
        for i in range(0,2):
            aux = 0
            for j in range(0, len(v0s)):
                aux += self.ws[i+1][j] * v0s[j]
            vs[i] = self.g(aux)
            hs[i] = aux
        return vs, hs
    
    def getV2(self, v1s):
        ret = 0
        for j in range(0, len(v1s)):
            ret += self.ws[0][j] * v1s[j]
        return self.g(ret), ret

    def getDeltaOutput(self, v2, u, h):
        return self.gDerivative(h) * (self.out[u] - v2)
    
    def getDeltaMid(self, delta_2_1, h1):
        ret = [0, 0]
        ret[0] = self.gDerivative(h1[0]) * self.ws[0][0] * delta_2_1
        ret[1] = self.gDerivative(h1[1]) * self.ws[0][1] * delta_2_1
        return ret

    def updateW(self, d2, d1, v1, v0):
        deltaW = [[0,0], [0,0,0], [0,0,0]]
        deltaW[0] = [self.n * d2 * v1[0], self.n * d2 * v1[1]]
        for i in range(0,2):
            for j in range(0, len(v0)):
                deltaW[i+1][j] = self.n * d1[i] * v0[j]
        for i in range(0, len(self.ws)):
            for j in range(0, len(self.ws[i])):
                self.ws[i][j] = self.ws[i][j] + deltaW[i][j]

A = MultiLayerPerceptronA(stimulus, eOutput, 0.01)
A.run()
