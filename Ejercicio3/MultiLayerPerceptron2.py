import numpy as np
import random
import matplotlib.pyplot as plt
# --- Variables Indexs ---
# i = indice de la neurona de salida
# j = indice de la neurona de capa intermedia
# k = indice de entrada o de capa anterior
# m = indice de capa intermedia
# p = cantidad de datos

stimuli = [[1,-1, 1],[1,1, -1],[1,-1, -1],[1,1, 1]]
expected_output = [[1],[1],[-1],[-1]]

class MultiLayerPerceptron:
    def __init__(self, stimuli, expected_output, learning_rate,layers,neurons_per_layer):
        self.stimuli = stimuli
        self.out = expected_output
        self.n = learning_rate
        self.beta = 0.5
        self.layers = layers
        self.neurons_per_layer = neurons_per_layer
    
        self.neurones = []
        self.exits = []
        #Initialize W's with random small numbers
        for i in range(0,self.layers):
            self.neurones.append([])
            for j in range(0,self.neurons_per_layer):
                self.neurones[i].append({})
                if(i == 0):
                    self.neurones[i][j]["w"] = np.random.uniform(size=len(self.stimuli[0]), low=-1, high=1)
                else:
                    self.neurones[i][j]["w"] = np.random.uniform(size=neurons_per_layer, low=-1,high=1)
                    
                self.neurones[i][j]["m"] = i
        
        for i in range(0,len(expected_output[0])):
            self.exits.append({})
            self.exits[i]["w"]= np.random.uniform(size=self.neurons_per_layer, low=-1, high=1)
        
    def g(self,x):
        return np.tanh(self.beta * x)
    
    def g_dx_dt(self, x):
        return self.beta * (1 - self.g(x))
    
    def run(self):
        error_min = 1000000
        cota = 100000
        i = 0
        while error_min > 0.0001 and i < cota:
            u = random.randint(0,len(self.stimuli)-1)
            self.propagation(self.stimuli[u],0)
            self.calculate_exit()
            self.backtracking(u)
            self.update_connections(self.layers,u)
            i += 1

        ret = []

        for i in range(0, self.layers):
            for n in self.neurones[i]:
                ret.append(n["w"])
        return ret
        # return self.neurones[0][0]["w"], self.neurones[0][1]["w"]
        #Agarro el O. Esta es mi salida, y tengo que sacar el error
            
        #      O1    
        #    **
        #   V21 V22
        #   V11 V12 
        # V01 V02 V03
    
    def propagation(self,prev_layer,m):
        #neurones[i] = [{"w" : w, "h": h, "v" : v, "m": m}]
        if(m == self.layers): return
        
        for i in range(0,self.neurons_per_layer):
            self.neurones[m][i]["h"] = 0
            for j in range(0,len(prev_layer)):
                if m != 0:
                    self.neurones[m][i]["h"] += self.neurones[0][i]["w"][j] * prev_layer[j]["v"]
                if m == 0:
                    self.neurones[m][i]["h"] += self.neurones[0][i]["w"][j] * prev_layer[j]
                    
            self.neurones[m][i]["v"] = self.g(self.neurones[m][i]["h"])
        
        self.propagation(self.neurones[m],m+1)
                
    def calculate_exit(self):
        m = self.layers - 1
        for i in range(0,len(self.out[0])):
            self.exits[i]["h"] = 0
            for j in range(0,self.neurons_per_layer):
                self.exits[i]["h"] += self.exits[i]["w"][j] * self.neurones[m][j]["v"]
            self.exits[i]["v"] = self.g(self.exits[i]["h"])
    
    def backtracking(self,u):
        self.calculate_exit_delta(u)
        for m in reversed(range(0,self.layers)):
            for i in range(0,self.neurons_per_layer):
                self.neurones[m][i]["delta"] = self.g_dx_dt(self.neurones[m][i]["h"])* (self.out[u] - self.neurones[m][i]["v"])
    
    def calculate_exit_delta(self,u):
        for i in range(0,len(self.out[0])):
            self.exits[i]["delta"] = self.g_dx_dt(self.exits[i]["h"])*(self.out[u][i] - self.exits[i]["v"])
            
    def update_exit_connections(self):
        for i in range(0,len(self.out[0])):
            for j in range(0,self.neurons_per_layer):
                self.exits[i]["w"][j] += self.n * self.exits[i]["delta"] * self.neurones[self.layers-1][j]["v"]
    
    def update_first_layer(self,u):
        for i in range(0,self.neurons_per_layer):
            for j in range(0,len(self.stimuli[0])):
                self.neurones[0][i]["w"][j] += self.n * self.neurones[0][i]["delta"] * self.stimuli[u][j]
    
    def update_connections(self,m,u):
        if m == 0: return self.update_first_layer(u)
        if m == self.layers:
            self.update_exit_connections()
        else:
            for i in range(0,self.neurons_per_layer):
                for j in range(0,self.neurons_per_layer):
                    # print(self.n * self.neurones[m][i]["delta"] * self.neurones[m-1][j]["v"])
                    self.neurones[m][i]["w"][j] += self.n * self.neurones[m][i]["delta"] * self.neurones[m-1][j]["v"]
        
        self.update_connections(m-1,u)
    
    def propagation_test(self, st):
        return self.propagation_test_rec(st, 0)

    def propagation_test_rec(self,st,m):
        #neurones[i] = [{"w" : w, "h": h, "v" : v, "m": m}]
        if(m == self.layers):
            ret = np.zeros(len(self.out[0]))
            for j in range(0,len(ret)):
                for i in range(0,len(st)):
                    ret[j] +=  self.exits[j]["w"][i] * st[i]
                ret[j] = np.tanh(self.beta * ret[j])
            return ret
        
        aux = np.zeros(self.neurons_per_layer)
        for i in range(0,self.neurons_per_layer):
            for j in range(0,len(st)):
                aux[i] += self.neurones[m][i]["w"][j] * st[j]
                
        return self.propagation_test_rec(aux,m+1)
        

    def propagation_test2(self,st,w1,w2):
        aux1 = 0
        aux2 = 0
        ret = 0
        for i in range(0, len(st)):
            aux1 += st[i] * w1[i]
            aux2 += st[i] * w2[i]
        aux = [aux1, aux2]
        for i in range(0, len(self.exits[0]["w"])):
            ret += aux[i] * self.exits[0]["w"][i] 
        return np.tanh(self.beta * ret) 
        

# ----------------------------------
def runEj3_1():
    optimus = MultiLayerPerceptron(stimuli,expected_output,0.01,1,2)
    w1,w2 = optimus.run()


    for st in optimus.stimuli:
        print(f'{st} -> {optimus.propagation_test2(st,w1,w2)}')

    A = w1
    B = w2

    xs = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    y1 = []
    y2 = []
    for x in xs:
        y1.append((-A[0] - A[1]*x)/A[2])
        y2.append((-B[0] - B[1]*x)/B[2])

    plt.scatter(1, 1, color = "red")
    plt.scatter(1, -1, color = "blue")
    plt.scatter(-1, 1, color = "blue")
    plt.scatter(-1, -1, color="red")
    plt.plot(xs, y1)
    plt.plot(xs, y2)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.show()

# ----------------------------------
train_set = [
    #0
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0 ],
    #2
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1 ],
    #4
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 ],
    #5
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0 ],
    #7
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0 ],
    #8
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    #9
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0 ],
]

train_set_output = [[-1], [-1], [-1], [1], [1], [-1], [1]]

test_set = [
    #1
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    #3
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    #6
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
]

test_set_output = [[1], [1], [-1]]
# ----------------------------------
def runEj3_2():
    digits = MultiLayerPerceptron(train_set,train_set_output,0.01, 2, 2)
    ws = digits.run()

    for st in train_set:
        print(f' -> {digits.propagation_test(st)}')

#----------------------------------
data = [
    #0
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0 ],
    #1
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0 ],
    #2
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1 ],
    #3
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0 ],
    #4
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 ],
    #5
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0 ],
    #6
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0 ],
    #7
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0 ],
    #8
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    #9
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0 ],
]

data_output = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
]

data_test = [
    #1
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    #3
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    #6
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
]

data_test_output = [[1], [3], [6]]
#----------------------------------
def runEj3_3():
    digits_v2 = MultiLayerPerceptron(data, data_output,0.01, 2, 2)
    ws = digits_v2.run()

    for st in test_set:
        print(f' -> {digits_v2.propagation_test(st)}')
#----------------------------------

runEj3_3()
