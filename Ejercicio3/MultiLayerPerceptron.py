
class MultiLayerPerceptron:
    def __init__(self, stimulus, expected_output, learning_rate, threshold):
        self.stimulus = stimulus
        self.expected_output = expected_output
        self.learning_rate = learning_rate
        self.threshold = threshold

    # ------ Utils --------
    def getH(self, u, W, w, j_range):
        h = 0
        for j in range(0, j_range):
            v_j = self.getV(u, j, w)
            h += W[j] * v_j
        return h

    def calculateActivation(self, value):
        return value

    def calculateActivationDerivative(self, value):
        return 1

    def getV(self, u, j, w):
        ret = 0
        for k in range(0, len(self.stimulus[u])):
            ret += w[j][k] * self.stimulus[u][k]
        return self.calculateActivation(ret)
    # ---------------------

    # ------ Calculamos el error --------
    def getG(self, j_range, W, w, u):
        ret = 0
        for j in range(0, j_range):
            aux = 0
            for k in range(0, len(self.stimulus[u])):
                aux += w[j][k] * self.stimulus[u][k]
            ret += W[j] * self.calculateActivation(aux)
        return self.calculateActivation(ret)

    def calculateError(self, j, W, w):
        error = 0
        for u in range(0, len(self.stimulus)):
            error += (self.expected_output[u] - self.getG(j, W, w, u))**2
        return 0.5 * error
    # -----------------------------------

    #------ Calculamos deltW Salida --------
    def getDeltaWOutput(self, W, w, j_range):
        ret = 0
        for j in range(0, j_range):
            for u in range(0, len(self.stimulus)):
                h = self.getH(u, W, w, j_range)
                o = self.calculateActivation(h)
                v_j = self.getV(u, j, w)
                g_derivative_i = self.calculateActivationDerivative(h)
                g_derivative_j = self.calculateActivationDerivative(v_j)
                ret += (self.expected_output[u] - o) * g_derivative_i * W[j] * g_derivative_j * self.stimulus[u]
        return self.learning_rate * ret
    #---------------------------------------


    #------ Calculamos deltW Intermedia --------
    def getDeltaWOutput(self, W, w, j):
        ret = 0
        for u in range(0, len(self.stimulus)):
            h = self.getH(u, W, w, j)
            o = self.calculateActivation(h)
            v_j = self.getV(u, j, w)
            ret += (self.expected_output[u] - o) * g_derivative * v_j
        return self.learning_rate * ret
    #-------------------------------------------
