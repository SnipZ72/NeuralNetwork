import numpy

def sigmoid(x):
    return 1/(1 + numpy.exp(-x))

class NeuralNetwork:
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate

        #self.activation_function = lambda x: sigmoid(x)

        pass

    def train(self, inputs_list, targets_list):
        
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = sigmoid(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = sigmoid(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        print(self.wih)

        pass

    def guess(self, inputs_list):
        
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)

        hidden_outputs = sigmoid(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)

        final_outputs = sigmoid(final_inputs)

        return final_outputs


inputs_nodes = 2
hidden_nodes = 4
output_nodes = 1

n = NeuralNetwork(inputs_nodes, hidden_nodes, output_nodes, 0.3)

trainA = [0,1]
trainB = [1,1]
trainC = [1,0]
trainD = [0,0]

for x in range(10000):
    n.train(trainA, 1)
    n.train(trainB, 0)
    n.train(trainC, 1)
    n.train(trainD, 0)

print ("0,1 " + str(n.guess([0,1])))
print ("1,0 " + str(n.guess([1,0])))
print ("1,1 " + str(n.guess([1,1])))
print ("0,0 " + str(n.guess([0,0])))
