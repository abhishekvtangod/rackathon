import numpy as np
from scipy.special import expit

class Classifier:
    def __init__(self, inodes, hnodes, onodes, lrate):
        #Initialize number of input, hidden, and output nodes 
        self.inodes = inodes 
        self.hnodes = hnodes
        self.onodes = onodes
        
        #Layers 
        self.l1, self.l2, self.l3 = [], [], [] 

        #Learning rate 
        self.lrate = lrate
        
        #Link weights! Because matrix multiplication is (each row in first matrix) *
        #(each column in second matrix), each row must represent the full incoming weights
        #for a node in the second layer. It will be multiplied by each of the second layer
        #nodes (AKA the inputs), so its size must be second_layer_recipients (#rows) 
        #x incoming_nodes(#cols)
        
        # Here is a helpful visual (weights) * (inputs):
        # (1, 5, 7)       (5)
        # (6, 6, 3)  *    (4)
        # (6, 2, 5)       (3)
        
     
        # The first step would be: 1(5) + 5(4) + 7(3)
        #Translated to English, this is: Top incoming weight(top input) + 2nd top-most incoming weight
        #(2nd top-most input) + bottom incoming weight(bottom input) 
        
        #weights from input to hidden
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.wih_normal = (self.wih - self.wih.min()) / (self.wih.max() - self.wih.min())
        #weights from hidden to output 
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.who_normal = (self.who - self.who.min()) / (self.who.max() - self.who.min())

        #Define activation (sigmoid) function 
        self.activation_function = expit
        

    def train(self, inputs_list, targets_list):
        #Convert inputs and targets to 2D matrices 
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T 

        #calculate final output from querying the network 
        hidden_outputs = self.activation_function(np.dot(self.wih, inputs))
        final_outputs = self.activation_function(np.dot(self.who, hidden_outputs))

        #error is the target outputs minus the actual ones 
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)



        self.who +=  self.lrate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lrate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs)) 






        
    def query(self, inpt_list):
        # print(inpt_normal)
        #convert input list to 2D array 
        inputs = np.array(inpt_list, ndmin=2).T #2 dimensions so that we have row x height, and transpose to allow multiplication 

        #calculate hidden layer outputs with matrix multiplicaion through the activation function 
        hidden_outputs = self.activation_function(np.dot(self.wih, inputs))
        outputs = self.activation_function(np.dot(self.who, hidden_outputs))

        #return the outputs for use in the network
        return outputs 


