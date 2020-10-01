import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        np.random.seed(2)
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### DONE: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1/(1+np.exp(-x))
        self.activation_derivative = lambda x : x*(1-x)
        
        
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    #def test_activation_function():
    #    network = NeuralNetwork()
    #    expected = 0.73106
    #    result = network.activation_function(1)
    #    assert expected == result
    
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        #print("features= ", type(features))
        #print("targets= ", targets)
        #print("features.head", features.head())
        features = np.array(features)
        #print("features.shape", features.shape)
        targets = np.array(targets)
        #print(features[0])
        
        n_records = features.shape[0]
        # print("features.shape", features.shape, "targets.shape", targets.shape)    features.shape (15435, 56) targets.shape (15435, 3)
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        counter = 0
        for X, y in zip(features, targets):

            final_outputs, hidden_outputs = self.forward_pass_train(X)  
            
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o)
            counter += 1
            if counter == 10: break
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)
        print("training finished", counter)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # DONE: Hidden layer - Replace these values with your calculations.
        if loglevel == "debug": print("============ starting forward prop =============")
        if loglevel == "debug": print("X shape", X.shape)
        
        if loglevel == "debug": print("self.weights_input_to_hidden.shape", self.weights_input_to_hidden.shape)
        hidden_inputs = np.dot(X,  self.weights_input_to_hidden) 
        if loglevel == "debug": print("hidden_inputs shape and value", hidden_inputs.shape, hidden_inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        if loglevel == "debug": print("hidden_outputs shape and value", hidden_outputs.shape, hidden_outputs)
        if loglevel == "debug": print("self.weights_hidden_to_output shape", self.weights_hidden_to_output.shape)
        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs,  self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs
        
        if loglevel == "debug": print("final_outputs shape and value", final_outputs.shape, final_outputs)
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###
        if loglevel == "debug": print("====== starting backprop ======")
        #print("X", X)
        #print("y", y)
        # TODO: Output error - Replace this value with your calculations.
        error = y[0] - final_outputs # Output layer error is the difference between desired target and actual output.
        if loglevel == "debug": print("error shape and value ", error.shape, error)
        
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = final_outputs * error
        if loglevel == "debug": print("output_error_term shape and value", output_error_term.shape, output_error_term)
        
        # я переставил это, раньше стояло перед предыдующим TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)
        if loglevel == "debug": print("self.weights_hidden_to_output.T", self.weights_hidden_to_output.T.shape)
        if loglevel == "debug": print("hidden_error shape and value", hidden_error.shape, hidden_error)
        
        hidden_error_term = hidden_error * self.activation_derivative(hidden_outputs)
        if loglevel == "debug": print("hidden_error_term shape and value", hidden_error_term.shape, hidden_error_term)
        
        # Weight step (input to hidden)
        if loglevel == "debug": print("before delta_weights_i_h shape and sum", delta_weights_i_h.shape, delta_weights_i_h.sum())
        if loglevel == "debug": print("X.T shape", X.T.shape)
        
        newXT = np.expand_dims(X, axis=0).T
        if loglevel == "debug": print("newXT shape", newXT.shape)
        new_hidden_error_term = np.expand_dims(hidden_error_term, axis=0)
        if loglevel == "debug": print("new_hidden_error_term shape", new_hidden_error_term.shape)
        deltaweightsstep_i_h = np.dot(newXT, new_hidden_error_term)
        if loglevel == "debug": print("deltaweightsstep_i_h shape", deltaweightsstep_i_h.shape)
        delta_weights_i_h += deltaweightsstep_i_h
        # Weight step (hidden to output)
        if loglevel == "debug": print("current delta_weights_h_o shape", delta_weights_h_o.shape)
        if loglevel == "debug": print("hidden_output.T shape", hidden_outputs.T.shape)
        if loglevel == "debug": print("output_error_term shape", output_error_term.T.shape)
        new_hidden_outputs = np.expand_dims(hidden_outputs, axis=0).T
        new_output_error_term = np.expand_dims(output_error_term, axis=0)
        if loglevel == "debug": print("new_hidden_outputs shape", new_hidden_outputs.shape)
        if loglevel == "debug": print("new_output_error_term shape", new_output_error_term.shape)
        deltaweightsstep_h_o = np.dot(new_hidden_outputs, new_output_error_term)
        if loglevel == "debug": print("deltaweightsstep_h_o shape", deltaweightsstep_h_o.shape)
        delta_weights_h_o += deltaweightsstep_h_o 
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        import pandas as pd
        if loglevel == "debug":print("starting weights update...")
        if loglevel == "debug": print("w_h_0 before", self.weights_hidden_to_output.sum())
        if loglevel == "debug": print("w_i_h before", self.weights_input_to_hidden.sum())
        if log_display_values == "True": print("self.weights_hidden_to_output value before", self.weights_hidden_to_output)
        if log_display_values == "True": print("self.weights_input_to_hidden value before", self.weights_input_to_hidden)
        self.weights_hidden_to_output += self.lr*delta_weights_h_o/n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr*delta_weights_i_h/n_records # update input-to-hidden weights with gradient descent step
        if loglevel == "debug": print("w_h_0 after", self.weights_hidden_to_output.sum())
        if loglevel == "debug":print("w_i_h after", self.weights_input_to_hidden.sum())
        if log_display_values == "Values": print("self.weights_hidden_to_output value after") 
        if log_display_values == "True": print(pd.DataFrame(data=self.weights_hidden_to_output))
        if log_display_values == "True": print("self.weights_input_to_hidden value after", self.weights_input_to_hidden)
                                               
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = hidden_inputs = np.dot(features,  self.weights_input_to_hidden) 
        hidden_outputs = self.activation_function(hidden_inputs)
        
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = None # signals into final output layer
        final_outputs = None # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1
#loglevel= "debug"
loglevel = "debug" #debug Normal
log_display_values = "False" #True
