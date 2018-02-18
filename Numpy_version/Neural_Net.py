import numpy as np 
from tqdm import tqdm #(pip install tqdm)
np.random.seed(42)


class MNISTNetwork(object):
    

    def __init__(self, saved_values=None, data=None, labels=None, learning_rate=0.3):


        self.data=data
        self.labels=labels
        #Learning rate for the optimisation algorithm
        self.learning_rate=learning_rate


        self.input_nodes=784
        self.output_nodes=10
        #Hidden layer neurons
        self.layer_1_nodes=30


        if saved_values is None:
            #Weights and biases for layer 1
            self.weights_1 = np.random.randn(self.layer_1_nodes, self.input_nodes)
            self.biases_1 = np.random.randn(self.layer_1_nodes, 1)

            #weights and biases for output layer 1
            self.out_weights = np.random.randn(self.output_nodes, self.layer_1_nodes)
            self.out_biases = np.random.randn(self.output_nodes, 1)

        #Allow for the possibility of reloading learned weights/biases
        else:
            weights_1, biases_1, out_weights, out_biases=saved_values

            self.weights_1=weights_1
            self.biases_1=biases_1

            self.out_weights=out_weights
            self.out_biases=out_biases




    def think(self, x, ret_full=False):
        # Multiply the input with weights and find its sigmoid activation for all layers

        #Layer 1
        z1=np.dot(self.weights_1, x)+self.biases_1
        layer1 = self.sigmoid(z1)

        #Pass this through to layer 2
        z2=np.dot(self.out_weights, layer1)+self.out_biases
        output = self.sigmoid(z2)

        if ret_full:
            return (layer1, output), (z1, z2)
        return output



    def backpropagation(self, data, targets):

        #Backpropagation algorithm

        #Make empty arrays 
        nabla_b = [np.zeros(b) for b in [self.biases_1.shape, self.out_biases.shape]]
        nabla_w = [np.zeros(w) for w in [self.weights_1.shape, self.out_weights.shape]]


        #Reshape the data to be a (N, 1) array
        inputs=data.reshape(-1, 1)
        #Pass them through the network
        (layer1, output), (z1, z2)=self.think(inputs, ret_full=True)
        
        
        #Get the error on the output:
        outputError = output - targets



        #The actual algorithm: see e.g her: http://neuralnetworksanddeeplearning.com/chap2.html
        delta = outputError * self.sigmoid_prime(z2)        

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta.reshape(-1, 1), layer1.reshape(-1, 1).T)

        delta = np.dot(self.out_weights.T, delta) * self.sigmoid_prime(z1)
        nabla_b[-2] = delta
        nabla_w[-2] = np.dot(delta.reshape(-1, 1), data.reshape(-1, 1).T)

        return nabla_b, nabla_w


    def train(self, n_steps):

        for i in tqdm(range(n_steps)):
            
            #Randomly shuffle the order of the data or labels
            indices=np.arange(self.data.shape[0])
            np.random.shuffle(indices)
            self.data=self.data[indices, ...]
            self.labels=self.labels[indices, ...]
            #Update each batch of data

            data_batches=self.data.reshape(-1, 10, 784)
            label_batches=self.labels.reshape(-1, 10, 10)

            #Loop through each batch of data
            for j, (data, targets) in enumerate(zip(data_batches, label_batches)):
                
                nabla_b = [np.zeros(b) for b in [self.biases_1.shape, self.out_biases.shape]]
                nabla_w = [np.zeros(w) for w in [self.weights_1.shape, self.out_weights.shape]]
                for d, t in zip(data, targets):
                    delta_nabla_b, delta_nabla_w=self.backpropagation(d, t.reshape(-1, 1))
                    
                    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                    
                self.weights_1 -= self.learning_rate*nabla_w[-2]
                self.biases_1 -= self.learning_rate*nabla_b[-2]
                self.out_weights -= self.learning_rate*nabla_w[-1]
                self.out_biases -= self.learning_rate*nabla_b[-1]



    #Activation function
    @staticmethod
    def sigmoid(z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))
   
    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))

if __name__=='__main__':

    # Standard library
    import cPickle
    import gzip

    #Data packaged up and taken from here: https://github.com/mnielsen/neural-networks-and-deep-learning
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()

    data=training_data[0]
    correct_labels=training_data[1]

    labels=np.zeros((data.shape[0], 10))
    #Turn it into 'one hot' form, i.e [0000001000]
    labels[np.arange(data.shape[0]), correct_labels]=1.0

    net=MNISTNetwork(data=data, labels=labels)
    net.train(30)

    #Save the weights and biases
    np.save('weights_1.npy', net.weights_1)
    np.save('weights_output.npy', net.out_weights)

    np.save('biases_1.npy', net.biases_1)
    np.save('biases_output.npy', net.out_biases)


    predictions=np.zeros(test_data[0].shape[0])

    for i, (data, label) in enumerate(zip(test_data[0], test_data[1])):
        predictions[i]=np.argmax(net.think(test_data[0][i].reshape(-1, 1)))

    print r"Accuracy on test data: {}%".format(len(np.where(predictions==test_data[1])[0])/float(len(predictions))*100.0)

