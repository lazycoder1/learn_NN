import numpy as np

#This is the neural net class, which has the methods to train and predict the outputs for the input
class neural_net:

    #This method gets called when the class is initialized
    def __init__(self):

        #You seed the random. This will make sure that the random number generated every time you run the program is the same
        #This makes it easier to debug
        np.random.seed(1)

        #We initialize the neural network weights with the random weights to start at
        neural_net.weights = 2 * np.random.random((6,1)) -1


    #All the outputs after being multiplied with the weights are pass to the sigmoid function
    #which converts the outputs to be in the range 0 - 1
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    #The derivative of the sigmoid
    #This function is used to adjust the weights
    def sigmoid_derivative(self,x):
        return x*(1-x)

    #Train the neural net using the training data and the outputs for the same
    def train(self,training_data,training_outputs,iterations):
        for i in range(iterations):
            output = self.predict(training_data)

            error = training_outputs - output

            adjustment = np.dot(training_data.T, error * self.sigmoid_derivative(output))


            self.weights += adjustment

    def predict(self,input):

        return self.sigmoid(np.dot(input,self.weights))


if __name__ == '__main__':

    #The output for the following input should be same as the 4th column or ( 1st OR 4th column )
    input = np.array([[1,0,0,1,1,0],
                      [0,1,1,1,0,1],
                      [1,0,1,1,0,1],
                      [0,1,1,0,1,1],
                      [0,0,0,1,0,1]])

    print("Input\n",input)

    output = np.array([[1,1,1,0,1]]).T

    print("Output\n",output)

    neural_net = neural_net()

    neural_net.train(input,output,1000)

    #Display the weights of the neural net after it gets trained
    print("Neural net weights ")
    print(neural_net.weights)

    np.set_printoptions(precision=2)
    print('\n\nprediction\n',neural_net.predict(np.array([[0,1,1,0,1,0],
                                                          [0,0,1,1,1,0],
                                                          [1,0,1,0,0,1],
                                                          [0,1,0,0,0,1],
                                                          [0,0,0,0,0,1]])))






















