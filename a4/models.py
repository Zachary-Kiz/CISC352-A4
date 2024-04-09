import nn

class PerceptronModel(object):
    def __init__(self, dim):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dim` is the dimensionality of the data.
        For example, dim=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dim)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x_point):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x_point: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x_point)

    def get_prediction(self, x_point):
        """
        Calculates the predicted class for a single data point `x_point`.

        Returns: -1 or 1
        """
        "*** YOUR CODE HERE ***"
        score = self.run(x_point)
        if nn.as_scalar(score) >= 0:
            return 1
        else:
            return -1


    def train_model(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:  # Continuously train until the model is fully accurate on the dataset
            weight_was_updated = False  # Initialize flag to false at the start of each pass through the dataset
            for x, y in dataset.iterate_once(1):  # Iterate through each example in the dataset
                prediction = self.get_prediction(x)  # Predict the class based on current weights
                actual = nn.as_scalar(y)  # Convert the expected label from a node to a scalar value
                
                if prediction != actual:  # If the prediction does not match the actual label
                    self.w.update(actual, x)  # Update weights to correct the misclassification
                    weight_was_updated = True  # Indicate that at least one weight was updated in this pass
                    
            if not weight_was_updated:  # If no weights were updated in this pass, the model is fully trained
                break
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w = nn.Parameter(1, 1)
        self.b = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        xw = nn.Linear(x, self.w)
        return nn.AddBias(xw, self.b)


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_pred = self.run(x)
        return nn.SquareLoss(y_pred, y)
    
    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate=0.01
        target_loss = 0.02
        while True:
            total_loss = 0
            batches = 0
            for x, y in dataset.iterate_once(1):
                loss = self.get_loss(x, y)
                total_loss = nn.as_scalar(loss)
                batches +=1
                grad_w, grad_b = nn.gradients([self.w,self.b],loss)
                # Compute gradients and update parameters
                self.w.update(-learning_rate, grad_w )
                self.b.update(-learning_rate, grad_b)
                if total_loss <= target_loss:
                    break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_weights = nn.Parameter(784, 128)  # Weights between input layer and hidden layer
        self.hidden_bias = nn.Parameter(1, 128)  # Bias for hidden layer
        
        # For the output layer
        self.output_weights = nn.Parameter(128, 10)  # Weights between hidden layer and output layer
        self.output_bias = nn.Parameter(1, 10) 
        
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        hidden = nn.ReLU(nn.AddBias(nn.Linear(x, self.hidden_weights), self.hidden_bias))
        
        # Output layer
        output = nn.AddBias(nn.Linear(hidden, self.output_weights), self.output_bias)
        return output


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        logits = self.run(x)
        return nn.SoftmaxLoss(logits, y)


    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate=0.1
        epochs=13
        for epoch in range(epochs):
            total_loss = 0
            for x, y in dataset.iterate_once(100):  
                loss = self.get_loss(x, y)
                total_loss += nn.as_scalar(loss)
                
                # Compute gradients
                gradients = nn.gradients( [self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias], loss)
                
                # Update parameters
                self.hidden_weights.update(-learning_rate, gradients[0])
                self.hidden_bias.update(-learning_rate, gradients[1])
                self.output_weights.update(-learning_rate, gradients[2])
                self.output_bias.update(-learning_rate, gradients[3])
        
                valid_acc = dataset.get_validation_accuracy()
                if valid_acc >= 0.975:
                    break
