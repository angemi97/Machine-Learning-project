import numpy as np

def sigmoid(z):     
    """
        Sigmoid activation function.
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(sigmoid_out):
    """
        Calculates the derivative of the result taken from the sigmoid.
    """
    return sigmoid_out * (1 - sigmoid_out)

def initDimensions(X, Y, num_hidden):
    """
        Initializes dimensions of the single layer perceptron.

        Returns:
            input:  Number of inputs.
            hidden: Number of neurons in hidden layer.
            output: Number of neurons in output layer (init to 1). 
    """

    input = X.shape[0]
    hidden = num_hidden
    output = Y.shape[0]
    
    return input, hidden, output

def initParameters(input, hidden, output):
    """
        Initializes weights by producing random numbers
        from a normal distribution.

        Returns:
            Dict with initialized weights.
    """

    W1 = np.random.randn(hidden, input) * 0.01
    b1 = np.zeros((hidden,1))
    W2 = np.random.randn(output, hidden) * 0.01
    b2 = np.zeros((output,1))
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

def forwardPropagation(X, parameters):
    """
        Forward proagation step.
        Step 1: Computes the weighted sums:
                Z1 for neurons in hidden layer.
                Z2 for neuron in outer layer.
        Step 2: Activations using sigmoid function:
                o1 for neurons in hidden layer.
                o2 for neuron in outer layer.

        Returns:
            Output of the sigmoid function in outer layer o2.
            Dict with the results of each step.
    """
    
    # For hidden layers:
    Z1 = np.dot(parameters['W1'], X) + parameters['b1']
    o1 = sigmoid(Z1)

    # For outer layer:
    Z2 = np.dot(parameters['W2'], o1) + parameters['b2']
    o2 = sigmoid(Z2)
    
    results = {"Z1": Z1, "Z2": Z2, "o1": o1, "o2": o2}

    return o2, results

def computeCostGrad(o2, Y):
    """
        Computes either the Binary Cross Entropy.
        Returns:
            Result of the cost function.
    """
    return binaryCrossEntropy(o2, Y)

def binaryCrossEntropy(o2, Y):
    """
        Computes and returns the Binary Cross Entropy cost.
    """
    m = Y.shape[1]
    
    logprobs = np.dot(Y,np.log(o2).T) + np.dot((1-Y),np.log((1-o2)).T)
    cost = -logprobs/m
    cost = float(np.squeeze(cost)) 

    a = (o2 - Y)
    b = np.dot(o2, (1 - o2.T))
    gradcost = a / b
    # gradcost = float(np.squeeze(np.mean(gradcost)))     
    return cost, gradcost 

def backwardPropagation(parameters, cache, X, Y):
    """
        Backward propagation step. 
        Step 1: 
            Calculate gradients for weights and biases.
        Step 2: 
            Update the weights and biases which are going 
            to be used for gradient descent.
    """
    m = X.shape[1]
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    o1 = cache['o1']
    o2 = cache['o2']

    # Backward propagation: calculate dW1, db1, dW2, db2. 
    dZ2 = np.subtract(o2, Y)
    dW2 = dZ2 * sigmoid_derivative(o2)
    db2 = np.sum(dZ2,axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2)
    dW1 = dZ1 * sigmoid_derivative(o1)
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    W2_update = np.dot(o1, dW2.T)/m
    b2_update = db2/m

    W1_update = np.dot(X, dW1.T)/m
    b1_update = db1/m
    
    grads = {"dW1": W1_update, "db1": b1_update, "dW2": W2_update, "db2": b2_update}
    
    return grads

def updateParameters(parameters, grads, n):
    """
        Updates the parameters with gradient descent.
        Returns the new parameters.
    """
    # Retrieve each parameter from the dictionaries "parameters" and "grads"
    W1 = parameters['W1']
    dW1 = grads['dW1']
    b1 = parameters['b1']
    db1 = grads['db1']
    W2 = parameters['W2']
    dW2 = grads['dW2']
    b2 = parameters['b2'] 
    db2 = grads['db2']
    
    # Update rule for each parameter

    W1 = W1 - n*dW1.T
    b1 = b1 - n*db1
    W2 = W2 - n*dW2.T
    b2 = b2 - n*db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

def trainedModel(X, Y, Xval, Yval, n_h, n=0.01, tolerance=5, binary=False, max_iterations=3000):
    """
        Trains the mlp model with binary cross entropy or mean squared error.
        Variables:
            X: x_train.
            Y: y_train.
            n_h: Number of neurons at hidden layer.
            num_iterations: Upper limit of number of epochs allowed during the training.
                            Training might end before reaching it with early stopping.
            cost_avg: The average cost value of the model at epoch i.
            cost_check: The average cost value of the model at epoch i-1.
            counter: Counts how many times the average cost of epoch i is >= the average cost of epoch i-1.
            costs: History data of the returned cost after each epoch.
        Returns: 
            costs: Cost history.
            epoch: Number of epochs during training.
            parameters: The parameters of the trained mlp model.
    """

    # Preparing shapes for X and Y vectors.
    X = X.T
    Y = Y.T
    Xval = Xval.T
    Yval = Yval.T
    print(f"X shape: {X.shape} \nY shape: {Y.shape} \nXval shape: {Xval.shape} \nYval shape: {Yval.shape}")

    # Initializing variables used during training.
    epoch = 0
    cost_check = 0.0
    cost_avg = 0.0
    counter = 0
    costs = []
    
    # Setting random seed for debugging.
    np.random.seed(43) 

    # Initializing dimensions.
    n_x, n_h, n_y = initDimensions(X, Y, n_h)
    
    # Initializing parameters.
    parameters = initParameters(n_x, n_h, n_y)

    # Printing the cost function that's going to be used.
    if binary:
        print(f"Starting training the mlp using Binary Cross Entropy for calculating costs.")
    else:
        print(f"Starting training the mlp using Mean Squared Error for calculating costs.")

    # Training loop using Gradient Descent with early stopping.
    while(True):

        # Forward propagation. (training)
        o2, cache = forwardPropagation(X, parameters)
        
        # Forward propagation. (validation)
        o2_val, cache_val = forwardPropagation(Xval, parameters)

        # Cost function.
        cost, _ = computeCostGrad(o2_val, Yval)
        costs.append(cost)
        cost_avg = np.mean(costs)

        # Backward propagation.
        grads = backwardPropagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update.
        parameters = updateParameters(parameters, grads, n)

        if round(cost_check, 4) <= round(cost_avg, 4):
            counter += 1
            if counter == tolerance:
                break
        else:
            counter = 0
        # Print the cost every 500 iterations
        if epoch % 500 == 0:
            print (f"The cost after epoch {epoch}: {round(cost, 4)}")
        
        cost_check = cost_avg
        epoch += 1

        # if epoch == max_iterations: break

    print("-------------------------------------------------------------")
    if epoch == max_iterations:
        print(f"Stopped at epoch: {epoch} with final cost: {round(cost, 4)}")
    else:
        print(f"Average cost did not get reduced for {tolerance} times. \nStopped at epoch: {epoch} with final average cost: {round(cost, 4)}")
    print("-------------------------------------------------------------")

    return cost, costs, epoch, parameters

def gradcheck_binary(X, t, option):
    
    X = X.T
    t = t.T
    #W = np.random.rand(*Winit.shape)
    epsilon = 1e-6
    
    _list = np.random.randint(X.shape[1], size=5)
    x_sample = np.array(X[:, _list])
    t_sample = np.array(t[:, _list])
    
    n_h = 4

    n_x, n_h, n_y = initDimensions(x_sample, t_sample, n_h)
    parameters = initParameters(n_x, n_h, n_y)

    o2, cache = forwardPropagation(x_sample, parameters)
    grads = backwardPropagation(parameters, cache, x_sample, t_sample)
    parameters = updateParameters(parameters, grads, n=0.1)

    if(option == 1):
        weights = parameters['W1']
    else:
        weights = parameters['W2']

    W = weights

    if(option == 1):
        gradValues = grads['dW1'].T
    else:
        gradValues = grads['dW2'].T

    
    
    gradEw = gradValues
    
    
    numericalGrad = np.zeros(gradEw.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad
    for k in range(numericalGrad.shape[0]):
        for d in range(numericalGrad.shape[1]):
            
            #add epsilon to the w[k,d]
            w_tmp = np.copy(W)
            w_tmp[k, d] += epsilon
            
            if(option == 1):
                parameters['W1'] = w_tmp
            else:
                parameters['W2'] = w_tmp

            o2, cache = forwardPropagation(x_sample, parameters)
            e_plus = computeCostGrad(o2, t_sample)
            
            #subtract epsilon to the w[k,d]
            w_tmp = np.copy(W)
            w_tmp[k, d] -= epsilon
            
            if(option == 1):
                parameters['W1'] = w_tmp
            else:
                parameters['W2'] = w_tmp

            o2, cache = forwardPropagation(x_sample, parameters)
            e_minus = computeCostGrad(o2, t_sample)
            
            #approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)
    return ( gradEw, numericalGrad )

def predict(parameters, X, Y):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    X = X.T
    Y = Y.T
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forwardPropagation(X, parameters)
    threshold = 0.5 
    
    predictions = (A2 > threshold)
    
    accuracy = '{:0.3f}'.format(np.mean( predictions.astype('int') == Y ) )
    print(f'Accuracy: {accuracy} ')

    return accuracy

