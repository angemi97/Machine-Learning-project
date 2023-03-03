import numpy as np
import matplotlib.pyplot as plt
import sys

def sigmoid(z): 
    #Sigmoid function
    return 1.0 / (1.0 + np.exp(-z))

def ComputeCostGrad( X, y, theta, _lambda):
    #Computes cost function and the gradient.

    #Sigmoid hypothesis. 
    h = sigmoid( X.dot(theta) )

    #Cost function.
    cur_j = (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h)))

    #Gradient function.
    grad = np.mean((y-h) * X.T, axis=1)

    #Regularization.
    if _lambda != 0: 
        reg = (_lambda / (2.0 ) ) * np.sum(theta**2)
        cur_j = cur_j - reg
        reg = _lambda * theta
        grad = grad - reg
    return cur_j, grad

def ComputeLogisticRegression( X, y, X_val, y_val, tot_iter=100, _lambda=0.0, alpha=0.01 ):
    #Computes Logistic Regression using Gradient Ascend and returns
    #history data and the new weights.

    #Weight function theta, J_train and J_test histories.
    theta = np.zeros(X.shape[1])
    J_train = []
    J_test = []
    
    #Computing train/test errors in tot_iter epochs.
    for i in range( tot_iter ):  
        train_error, train_grad = ComputeCostGrad( X, y, theta, _lambda )
        test_error, _ = ComputeCostGrad( X_val, y_val, theta, _lambda )
        
        #Updates weights by adding gradient values.
        theta += alpha * train_grad

        #Stores history data.
        J_train.append( train_error )
        J_test.append( test_error )
        
    return J_train, J_test, theta

def predict(theta, X):
    #PREDICT Predict whether the label is 0 or 1 using learned Logistic 
    #Regression weights theta
    #   p = PREDICT(theta, X) computes the predictions for X using a 
    #   threshold at 0.5 (i.e., if sigmoid( X * theta ) >= 0.5, predict 1)

    m = X.shape[0]
    
    p = np.zeros( (m,1) )
    p = sigmoid( np.dot(X,theta) )
    
    prob = p
    p = p > 0.5 - 1e-6
    
    return p, prob

def regularizationResults(train_stats, test_stats):
    #Prints a plot for train data using different λ during L2 regularization.
    plt.figure( figsize=(10,4) )
    plt.subplot( 1, 2, 1 )

    #Making 10 plot lines to analyze the train results.
    for i in range( 0, len(train_stats), 10 ):
        plt.plot( np.arange( len(train_stats[i]['history']) ), train_stats[i]['history'], label='λ=' + str( round(train_stats[i]['lambda'], 4)) )
    
    plt.xlabel( 'Number of iterations' )
    plt.ylabel( 'Train Error' )
    plt.legend()

    #Prints a plot for test data using different λ during L2 regularization.
    plt.subplot( 1, 2, 2)

    #Making 10 plot lines to analyze the train results.
    for i in range( 0, len(train_stats), 10 ):
        plt.plot( np.arange( len(test_stats[i]['history']) ), test_stats[i]['history'], label='λ='+ str( round(test_stats[i]['lambda'], 4) ) )
    
    plt.xlabel( 'Number of iterations' )
    plt.ylabel( 'Test error' )
    plt.legend()
    plt.tight_layout()
    plt.show()

def trainTestResults(J_train, J_test):
    #Plots (train, test) for a single run of Logistic Regression.
    plt.figure( figsize=(10,4) )
    plt.subplot( 1, 2, 1 )
    plt.plot( np.arange( len(J_train) ), J_train, label='λ=0' )
    plt.xlabel( 'Number of iterations' )
    plt.ylabel( 'Train Error' )
    plt.legend()
    plt.subplot( 1,2,2)
    plt.plot( np.arange( len(J_test) ), J_test, label='λ=0' )
    plt.xlabel( 'Number of iterations' )
    plt.ylabel( 'Test error' )
    plt.legend()
    plt.show()

def testMultipleIterations(x_train, y_train, x_test, y_test):
    #Prints accuracy of training and test sets for 10, 100, 1.000, 10.000 epochs.
    for i in range(1,5):
        #Number of epochs.
        tot_iter = 10**i

        print("-------------------------------------------------------------")
        print(f"Testing accuracy of Logistic Regression with {tot_iter} iterations.")
        J_train, J_test, theta = ComputeLogisticRegression( x_train, y_train, x_test, y_test, tot_iter, _lambda=0.0 )

        #Printing train and test error plots.
        #trainTestResults(J_train, J_test)

        p_train, prob_train = predict( theta, x_train )
        p_test, prob_test = predict( theta, x_test )
        print( 'Accuracy of training set', '{:0.3f}'.format(np.mean( p_train.astype('int') == y_train ) ) )
        print( 'Accuracy of testing set', '{:0.3f}'.format(np.mean( p_test.astype('int') == y_test ) ) ) 
        print("-------------------------------------------------------------\n")
        
def testRegularization(x_train, y_train, x_test, y_test, x_val, y_val):
    #Tests Logistic Regression results using different λ values and
    #prints plot results.

    #Number of splits in [10**-4, 10] range.
    splits = 100

    #Dictionaries to save stats for different λ values.
    test_stats, train_stats, theta_stats, acc_stats, lamda = [], [], [], [], []


    #Testing λ values in [10**-4, 10] range.
    for idx, value in enumerate(np.linspace(1e-4, 10, num=splits)):
        percent_done = '{:0.2f}'.format((idx+1)/splits*100)
        print(f"Computing Logistic Regression for λ = {'{:0.5f}'.format(value)} ({percent_done}% done).", end='\r')

        J_train, J_test, theta = ComputeLogisticRegression( x_train, y_train, x_test, y_test, _lambda=value )

        p_val, prob_test = predict( theta, x_val )
        accuracy = '{:0.3f}'.format(np.mean( p_val.astype('int') == y_val ) )
        
        #Updating history dictionaries.
        test_stats.append( {'history' : J_test, 'lambda': value} )
        train_stats.append( {'history' : J_train, 'lambda': value} )
        theta_stats.append( {'history' : theta, 'lambda': value} )
        acc_stats.append(accuracy)
        lamda.append(value)

    #Finding index of the maximum accuracy found in the results.
    index = acc_stats.index(max(acc_stats))
    print(f"\nMax accuracy in validation data achieved with λ = {lamda[index]}: {max(acc_stats)}.")

    p_val, prob_test = predict( theta_stats[index]['history'], x_test )
    accuracy = '{:0.3f}'.format(np.mean( p_val.astype('int') == y_test ) )
    print(f'Accuracy of test set with λ = {lamda[index]}: {accuracy} ')
    
    #Prints a plot for train data using different λ during L2 regularization.
    regularizationResults(train_stats, test_stats)
