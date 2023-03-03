from load_dataset import load_dataset, split_dataset, reshape_data, normalization
from logistic_regression import testMultipleIterations, testRegularization
from slp import *
import numpy as np
import matplotlib.pyplot as plt

def main():


    #-------------------------------------------------------------#
    #                         Part A                              #
    #         Downloading and editing data to prepare for         #
    #                   Logistic Regression.                      #
    #-------------------------------------------------------------#

    print("Loading dataset...")
    x_train, y_train, x_test, y_test = load_dataset('Part A')
    print("Loaded class 5, 6 data! \n")

    print("Rescaling from [0,255] to [0,1]...")
    x_train, x_test = normalization(x_train, x_test)

    print("Spliting data to 80% train and 20% validation... \n")
    x_train, y_train, x_val, y_val = split_dataset(x_train, y_train)

    print("Reshaping data...\n")
    x_train, x_val, x_test = reshape_data(x_train, x_val, x_test)
    print(f"x_train new shape: {x_train.shape} \nx_val new shape: {x_val.shape} \nx_test new shape: {x_test.shape}\n")
    

    #-------------------------------------------------------------#
    #                         Part B                              #
    #   Computing Logistic Regression for binary classification.  #
    #   Printing accuracy results and plots for regularization.   #
    #-------------------------------------------------------------#

    print("Testing Logistic Regression with different number of epochs.")
    testMultipleIterations(x_train, y_train, x_test, y_test)

    print("Testing Logistic Regression with L2 regularization, printing plot results for different λ values.")
    testRegularization(x_train, y_train, x_test, y_test, x_val, y_val)

    #-------------------------------------------------------------#
    #                         Part C                              #
    #             Creating single layer mlp model.                #
    #                       Implements:                           #
    #                    Gradient checking.                       #
    #                     Early stopping.                         #
    #                                                             #
    #-------------------------------------------------------------#

    # Reshaping Y vectors in preparation for training the mlp.
    y_train = y_train.reshape(-1,1)
    y_val = y_val.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    # Training MLP model with 1 hidden layer consisting of M = 32 neurons.
    print(f"Training MLP model \nParameters: M = {2} n = {0.3}")
    c, costs, epoch, parameters = trainedModel(x_train, y_train, x_val, y_val, 8, n=0.3, binary=True)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(format(0.1, 'f')))
    plt.show()    

    
    # Gradient Checking for weights w1, w2
    # gradEw, numericalGrad = gradcheck_binary(x_train, y_train, 1)
    # print( "The difference estimate for gradient of w1 is : ", np.max(np.abs(gradEw - numericalGrad)) )
    # gradEw, numericalGrad = gradcheck_binary(x_train, y_train, 2)
    # print( "The difference estimate for gradient of w2 is : ", np.max(np.abs(gradEw - numericalGrad)) )
    


    history_n = []
    history_E = []
    history_M = []
    bestModel = []
    bestCost = 100000
    index = 0
    counter = 0
    # Testing different n values and number of neutrons in hidden layer for training.
    f = open("results.txt", "w")
    f.close()
    for value in np.linspace(1e-5, 0.5, num=10):
        for i in range(1,11):
            M = 2**i
            print(f"Testing with n = {value}, M = {M}...")
            c, costs, epoch, parameters = trainedModel(x_train, y_train, x_val, y_val, M, n=value, binary=True)
            
            history_n.append(value)
            history_E.append(epoch)
            history_M.append(M)

            if counter == 0 or c<bestCost: 
                bestCost = c
                bestModel = parameters
                index = counter # Save index of best model.

            f = open("results.txt", "a")
            f.write(f"n = {value} | M = {M} | E = {epoch} | cost = {round(c, 4)}\n")
            f.close()

            counter += 1
    
    foundError = False
    ##  Results reader.
    ##  Used if we want to run the fit the above algorithm and the prediction in different instances.
    ##  Used for E(n) plots, for the different M values.

    with open('results.txt') as f:
        idx = 0
        n = 0
        M = 0
        E = 0
        history_n = []
        history_E = []
        history_M = []
        cost = 0
        bestCost = 0
        for line in f:
            info = line.strip().split(" | ")

            if len(info) != 4:
                foundError = True
                print(f"Error reading results.txt file at line {idx}.")
            else:
                cost = info[3].strip().split("=")[1].strip()
                history_M.append(int(info[1].strip().split("=")[1].strip()))
                history_n.append(float(info[0].strip().split("=")[1].strip()))
                history_E.append(int(info[2].strip().split("=")[1].strip()))

                # For calculating best model
                # if idx == 0 or cost<bestCost:
                #     bestCost = cost
                #     n    = info[0].strip().split("=")[1].strip()
                #     M    = info[1].strip().split("=")[1].strip()
                #     E    = info[2].strip().split("=")[1].strip()

            idx += 1
    # For calculating best model
    #     c, costs, epoch, parameters = trainedModel(x_train, y_train, x_val, y_val, int(M), float(n), binary=True)
    #     bestModel = parameters

    colors = plt.cm.jet(np.linspace(0,1,10)) 
    M_value = 2 
    for i in range(10): 
        epochs = []  
        n = [] 
        for j in range(len(history_M)): 
            if history_M[j] == M_value: 
                if history_E[j] > 200: # Removing outliers
                    epochs.append(history_E[j]) 
                    n.append(history_n[j]) 
        plt.plot(n, epochs, color=colors[i], label='M=' + str(M_value)) 
        M_value *= 2 

    plt.xlabel("Learning rate") 
    plt.ylabel("Epochs") 
    plt.title("E(n) function plot \nfor different sizes of hidden layer (M)") 

    plt.legend() 
    plt.savefig("epochs-learning rate.png") 
    print("Plot saved to 'epochs-learning rate.png'") 
    # plt.show()

    with open("bestModel.txt", "w") as f2:
        if not foundError:
            print(f"Best model after parameter optimization:\nn = {history_n[index]} | M = {history_M[index]} | E = {history_E[index]} | cost = {bestCost}")
            accuracy = predict(bestModel, x_test, y_test)
            f2.write(f"n = {history_n[index]} | M = {history_M[index]} | E = {history_E[index]} | cost = {round(c, 4)}\nAccuracy of best model: {accuracy}")
    

if __name__ == "__main__":
    main()