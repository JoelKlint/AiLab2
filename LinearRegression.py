import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

def regressionBatch(nr_letters, nr_a, language):
    eps = 0.1
    alpha = 0.95
    w_0 = 1
    w_1 = 1
    w = np.matrix([w_0, w_1]).T
    nr_iterations = 0
    #Av någon anledning så stämmer inte funktionen på slides. Alla andra gör på följande sätt
    error = nr_letters*w  - nr_a
    while np.linalg.norm(error) > eps and nr_iterations < 1000:
        error = nr_letters*w -  nr_a
        gradient = (nr_letters.T*error)/nr_datapoints
        gradient_step = alpha*gradient
        w = w - gradient_step
        nr_iterations += 1


    print ('We are done with ', language ,' batch gradient descent after ', nr_iterations, ' iterations w0 = %.4f ; w1 = %.4f' %(w[0], w[1]))
    return w

def regressionStochastic(nr_letters, nr_a, language):
    eps = 0.01
    alpha = 0.95
    w_0 = 1
    w_1 = 1
    w = np.matrix([w_0, w_1]).T
    nr_iterations = 0
    error = nr_letters*w  - nr_a
    while (np.linalg.norm(np.square(error))) > eps and nr_iterations < 500:
        #Numpy arrays can use arrays as indexes. Shuffling our data
        randomized_indexes = np.arange(len(nr_letters))
        np.random.shuffle(randomized_indexes)
        nr_letters = nr_letters[randomized_indexes]
        nr_a = nr_a[randomized_indexes]

        for i in range(nr_datapoints):
            individual_error = nr_letters[i]*w -  nr_a[i]
            tuple = nr_letters[i]
            gradient = (tuple.T*individual_error)/nr_datapoints
            gradient_step = alpha*gradient
            w = w - gradient_step

        error = nr_letters*w  - nr_a
        nr_iterations += 1
    print ('We are done with ', language, ' stochastic gradient descent after ', nr_iterations, ' iterations w0 = %.4f ; w1 = %.4f' %(w[0], w[1]))
    return  w


def normalization(values):
    max_val = max(values)
    for i in range(len(values)):
        values[i] = values[i]/max_val
    return values

def readFile(file):
    nr_letters = []
    nr_a = []
    for line in file:
        nrLetters, nrAs = line.split('\t')
        nr_letters.append(int(nrLetters.strip()))
        nr_a.append(int(nrAs.strip()))
    return [nr_letters, nr_a]


#Load Data
english = open('english.txt', 'r')
nr_letters_english, nr_a_english = readFile(english)
english.close()

french = open('french.txt', 'r')
nr_letters_french, nr_a_french = readFile(french)
french.close()
nr_datapoints = len(nr_letters_english)

#Formatting data
nr_letters_english = normalization(nr_letters_english)
nr_a_english = normalization(nr_a_english)
nr_letters_french = normalization(nr_letters_french)
nr_a_french = normalization(nr_a_french)
nr_letters_english = np.matrix(nr_letters_english).T
nr_letters_french = np.matrix(nr_letters_french).T
#Adding padding with ones
nr_letters_enlgish_padded = np.hstack([np.matrix(np.ones(len(nr_letters_english))).T, nr_letters_english])
nr_letters_french_padded = np.hstack([np.matrix(np.ones(len(nr_letters_french))).T, nr_letters_french])
nr_a_english = np.matrix(nr_a_english).T
nr_a_french = np.matrix(nr_a_french).T

#Do Gradient descent
res_batch_english = regressionBatch(nr_letters_enlgish_padded, nr_a_english, 'English')
res_stoch_english = regressionStochastic(nr_letters_enlgish_padded, nr_a_english, 'English')
res_batch_french = regressionBatch(nr_letters_french_padded, nr_a_french, 'French')
res_stoch_french = regressionStochastic(nr_letters_french_padded, nr_a_french, 'French')


x = np.linspace(0, 1, 15)

#Prepearing data for visualization
regression_line_batch_english = res_batch_english[0] + res_batch_english[1] * x
regression_line_batch_english = regression_line_batch_english.T
regression_line_stoch_english = res_stoch_english[0] + res_stoch_english[1] * x
regression_line_stoch_english = regression_line_stoch_english.T

regression_line_batch_french = res_batch_french[0] + res_batch_french[1] * x
regression_line_batch_french = regression_line_batch_french.T
regression_line_stoch_french = res_stoch_french[0] + res_stoch_french[1] * x
regression_line_stoch_french = regression_line_stoch_french.T


#Plotting data
engData = plt.scatter(nr_letters_english, nr_a_english, c='r', label='English')
engBatch, = plt.plot(x, regression_line_batch_english, label='Batch English')
engStoch, = plt.plot(x, regression_line_stoch_english, label='Stochastic English')
frData = plt.scatter(nr_letters_french, nr_a_french, c='y', label='French')
frBatch, = plt.plot(x, regression_line_batch_french, label='Batch French')
frStoch, = plt.plot(x, regression_line_stoch_french, label='Stochastic French')

plt.legend([engBatch, engStoch, engData, frBatch, frStoch, frData],
           ['Batch English', 'Stochastic English', 'English', 'Batch French', 'Stochastic English', 'French'], loc='upper left')



plt.show()