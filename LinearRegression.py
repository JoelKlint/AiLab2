import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

def regressionBatch(nr_letters, nr_a):
    eps = 0.03
    alpha = 0.95
    w_0 = 1
    w_1 = 1
    w = np.matrix([w_0, w_1]).T
    nr_iterations = 0
    #Av någon anledning så stämmer inte funktionen på slides. Alla andra gör på följande sätt
    error = nr_letters*w  - nr_a
    print(error)
    print (np.linalg.norm(np.square(error)))
    while np.linalg.norm(error) > eps and nr_iterations < 500:
        error = nr_letters*w -  nr_a
        gradient = (nr_letters.T*error)/nr_datapoints
        gradient_step = alpha*gradient
        w = w - gradient_step
        print (w)
        nr_iterations += 1


    print ('We are done with batch gradient descent after ', nr_iterations, ' iterations', w[0], w[1])
    return w

def regressionStochastic(nr_letters, nr_a):
    eps = 0.001
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
        print (np.linalg.norm(np.square(error)))
        nr_iterations += 1
    print ('We are done with stochastic gradient descent after ', nr_iterations, ' iterations', w[0], w[1])
    return  w


def normalization(values):
    max_val = max(values)
    for i in range(len(values)):
        values[i] = values[i]/max_val
    return values

english = open('english.txt', 'r')

nr_letters = []
nr_a = []


for line in english:
    nrLetters, nrAs = line.split('\t')
    nr_letters.append(int(nrLetters.strip()))
    nr_a.append(int(nrAs.strip()))

english.close()

nr_datapoints = len(nr_letters)
nr_letters = normalization(nr_letters)
nr_a = normalization(nr_a)

nr_letters = np.matrix(nr_letters).T
#Adding padding with ones
nr_letters_padded = np.hstack([np.matrix(np.ones(len(nr_letters))).T, nr_letters])
nr_a = np.matrix(nr_a).T

resBatch = regressionBatch(nr_letters_padded, nr_a)
resStoch = resStoch = regressionStochastic(nr_letters_padded, nr_a)

x = np.linspace(0, 1, 15)


regression_line_batch = resBatch[0] + resBatch[1]*x
regression_line_batch = regression_line_batch.T

regression_line_stoch = resStoch[0] + resStoch[1]*x
regression_line_stoch = regression_line_stoch.T


plt.plot(nr_letters, nr_a, 'ro')
Batch, = plt.plot(x, regression_line_batch, label='Batch')
Stoch, = plt.plot(x, regression_line_stoch, label='Stochastic')
plt.legend([Batch, Stoch], ['Batch', 'Stochastic'])



plt.show()