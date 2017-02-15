from random import randint
from math import exp
import numpy
import sys
if (len(sys.argv) > 1):
    version = sys.argv[1].lower()
else:
    version = 'batch'

print("Version is ", version)


def parse_data():
    file = open('total_formated.txt', 'r')
    examples = []
    for line in file:
        line_split = line.split(' ')
        y = int(line_split[0])
        chars = int(line_split[1].split(':')[1])
        As = int(line_split[2].split(':')[1])
        tot = [y, chars, As]
        examples.append(tot)
    return examples

def normalization(examples):
    #Finding max val
    max_val = 0
    for i in range(len(examples)):
        if examples[i][1] > max_val:
            max_val = examples[i][1]

    for i in range(len(examples)):
        examples[i][1] = examples[i][1]/max_val
        examples[i][2] = examples[i][2]/max_val

    return examples

def loss(w, examples):
    w_0, w_1, w_2 = w
    w_0_grad = 0
    w_1_grad = 0
    w_2_grad = 0
    for each_example in examples:
        y = each_example[0]
        exponent = - (w_0 + w_1*each_example[1] + w_2*each_example[2])
        log_y_hat = 1/(1 + exp(exponent))
        parantheses = y - log_y_hat
        w_0_grad += 1*parantheses
        w_1_grad += each_example[1]*parantheses
        w_2_grad += each_example[2]*parantheses
    return [w_0_grad, w_1_grad, w_2_grad]

examples = parse_data()

examples = normalization(examples)



w_0 = 1
w_1 = 1
w_2 = 1
alpha = 1
number_of_missclassified = 100000000
eps = 50
iterations = 0
w = [w_0, w_1, w_2]


while numpy.linalg.norm(loss(w,examples)) > eps:
    l = []
    y_diff = []

    # For each row
    for each_example in examples:
        y = each_example[0]
        y_hat = 0

        exponent = - (w_0 + w_1*each_example[1] + w_2*each_example[2])
        log_y_hat = 1/(1 + exp(exponent))
        y_diff.append(y - log_y_hat)





    w_0_sum = 0
    w_1_sum = 0
    w_2_sum = 0

    if (version == 'stochastic'):
        rand_i = randint(0,29)
        w_0 = w_0 + alpha*y_diff[rand_i]
        w_1 = w_1 + alpha*examples[rand_i][1]*y_diff[rand_i]
        w_2 = w_2 + alpha*examples[rand_i][2]*y_diff[rand_i]
    else:

        for i in range(len(examples)):
            w_0_sum += y_diff[i]
            w_1_sum += examples[i][1]*y_diff[i]
            w_2_sum += examples[i][2]*y_diff[i]

        w_0 = w_0 + alpha*w_0_sum
        w_1 = w_1 + alpha*w_1_sum
        w_2 = w_2 + alpha*w_2_sum

    iterations +=1
    alpha = 1000/(1000 + iterations)

print('Iterations: ', iterations)
print('The perceptron has set w0=%f w1=%f w2=%f with %f missclassifications' % (w_0, w_1, w_2, number_of_missclassified))

