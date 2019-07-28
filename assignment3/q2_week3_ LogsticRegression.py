#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
demo for Linear Regression:
step1:creat sample data
step2:training
step2.1:set up a hypothesis
step2.2:calculate the predict result according to the hypothesis
step2.3:calculate the loss (for showing)
step2.4:calculate the gradient and  descent it, generating a new model(hypothesis)
step2.5:go to step2.2, till the epoch arrived the target
step3:show the result
'''

import random
import matplotlib.pyplot as plt
import numpy as np


def gen_sample_data(size):
    '''
    :param size: number of sample data
    :return: tuple((x1,x2),y) of sample data
    '''
    train_X_1 = np.array([random.random() * 10 for _ in range(size)])
    train_X_2 = np.array([random.random() * 10 for _ in range(size)])

    def random_y(x1, x2):
        boundry = x1 * 2 - 6  # y=2x-6
        return 1 if x2 > boundry else 0

    y = [random_y(train_X_1[i], train_X_2[i]) for i in range(size)]
    train_Y = np.array(y, dtype=np.float)
    return (train_X_1, train_X_2), train_Y


def sigmoid_model(X, Theta):
    '''
    :param X: (x1,x2) of sample data
    :param Theta: (theta_0, theta_1, theta_2) for model z = theta_0 * x1 + theta_1 * x2 + theta_3, and
    z for model 1 / (1 + np.exp(-z))
    :return: y of model
    '''
    matrix_X = np.vstack((X, (np.ones_like(X[0]))))
    z = Theta @ matrix_X
    predit_Y = 1 / (1 + np.exp(-z))
    return predit_Y


def cal_cost(Y, predit_Y):
    '''
    :param Y: y of sample data
    :param predit_Y: y of model
    :return: loss of model
    '''
    avg_loss = -1 / len(Y) * (sum(Y * np.log(predit_Y)) + sum((1 - Y) * np.log(1 - predit_Y)))
    return avg_loss


def gradient_descent(X, Y, predict_Y, Theta, learning_rate):
    '''
    :param X: (x1,x2) of sample data
    :param Y: y of sample data
    :param predict_Y: y of model
    :param Theta: (theta_0,theta_1, theta_2) for current model
    :param learning_rate: learning rate of training
    :return: (theta_0, theta_1, theta_2) for next model
    '''
    num = len(Y)
    theta_0, theta_1, theta_2 = Theta

    descent_theta_0 = sum((predict_Y - Y) * X[0]) / num
    descent_theta_1 = sum((predict_Y - Y) * X[1]) / num
    descent_theta_2 = sum(predict_Y - Y) / num

    theta_0 = theta_0 - learning_rate * descent_theta_0
    theta_1 = theta_1 - learning_rate * descent_theta_1
    theta_2 = theta_2 - learning_rate * descent_theta_2

    Theta = np.array((theta_0, theta_1, theta_2))
    return Theta


def train(X, Y, training_epoch, learning_rate):
    '''
    :param X: x1 and x2 of sample data
    :param Y: y of sample data
    :param training_epoch: training epoch
    :param learning_rate: learning rate of training
    :return: (theta_0, theta_1, theta2) and loss for final model
    '''

    Theta = np.array(np.zeros(3), dtype=np.float)
    loss = []
    for epoch in range(training_epoch):
        predict_Y = sigmoid_model(X, Theta)
        Theta = gradient_descent(X, Y, predict_Y, Theta, learning_rate)
        cost = cal_cost(Y, predict_Y, )
        loss.append(cost)
        Theta = gradient_descent(X, Y, predict_Y, Theta, learning_rate)
        print("epoch: {}, cost: {}, Theta: {}".format(epoch, cost, Theta))
    return Theta, loss


def run():
    training_epoch = 200
    learning_rate = 0.01
    sample_data_num = 500
    train_X, train_Y = gen_sample_data(sample_data_num)
    Theta, loss = train(train_X, train_Y, training_epoch, learning_rate)
    draw(train_X, train_Y, Theta, loss, training_epoch, learning_rate, sample_data_num)


def draw(X, Y, Theta, loss, epoch, lr, sample_size):
    fig, [plt1, plt2] = plt.subplots(1, 2)
    fig.suptitle(("q2_logistic_regression\n" + "epoch:" + str(epoch) + "  " + "learning rate:" + str(lr)))

    plt1.set_xlabel("x1")
    plt1.set_ylabel("x2")
    pos_samples = (Y == 1)
    neg_samples = pos_samples == False
    pos = X[0][pos_samples], X[1][pos_samples]
    neg = X[0][neg_samples], X[1][neg_samples]

    plt1.scatter(pos[0], pos[1], label="positive", color="y")
    plt1.scatter(neg[0], neg[1], label="negative", color="red")

    theta_0, theta_1, theta_2 = Theta

    xs1 = np.linspace(0, 10, 10)
    xs2 = []

    if theta_1 != 0:
        xs2 = [-(theta_0 * x + theta_2) / theta_1 for x in xs1]
    plt1.plot(xs1, xs2, label="h_theta", color='blue')

    plt2.plot(list(range(len(loss))), loss, label='loss')
    plt2.set_xlabel("training_epoch")
    plt2.set_ylabel("loss")
    # plt.savefig("q2_logstic_regression_" + str(epoch) + "_" + str(lr) + ".png")
    plt.show()


if __name__ == '__main__':
    run()
