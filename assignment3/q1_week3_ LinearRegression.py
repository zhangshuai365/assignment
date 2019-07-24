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


def creat_sample_data(size):
    '''
    :param size: number of sample data
    :return: tuple of sample data
    '''
    train_X = np.linspace(-1, 1, size)
    train_Y = 5 * train_X + 6 + np.random.randn(size) * 0.2  # y=2x+6,and add some noise
    return train_X, train_Y


def linear_model(X, Theta):
    '''
    :param X: x of sample data
    :param Theta: (theta_0 and theta_1) for model y = theta_0 + theta_1 * x
    :return: y of model
    '''
    matrix_X = np.vstack((np.ones_like(X), X))
    predit_Y = Theta @ matrix_X
    return predit_Y


def cal_cost(Y, predit_Y, ):
    '''
    :param Y: y of sample data
    :param predit_Y: y of model
    :return: loss of model
    '''
    diff = Y - predit_Y
    avg_loss = 0.5 * sum(diff * diff) / len(Y)
    return avg_loss


def gradient_descent(X, Y, predict_Y, Theta, learning_rate):
    '''

    :param X: x of sample data
    :param Y: y of sample data
    :param predict_Y: y of model
    :param Theta: (theta_0 and theta_1) for current model
    :param learning_rate: learning rate of training
    :return: (theta_0 and theta_1) for next model
    '''
    num = len(Y)
    theta_0, theta_1 = Theta

    descent_theta_0 = sum(predict_Y - Y) / num
    descent_theta_1 = sum((predict_Y - Y) * X) / num

    theta_0 = theta_0 - learning_rate * descent_theta_0
    theta_1 = theta_1 - learning_rate * descent_theta_1

    Theta = np.array((theta_0, theta_1))
    return Theta


def train(X, Y, training_epoch, learning_rate):
    '''
    :param X: x of sample data
    :param Y: y of sample data
    :param training_epoch: training epoch
    :param learning_rate: learning rate of training
    :return: (theta_0 and theta_1) and loss for final model
    '''
    theta_0, theta_1 = 0, 0
    Theta = np.array([theta_0, theta_1], dtype=X.dtype)
    loss = []
    for epoch in range(training_epoch):
        predict_Y = linear_model(X, Theta)
        Theta = gradient_descent(X, Y, predict_Y, Theta, learning_rate)
        cost = cal_cost(Y, predict_Y, )
        loss.append(cost)
        Theta = gradient_descent(X, Y, predict_Y, Theta, learning_rate)
        print("epoch: {}, cost: {}, Theta: {}".format(epoch, cost, Theta))
    return Theta, loss


def run():
    training_epoch = 20
    learning_rate = 0.5
    train_X, train_Y = creat_sample_data(20)
    Theta, loss = train(train_X, train_Y, training_epoch, learning_rate)
    draw(train_X, train_Y, Theta, loss)


def draw(X, Y, Theta, loss):
    fig, [plt1, plt2] = plt.subplots(1, 2)
    fig.suptitle("q1_linear_regression")

    plt1.set_xlabel("x")
    plt1.set_ylabel("y")
    plt1.plot(X, Y, 'bo', label="samples")

    x = np.linspace(-1, 1, 20)
    y = Theta[0] + x * Theta[1]
    plt1.plot(x, y, 'r', label="hypothesis")
    plt1.legend()

    plt2.plot(list(range(len(loss))), loss, label='loss')
    plt2.set_xlabel("training_epoch")
    plt2.set_ylabel("loss")
    plt.show()


if __name__ == '__main__':
    run()
