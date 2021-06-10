import random
import math
import pandas as pd
import numpy as np
import xgboost
from geneticalgorithm import geneticalgorithm as ga
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#ensemble  methods
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from simanneal import Annealer
from math import exp
from math import log
import numpy as np
import tensorflow as tf
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from geneticalgorithm import geneticalgorithm as ga
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
#ensemble  methods
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Normalizer
import xgboost as xgb
from xgboost import plot_importance
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
# function for stacking and voting
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier



''' 

class minimize():
    Simple Simulated Annealing
    

    def __init__(self, func, x0, opt_mode, cooling_schedule='linear', step_max=1000, t_min=0, t_max=100, bounds=[], alpha=None, damping=1):

        # checks
        assert opt_mode in ['combinatorial','continuous'], 'opt_mode must be either "combinatorial" or "continuous"'
        assert cooling_schedule in ['linear','exponential','logarithmic', 'quadratic'], 'cooling_schedule must be either "linear", "exponential", "logarithmic", or "quadratic"'


        # initialize starting conditions
        self.t = t_max
        self.t_max = t_max
        self.t_min = t_min
        self.step_max = step_max
        self.opt_mode = opt_mode
        self.hist = []
        self.cooling_schedule = cooling_schedule

        self.cost_func = func
        self.x0 = x0
        self.bounds = bounds[:]
        self.damping = damping
        self.current_state = self.x0
        self.current_energy = func(self.x0)
        self.best_state = self.current_state
        self.best_energy = self.current_energy


        # initialize optimization scheme
        if self.opt_mode == 'combinatorial': self.get_neighbor = self.move_combinatorial
        if self.opt_mode == 'continuous': self.get_neighbor = self.move_continuous


        # initialize cooling schedule
        if self.cooling_schedule == 'linear':
            if alpha != None:
                self.update_t = self.cooling_linear_m
                self.cooling_schedule = 'linear multiplicative cooling'
                self.alpha = alpha

            if alpha == None:
                self.update_t = self.cooling_linear_a
                self.cooling_schedule = 'linear additive cooling'

        if self.cooling_schedule == 'quadratic':
            if alpha != None:
                self.update_t = self.cooling_quadratic_m
                self.cooling_schedule = 'quadratic multiplicative cooling'
                self.alpha = alpha

            if alpha == None:
                self.update_t = self.cooling_quadratic_a
                self.cooling_schedule = 'quadratic additive cooling'

        if self.cooling_schedule == 'exponential':
            if alpha == None: self.alpha =  0.8
            else: self.alpha = alpha
            self.update_t = self.cooling_exponential

        if self.cooling_schedule == 'logarithmic':
            if alpha == None: self.alpha =  0.8
            else: self.alpha = alpha
            self.update_t = self.cooling_logarithmic


        # begin optimizing
        self.step, self.accept = 1, 0
        while self.step < self.step_max and self.t >= self.t_min and self.t>0:

            # get neighbor
            proposed_neighbor = self.get_neighbor()

            # check energy level of neighbor
            E_n = self.cost_func(proposed_neighbor)
            dE = E_n - self.current_energy

            # determine if we should accept the current neighbor
            if random() < self.safe_exp(-dE / self.t):
                self.current_energy = E_n
                self.current_state = proposed_neighbor[:]
                self.accept += 1

            # check if the current neighbor is best solution so far
            if E_n < self.best_energy:
                self.best_energy = E_n
                self.best_state = proposed_neighbor[:]

            # persist some info for later
            self.hist.append([
                self.step,
                self.t,
                self.current_energy,
                self.best_energy])

            # update some stuff
            self.t = self.update_t(self.step)
            self.step += 1

        # generate some final stats
        self.acceptance_rate = self.accept / self.step


    def move_continuous(self):
        # preturb current state by a random amount
        neighbor = [item + ((random() - 0.5) * self.damping) for item in self.current_state]

        # clip to upper and lower bounds
        if self.bounds:
            for i in range(len(neighbor)):
                x_min, x_max = self.bounds[i]
                neighbor[i] = min(max(neighbor[i], x_min), x_max)

        return neighbor


    def move_combinatorial(self):
        
        Swaps two random nodes along path
        Not the most efficient, but it does the job...
        
        p0 = randint(0, len(self.current_state)-1)
        p1 = randint(0, len(self.current_state)-1)

        neighbor = self.current_state[:]
        neighbor[p0], neighbor[p1] = neighbor[p1], neighbor[p0]

        return neighbor


    def results(self):
        print('+------------------------ RESULTS -------------------------+\n')
        print(f'      opt.mode: {self.opt_mode}')
        print(f'cooling sched.: {self.cooling_schedule}')
        if self.damping != 1: print(f'       damping: {self.damping}\n')
        else: print('\n')

        print(f'  initial temp: {self.t_max}')
        print(f'    final temp: {self.t:0.6f}')
        print(f'     max steps: {self.step_max}')
        print(f'    final step: {self.step}\n')

        print(f'  final energy: {self.best_energy:0.6f}\n')
        print('+-------------------------- END ---------------------------+')

    # linear multiplicative cooling
    def cooling_linear_m(self, step):
        return self.t_max /  (1 + self.alpha * step)

    # linear additive cooling
    def cooling_linear_a(self, step):
        return self.t_min + (self.t_max - self.t_min) * ((self.step_max - step)/self.step_max)

    # quadratic multiplicative cooling
    def cooling_quadratic_m(self, step):
        return self.t_min / (1 + self.alpha * step**2)

    # quadratic additive cooling
    def cooling_quadratic_a(self, step):
        return self.t_min + (self.t_max - self.t_min) * ((self.step_max - step)/self.step_max)**2

    # exponential multiplicative cooling
    def cooling_exponential_m(self, step):
        return self.t_max * self.alpha**step

    # logarithmical multiplicative cooling
    def cooling_logarithmic_m(self, step):
        return self.t_max / (self.alpha * log(step + 1))


    def safe_exp(self, x):
        try: return exp(x)
        except: return 0
'''

# Import the math function for calculations
import math
# Tensorflow library. Used to implement machine learning models
import tensorflow as tf
# Numpy contains helpful functions for efficient mathematical calculations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# Image library for image manipulation
# import Image
# Utils file
# Getting the MNIST data provided by Tensorflow


# Class that defines the behavior of the RBM
class RBM(object):
        def __init__(self, input_size, output_size):
                # Defining the hyperparameters
                self._input_size = input_size  # Size of input
                self._output_size = output_size  # Size of output
                self.epochs = 5  # Amount of training iterations
                self.learning_rate = 1.0  # The step used in gradient descent
                self.batchsize = 100  # The size of how much data will be used for training per sub iteration

                # Initializing weights and biases as matrices full of zeroes
                self.w = np.zeros([input_size, output_size], np.float32)  # Creates and initializes the weights with 0
                self.hb = np.zeros([output_size], np.float32)  # Creates and initializes the hidden biases with 0
                self.vb = np.zeros([input_size], np.float32)  # Creates and initializes the visible biases with 0

        # Fits the result from the weighted visible layer plus the bias into a sigmoid curve
        def prob_h_given_v(self, visible, w, hb):
                # Sigmoid
                return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

        # Fits the result from the weighted hidden layer plus the bias into a sigmoid curve
        def prob_v_given_h(self, hidden, w, vb):
                return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

        # Generate the sample probability
        def sample_prob(self, probs):
                return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

        # Training method for the model
        def train(self, X):
                # Create the placeholders for our parameters
                _w = tf.placeholder("float", [self._input_size, self._output_size])
                _hb = tf.placeholder("float", [self._output_size])
                _vb = tf.placeholder("float", [self._input_size])

                prv_w = np.zeros([self._input_size, self._output_size],
                                 np.float32)  # Creates and initializes the weights with 0
                prv_hb = np.zeros([self._output_size], np.float32)  # Creates and initializes the hidden biases with 0
                prv_vb = np.zeros([self._input_size], np.float32)  # Creates and initializes the visible biases with 0

                cur_w = np.zeros([self._input_size, self._output_size], np.float32)
                cur_hb = np.zeros([self._output_size], np.float32)
                cur_vb = np.zeros([self._input_size], np.float32)
                v0 = tf.placeholder("float", [None, self._input_size])

                # Initialize with sample probabilities
                h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
                v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
                h1 = self.prob_h_given_v(v1, _w, _hb)

                # Create the Gradients
                positive_grad = tf.matmul(tf.transpose(v0), h0)
                negative_grad = tf.matmul(tf.transpose(v1), h1)

                # Update learning rates for the layers
                update_w = _w + self.learning_rate * (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
                update_vb = _vb + self.learning_rate * tf.reduce_mean(v0 - v1, 0)
                update_hb = _hb + self.learning_rate * tf.reduce_mean(h0 - h1, 0)

                # Find the error rate
                err = tf.reduce_mean(tf.square(v0 - v1))

                # Training loop
                with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        # For each epoch
                        for epoch in range(self.epochs):
                                # For each step/batch
                                for start, end in zip(range(0, len(X), self.batchsize),
                                                      range(self.batchsize, len(X), self.batchsize)):
                                        batch = X[start:end]
                                        # Update the rates
                                        cur_w = sess.run(update_w,
                                                         feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                                        cur_hb = sess.run(update_hb,
                                                          feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                                        cur_vb = sess.run(update_vb,
                                                          feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                                        prv_w = cur_w
                                        prv_hb = cur_hb
                                        prv_vb = cur_vb
                                error = sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})
                                print('Epoch: %d' % epoch, 'reconstruction error: %f' % error)
                        self.w = prv_w
                        self.hb = prv_hb
                        self.vb = prv_vb

        # Create expected output for our DBN
        def rbm_outpt(self, X):
                input_X = tf.constant(X)
                _w = tf.constant(self.w)
                _hb = tf.constant(self.hb)
                out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
                with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        return sess.run(out)


class NN(object):

        def __init__(self, sizes, X, Y):
                # Initialize hyperparameters
                self._sizes = sizes
                self._X = X
                self._Y = Y
                self.w_list = []
                self.b_list = []
                self._learning_rate = 0.1
                self._momentum = 0.0
                self._epoches = 20
                self._batchsize = 100
                input_size = X.shape[1]

                # initialization loop
                for size in self._sizes + [Y.shape[1]]:
                        # Define upper limit for the uniform distribution range
                        max_range = 4 * math.sqrt(6. / (input_size + size))

                        # Initialize weights through a random uniform distribution
                        self.w_list.append(
                                np.random.uniform(-max_range, max_range, [input_size, size]).astype(np.float32))

                        # Initialize bias as zeroes
                        self.b_list.append(np.zeros([size], np.float32))
                        input_size = size

        # load data from rbm
        def load_from_rbms(self, dbn_sizes, rbm_list):
                # Check if expected sizes are correct
                assert len(dbn_sizes) == len(self._sizes)

                for i in range(len(self._sizes)):
                        # Check if for each RBN the expected sizes are correct
                        assert dbn_sizes[i] == self._sizes[i]

                # If everything is correct, bring over the weights and biases
                for i in range(len(self._sizes)):
                        self.w_list[i] = rbm_list[i].w
                        self.b_list[i] = rbm_list[i].hb

        # Training method
        def train(self):
                # Create placeholders for input, weights, biases, output
                _a = [None] * (len(self._sizes) + 2)
                _w = [None] * (len(self._sizes) + 1)
                _b = [None] * (len(self._sizes) + 1)
                _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
                y = tf.placeholder("float", [None, self._Y.shape[1]])

                # Define variables and activation functoin
                for i in range(len(self._sizes) + 1):
                        _w[i] = tf.Variable(self.w_list[i])
                        _b[i] = tf.Variable(self.b_list[i])
                for i in range(1, len(self._sizes) + 2):
                        _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])

                # Define the cost function
                cost = tf.reduce_mean(tf.square(_a[-1] - y))

                # Define the training operation (Momentum Optimizer minimizing the Cost function)
                train_op = tf.train.MomentumOptimizer(
                        self._learning_rate, self._momentum).minimize(cost)

                # Prediction operation
                predict_op = tf.argmax(_a[-1], 1)

                # Training Loop
                with tf.Session() as sess:
                        # Initialize Variables
                        sess.run(tf.global_variables_initializer())

                        # For each epoch
                        for i in range(self._epoches):

                                # For each step
                                for start, end in zip(
                                        range(0, len(self._X), self._batchsize),
                                        range(self._batchsize, len(self._X), self._batchsize)):
                                        # Run the training operation on the input data
                                        sess.run(train_op, feed_dict={
                                                _a[0]: self._X[start:end], y: self._Y[start:end]})

                                for j in range(len(self._sizes) + 1):
                                        # Retrieve weights and biases
                                        self.w_list[j] = sess.run(_w[j])
                                        self.b_list[j] = sess.run(_b[j])

                                print("Accuracy rating for epoch " + str(i) + ": " + str(
                                        np.mean(np.argmax(self._Y, axis=1) == \
                                                sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y}))))


if __name__ == '__main__':

        df = pd.read_excel('data_for_DNBmodel_nounknown.xlsx')  # this time use absolute address to input data

        X_new = df[['smoking_situation', 'health_investigation', 'health_awareness', 'stress',
                    'obesity', 'hyperlipidaemia', 'profession', 'work_day', 'week_work_hours',
                    'hypertension', 'worktime', 'public_pension', 'insurance', 'gender',
                    'age', 'room_num', 'room_space', 'family_num', 'spense']]
        # y = df['diabetes'].values
        # X_new=X_new[0:20000]
        # y = df['diabetes'][0:20000].values
        y = df['diabetes'].values

        X_new = X_new.to_numpy()
        X_new = np.float32(X_new)
        print(X_new.shape)
        print(y.shape)

        '''
    
         X_new = df[['smoking_situation', 'health_investigation', 'health_awareness', 'stress',
                    'obesity', 'hyperlipidaemia', 'profession', 'work_day', 'week_work_hours',
                    'hypertension', 'worktime', 'public_pension', 'insurance', 'gender',
                    'age', 'room_num', 'room_space', 'family_num', 'spense']]
        # X_new=np.transpose(X_new)
        X_new = np.float32(X_new)
        y = df['diabetes'].values
        '''
        # trX=X_new
        # trY=y

        # #split data to training data and test data
        trX, trY, teX, teY = train_test_split(X_new, y, test_size=0.50,
                                              random_state=100)

        RBM_hidden_sizes = [15, 10, 5]  # create 4 layers of RBM with size 785-500-200-50
        # Since we are training, set input as training data
        inpX = trX
        # Create list to hold our RBMs
        rbm_list = []
        # Size of inputs is the number of inputs in the training set
        input_size = inpX.shape[1]
        print(input_size)

        # For each RBM we want to generate
        for i, size in enumerate(RBM_hidden_sizes):
                print('RBM: ', i, ' ', input_size, '->', size)
                rbm_list.append(RBM(input_size, size))
                input_size = size

        # For each RBM in our list
        for rbm in rbm_list:
                print('New RBM:')
                # Train a new one
                rbm.train(inpX)
                # Return the output layer
                inpX = rbm.rbm_outpt(inpX)

        nNet = NN(RBM_hidden_sizes, trX, trY)
        nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
        nNet.train()