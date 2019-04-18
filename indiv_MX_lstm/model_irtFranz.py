'''
Social LSTM model implementation using Tensorflow
Social LSTM Paper: http://vision.stanford.edu/pdf/CVPR16_N_LSTM.pdf

Author : Anirudh Vemula
Date: 10th October 2016
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell
import ipdb
from numpy import linalg as la
import scipy.io


# The Vanilla LSTM model
class Model_irfr():

    def __init__(self, args, infer=False):
        '''
        Initialisation function for the class Model.
        Params:
        args: Contains arguments required for the Model creation
        '''

        # If sampling new trajectories, then infer mode
        if infer:
            # Infer one position at a time
            args.batch_size = 1
            args.seq_length = 1

        # Store the arguments
        self.args = args
        self.irtFlag = args.myCounter



        # Initialize a BasicLSTMCell recurrent unit
        # args.rnn_size contains the dimension of the hidden state of the LSTM
        cell = rnn_cell.BasicLSTMCell(args.rnn_size, state_is_tuple=False)

        # Multi-layer RNN construction, if more than one layer
        cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=False)

        # TODO: (improve) Dropout layer can be added here
        # Store the recurrent unit
        self.cell = cell

        # TODO: (resolve) Do we need to use a fixed seq_length?
        # Input data contains sequence of (x,y,x',y') points Stamp: editing irtiza chnaged last argument 2 to 4
        self.input_data = tf.placeholder(tf.float32, [None, args.seq_length, 4])
        # target data contains sequences of (x,y,x',y') points as well Stamp editing irtiza editing irtiza chnaged last argument 2 to 4
        self.target_data = tf.placeholder(tf.float32, [None, args.seq_length, 4])

        # Learning rate
        self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")

        # self.irtFlag = tf.Variable(args.myCounter,trainable=False, name="myCounter")

        # Initial cell state of the LSTM (initialised with zeros)
        self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        # Output size is the set of parameters (mu, sigma, corr)
        output_size = 14  # 2 mu, 2 sigma and 1 corr # Stamp editing irtiza editing irtiza chnaged output_size -> from 5 to 10->14

        # Embedding for the spatial coordinates
        with tf.variable_scope("coordinate_embedding"):
            #  The spatial embedding using a ReLU layer
            #  Embed the 2D coordinates into embedding_size dimensions
            #  TODO: (improve) For now assume embedding_size = rnn_size
            embedding_w = tf.get_variable("embedding_w", [4, args.embedding_size]) # Stamp editing irtiza editing irtiza chnaged first argument 2 to 4
            embedding_b = tf.get_variable("embedding_b", [args.embedding_size])

        # Output linear layer
        with tf.variable_scope("rnnlm"):
            output_w = tf.get_variable("output_w", [args.rnn_size, output_size], initializer=tf.truncated_normal_initializer(stddev=0.01), trainable=True)
            output_b = tf.get_variable("output_b", [output_size], initializer=tf.constant_initializer(0.01), trainable=True)

        # Split inputs according to sequences.
        inputs = tf.split(self.input_data, args.seq_length, 1)
        # Get a list of 2D tensors. Each of size numPoints x 4
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # Embed the input spatial points into the embedding space
        embedded_inputs = []
        for x in inputs:
            # Each x is a 2D tensor of size numPoints x 4
            # Embedding layer
            embedded_x = tf.nn.relu(tf.add(tf.matmul(x, embedding_w), embedding_b))
            embedded_inputs.append(embedded_x)

        # Feed the embedded input data, the initial state of the LSTM cell, the recurrent unit to the seq2seq decoder
        outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(embedded_inputs, self.initial_state, cell, loop_function=None, scope="rnnlm")

        # Concatenate the outputs from the RNN decoder and reshape it to ?xargs.rnn_size
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

        # Apply the output linear layer
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        # Store the final LSTM cell state after the input data has been feeded
        self.final_state = last_state

        # reshape target data so that it aligns with predictions
        flat_target_data = tf.reshape(self.target_data, [-1, 4]) # Stamp editing irtiza editing irtiza chnaged last argument 2 to 4
        # Extract the x-coordinates and y-coordinates from the target data
        [x_data, y_data, x_prime, y_prime] = tf.split(flat_target_data, 4, 1) # Stamp editing irtiza editing irtiza chnaged middle argument 2 to 4

        # TODO: (Irtiza)this function requiures changing potentailly twice number of argumenst stamep unchanged so far?

        def tf_Nd_normal(x, y, x_prime,y_prime,mux, muy,mux_prime, muy_prime, a11, a21, a22, a31, a32, a33, a41, a42, a43, a44):

            obs = tf.concat([x, y, x_prime,y_prime], 1)
            mu = tf.concat([mux, muy, mux_prime,muy_prime], 1)
            # cov = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]

            tst = a11.shape
            tst = tf.cast(tst, tf.int32)
            dumVar=tf.zeros(tst)
            firstRow = tf.concat([tf.exp(a11), dumVar, dumVar, dumVar], 1)
            secondRow = tf.concat([a21, tf.exp(a22), dumVar, dumVar], 1)
            thirdRow = tf.concat([a31, a32, tf.exp(a33), dumVar], 1)
            fourthRow = tf.concat([a41, a42, a43, tf.exp(a44)], 1)
            nst = a11.shape[0]
            nst = tf.cast(nst, tf.int32)
            aL = tf.concat([firstRow, secondRow, thirdRow, fourthRow], 1)
            L = tf.reshape(aL, [nst, 4, 4])
            Lt=tf.transpose(L,perm=[0,2,1])

            cov=tf.matmul(L,Lt)



            # chol=tf.cholesky(pdCovMat)
            mvn = tf.contrib.distributions.MultivariateNormalFullCovariance(mu,cov)
            # asd=self.irtFlag
            self.tval = tf.cast(self.irtFlag, tf.int32)

            temp = mvn.prob(obs)
            temp = tf.reshape(temp, [nst, 1])
            # temp=tf.div(temp,10)
            # temp = tf.exp(temp)
            self.cov=cov
            self.mu=mu
            # self.val=temp
            self.obs = obs
            self.pdf=temp
            # Calculate sx=*sy
            result = temp
            self.result = result

            return result
        # Important difference between loss func of Social LSTM and Graves (2013)
        # is that it is evaluated over all time steps in the latter whereas it is
        # done from t_obs+1 to t_pred in the former
        def get_lossfunc( z_mux, z_muy,z_mux_prime, z_muy_prime, a11, a21, a22, a31, a32, a33, a41, a42, a43, a44 , x_data, y_data, x_prime, y_prime):

            # step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))

            # Calculate the PDF of the data w.r.t to the distribution
            result0 = tf_Nd_normal(x_data, y_data,x_prime,y_prime,z_mux, z_muy,z_mux_prime, z_muy_prime, a11, a21, a22, a31, a32, a33, a41, a42, a43, a44 )
            # For numerical stability purposes



            epsilon = 1e-20
            myTemp=tf.maximum(result0, epsilon)
            result1 = -tf.log(myTemp)
            self.sval=tf.reduce_sum(result1)
            # self.mval = tf.reduce_mean(result1)

            return tf.reduce_sum(result1)

        def get_coef(output):   # function modified by irtiza accordingly
            # eq 20 -> 22 of Graves (2013)
            # TODO : (resolve) Does Social LSTM paper do this as well?
            # the paper says otherwise but this is essential as we cannot
            # have negative standard deviation and correlation needs to be between
            # -1 and 1

            z = output
            # Split the output into 5 parts corresponding to means, std devs and corr
            z_mux, z_muy, z_mux_prime, z_muy_prime, a11, a21, a22, a31, a32, a33, a41, a42, a43, a44  = tf.split(z, 14, 1)

            # add Priors
            # tf.get_variable('a11',initializer=0.1359)
            # tf.get_variable('a21', initializer=0.0024)
            # tf.get_variable('a22', initializer=0.0079)
            # tf.get_variable('a31', initializer=0.1357)
            # tf.get_variable('a32', initializer=0.0024)
            # tf.get_variable('a33', initializer=0.1358)
            # tf.get_variable('a41', initializer=0.0021)
            # tf.get_variable('a42', initializer=0.0077)
            # tf.get_variable('a43', initializer=0.0022)
            # tf.get_variable('a44', initializer=0.0078)


            #


            return [ z_mux, z_muy,z_mux_prime, z_muy_prime,a11,a21,a22,a31,a32,a33,a41,a42,a43,a44]


        # Extract the coef from the output of the linear layer # Stamp editing irtiza editing irtiza change the numb of outputs to 10
        [o_mux, o_muy, o_mux_prime, o_muy_prime, o_a11,o_a21,o_a22,o_a31,o_a32,o_a33,o_a41,o_a42,o_a43,o_a44] = get_coef(output)
        # Store the output from the model
        self.output = output


        # prior cov Mat


        #




        # Store the predicted outputs
        self.mux = o_mux
        self.muy = o_muy
        self.mux_prime = o_mux_prime
        self.muy_prime = o_muy_prime
        self.a11 = o_a11
        self.a21 = o_a21
        self.a22 = o_a22
        self.a31 = o_a31
        self.a32 = o_a32
        self.a33 = o_a33
        self.a41 = o_a41
        self.a42 = o_a42
        self.a43 = o_a43
        self.a44 = o_a44




        lossfunc = get_lossfunc(o_mux, o_muy,o_mux_prime, o_muy_prime, o_a11, o_a21, o_a22, o_a31, o_a32, o_a33, o_a41, o_a42, o_a43, o_a44 , x_data, y_data, x_prime, y_prime)

        # Compute the cost
        self.cost = tf.div(lossfunc, (args.batch_size * args.seq_length))
        self.mval=self.cost

        # Get trainable_variables
        tvars = tf.trainable_variables()

        # L2 loss
        l2 = args.lambda_param * sum(tf.nn.l2_loss(tvar) for tvar in tvars)
        self.cost = self.cost + l2

        # TODO: (resolve) We are clipping the gradients as is usually done in LSTM
        # implementations. Social LSTM paper doesn't mention about this at all
        # Calculate gradients of the cost w.r.t all the trainable variables
        self.gradients = tf.gradients(self.cost, tvars)
        # Clip the gradients if they are larger than the value given in args
        grads, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        # NOTE: Using RMSprop as suggested by Social LSTM instead of Adam as Graves(2013) does
        # optimizer = tf.train.AdamOptimizer(self.lr)
        # initialize the optimizer with teh given learning rate
        # optimizer = tf.train.RMSPropOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)

        # Train operator
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, traj, true_traj, num=10):
        '''
        Given an initial trajectory (as a list of tuples of points), predict the future trajectory
        until a few timesteps
        Params:
        sess: Current session of Tensorflow
        traj: List of past trajectory points
        true_traj : List of complete trajectory points
        num: Number of time-steps into the future to be predicted
        # '''
        def sample_gaussian_2d(mux, muy, muvx, muvy,  a11, a21, a22, a31, a32, a33, a41, a42, a43, a44):
            '''
            Function to sample a point from a given 2D normal distribution
            params:
            mux : mean of the distribution in x
            muy : mean of the distribution in y
            sx : std dev of the distribution in x
            sy : std dev of the distribution in y
            rho : Correlation factor of the distribution
            '''
            # Extract mean
            mean = np.array([mux, muy, muvx, muvy])
            # Extract covariance matrix
            # dumVar = np.zeros(1)
            L=np.array([[np.exp(a11), 0, 0, 0],
            [a21, np.exp(a22), 0, 0],
            [a31, a32, np.exp(a33), 0],
            [a41, a42, a43, np.exp(a44)]])

            Lt=np.transpose(L)
            cov=np.dot(L,Lt)

            #
            # cov = np.array([[sx * sx, rhoxy * sx * sy, rhoxvx * sx * svx, rhoxvy * sx * svy],
            #            [rhoxy * sx * sy, sy * sy, rhoyvx * sy * svx, rhoyvy * sy * svy],
            #            [rhoxvx * sx * svx, rhoyvx * sy * svx, svx * svx, rhovxvy * svx * svy],
            #            [rhoxvy * sx * svy, rhoyvy * sy * svy, rhovxvy * svx * svy, svy * svy]])
            # cov = newPSD(notacov)
            #cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
            # Sample a point from the multivariate normal distribution
            # TODO: (irtiza) why is he sampling a point randomly ?? not pickign the liklihood
            # pcov=nearestPD(cov)
            # ncov=la.cholesky(pcov)
            # try:p
            #     ncov = la.cholesky(cov)
            #
            # except la.LinAlgError:
            #     filesave = 'VisualizeUtils/debug/' + 'choleskyDecomp.mat'
            #     scipy.io.savemat(filesave, mdict={'cov': cov, 'pcov': pcov, 'mean': mean})
                # ncov = nearestPD(cov)

            x = np.random.multivariate_normal(mean, cov, 1)
            # return x[0][0], x[0][1],x[0][2],x[0][3],cov, mean
            return mux, muy, muvx, muvy, cov, mean

        def nearestPD(A):
            """Find the nearest positive-definite matrix to input

            A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
            credits [2].

            [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

            [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
            matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
            """

            B = (A + A.T) / 2
            _, s, V = la.svd(B)

            H = np.dot(V.T, np.dot(np.diag(s), V))

            A2 = (B + H) / 2

            A3 = (A2 + A2.T) / 2
            idx = np.diag_indices(A3.shape[0])
            factor=1e-6
            A3[idx] = A3[idx] + factor

            if isPD(A3):
                return A3

            spacing = np.spacing(la.norm(A))
            # spacing=[]
            # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
            # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
            # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
            # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
            # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
            # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
            # `spacing` will, for Gaussian random matrixes of small dimension, be on
            # othe order of 1e-16. In practice, both ways converge, as the unit test
            # below suggests.
            I = np.eye(A.shape[0])
            k = 1
            while not isPD(A3):
                # print('in while')
                mineig = np.min(np.real(la.eigvals(A3)))
                # mineig=0.01
                factor=(-mineig * k ** 2 + spacing)
                # factor=factor*1000
                A3 += I * factor

                # np.diag(A3)=2
                k += 1

            return A3

        def isPD(B):
            """Returns true when input is positive-definite, via Cholesky"""
            try:
                _ = la.cholesky(B)
                return True
            except la.LinAlgError:
                return False


        # Initial state with zeros
        state = sess.run(self.cell.zero_state(1, tf.float32))

        # Iterate over all the positions seen in the trajectory # editing by irtiza data size
        for pos in traj[:-1]:
            # Create the input data tensor
            data = np.zeros((1, 1, 4), dtype=np.float32)
            data[0, 0, 0] = pos[0]  # x
            data[0, 0, 1] = pos[1]  # y
            data[0, 0, 2] = pos[2]  # xprime
            data[0, 0, 3] = pos[3]  # yprime5
            # Create the feed dict
            feed = {self.input_data: data, self.initial_state: state}
            # Get the final state after processing the current position
            [state] = sess.run([self.final_state], feed)

        ret = traj

        # Last position in the observed trajectory
        last_pos = traj[-1]

        # Construct the input data tensor for the last point
        prev_data = np.zeros((1, 1, 4), dtype=np.float32)
        prev_data[0, 0, 0] = last_pos[0]  # x
        prev_data[0, 0, 1] = last_pos[1]  # y
        prev_data[0, 0, 2] = last_pos[2]  # xprime
        prev_data[0, 0, 3] = last_pos[3]  # yprime

        prev_target_data = np.reshape(true_traj[traj.shape[0]], (1, 1, 4))
        myVal = np.zeros((12, 10))
        mean_list=[]
        cov_list = []
        for t in range(num):
            # Create the feed dict
            feed = {self.input_data: prev_data, self.initial_state: state, self.target_data: prev_target_data}

            # Get the final state and also the coef of the distribution of the next point
            [o_mux, o_muy,o_mux_prime, o_muy_prime, o_a11, o_a21, o_a22, o_a31, o_a32, o_a33, o_a41, o_a42, o_a43, o_a44, state, cost] = sess.run([self.mux, self.muy,self.mux_prime,self.muy_prime, self.a11,self.a21,self.a22,self.a31,self.a32,self.a33,self.a41,self.a42,self.a43,self.a44,self.final_state, self.cost], feed)
            # print cost
            # Sample the next point from the distribution


            next_x, next_y,next_x_prime,next_y_prime,cov,meanval = sample_gaussian_2d(o_mux[0][0], o_muy[0][0],o_mux_prime[0][0], o_muy_prime[0][0], o_a11[0][0], o_a21[0][0], o_a22[0][0], o_a31[0][0], o_a32[0][0], o_a33[0][0], o_a41[0][0], o_a42[0][0], o_a43[0][0], o_a44[0][0] )
            # next_x_prime, next_y_prime = sample_gaussian_2d(o_mux_prime[0][0], o_muy_prime[0][0], o_sx_prime[0][0], o_sy_prime[0][0], o_corr_prime[0][0],)
            # Append the new point to the trajectory
            ret = np.vstack((ret, [next_x, next_y, next_x_prime, next_y_prime]))
            cov_list.append(cov)
            mean_list.append(meanval)
            # pcov_list.append(pcov)

            # Set the current sampled position as the last observed position
            prev_data[0, 0, 0] = next_x
            prev_data[0, 0, 1] = next_y
            prev_data[0, 0, 2] = next_x_prime
            prev_data[0, 0, 3] = next_y_prime

        # ipdb.set_trace()
        return ret,myVal, cov_list, mean_list
