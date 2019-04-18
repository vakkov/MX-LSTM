'''
Social LSTM model implementation using Tensorflow
Social LSTM Paper: http://vision.stanford.edu/pdf/CVPR16_N_LSTM.pdf

Author : Anirudh Vemula
Date: 17th October 2016
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell
from grid import getSequenceGridMask
import pdb
from numpy import linalg as la
import scipy.io


class SocialModel():

    def __init__(self, args, infer=False):
        '''
        Initialisation function for the class SocialModel
        params:
        args : Contains arguments required for the model creation
        '''

        # If sampling new trajectories, then infer mode
        if infer:
            # Sample one position at a time
            args.batch_size = 1
            args.seq_length = 1

        # Store the arguments
        self.args = args
        self.infer = infer

        # Store rnn size and grid_size
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size

        # Maximum number of peds
        self.maxNumPeds = args.maxNumPeds

        # NOTE : For now assuming, batch_size is always 1. That is the input
        # to the model is always a sequence of frames

        # Construct the basicLSTMCell recurrent unit with a dimension given by args.rnn_size
        with tf.name_scope("LSTM_cell"):
            cell = rnn_cell.BasicLSTMCell(args.rnn_size, state_is_tuple=False)
            # if not infer and args.keep_prob < 1:
            # cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=args.keep_prob)

        # placeholders for the input data and the target data
        # A sequence contains an ordered set of consecutive frames
        # Each frame can contain a maximum of 'args.maxNumPeds' number of peds
        # For each ped we have their (pedID, x, y) positions as input
        self.input_data = tf.placeholder(tf.float32, [args.seq_length, args.maxNumPeds, 5], name="input_data")

        # target data would be the same format as input_data except with
        # one time-step ahead
        self.target_data = tf.placeholder(tf.float32, [args.seq_length, args.maxNumPeds, 5], name="target_data")

        # Grid data would be a binary matrix which encodes whether a pedestrian is present in
        # a grid cell of other pedestrian
        self.grid_data = tf.placeholder(tf.float32, [args.seq_length, args.maxNumPeds, args.maxNumPeds, args.grid_size*args.grid_size], name="grid_data")

        # Variable to hold the value of the learning rate
        self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")

        # Output dimension of the model
        self.output_size = 14

        # Define embedding and output layers
        self.define_embedding_and_output_layers(args)

        # Define LSTM states for each pedestrian
        with tf.variable_scope("LSTM_states"):
            self.LSTM_states = tf.zeros([args.maxNumPeds, cell.state_size], name="LSTM_states")
            self.initial_states = tf.split(self.LSTM_states, args.maxNumPeds, 0)

        # Define hidden output states for each pedestrian
        with tf.variable_scope("Hidden_states"):
            # self.output_states = tf.zeros([args.maxNumPeds, cell.output_size], name="hidden_states")
            self.output_states = tf.split(tf.zeros([args.maxNumPeds, cell.output_size]), args.maxNumPeds, 0)

        # List of tensors each of shape args.maxNumPedsx3 corresponding to each frame in the sequence
        with tf.name_scope("frame_data_tensors"):
            # frame_data = tf.split(0, args.seq_length, self.input_data, name="frame_data")
            frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.input_data, args.seq_length, 0)]

        with tf.name_scope("frame_target_data_tensors"):
            # frame_target_data = tf.split(0, args.seq_length, self.target_data, name="frame_target_data")
            frame_target_data = [tf.squeeze(target_, [0]) for target_ in tf.split(self.target_data, args.seq_length, 0)]

        with tf.name_scope("grid_frame_data_tensors"):
            # This would contain a list of tensors each of shape MNP x MNP x (GS**2) encoding the mask
            # grid_frame_data = tf.split(0, args.seq_length, self.grid_data, name="grid_frame_data")
            grid_frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.grid_data, args.seq_length, 0)]

        # Cost
        with tf.name_scope("Cost_related_stuff"):
            self.cost = tf.constant(0.0, name="cost")
            self.counter = tf.constant(0.0, name="counter")
            self.increment = tf.constant(1.0, name="increment")

        # Containers to store output distribution parameters
        with tf.name_scope("Distribution_parameters_stuff"):
            # self.initial_output = tf.zeros([args.maxNumPeds, self.output_size], name="distribution_parameters")
            self.initial_output = tf.split(tf.zeros([args.maxNumPeds, self.output_size]), args.maxNumPeds, 0)

        # Tensor to represent non-existent ped
        with tf.name_scope("Non_existent_ped_stuff"):
            nonexistent_ped = tf.constant(0.0, name="zero_ped")

        # Iterate over each frame in the sequence
        for seq, frame in enumerate(frame_data):
            print "Frame number", seq

            current_frame_data = frame  # MNP x 3 tensor
            current_grid_frame_data = grid_frame_data[seq]  # MNP x MNP x (GS**2) tensor
            social_tensor = self.getSocialTensor(current_grid_frame_data, self.output_states)  # MNP x (GS**2 * RNN_size)
            # NOTE: Using a tensor of zeros as the social tensor
            # social_tensor = tf.zeros([args.maxNumPeds, args.grid_size*args.grid_size*args.rnn_size])

            for ped in range(args.maxNumPeds):
                print "Pedestrian Number", ped

                # pedID of the current pedestrian
                pedID = current_frame_data[ped, 0]

                with tf.name_scope("extract_input_ped"):
                    # Extract x and y along with vis_x and vis_y positions of the current ped
                    self.spatial_input = tf.slice(current_frame_data, [ped, 1], [1,4])  # Tensor of shape (1,4)
                    # Extract the social tensor of the current ped
                    self.tensor_input = tf.slice(social_tensor, [ped, 0], [1, args.grid_size*args.grid_size*args.rnn_size])  # Tensor of shape (1, g*g*r)

                with tf.name_scope("embeddings_operations"):
                    # Embed the spatial input
                    embedded_spatial_input = tf.nn.relu(tf.nn.xw_plus_b(self.spatial_input, self.embedding_w, self.embedding_b))
                    # Embed the tensor input
                    embedded_tensor_input = tf.nn.relu(tf.nn.xw_plus_b(self.tensor_input, self.embedding_t_w, self.embedding_t_b))

                with tf.name_scope("concatenate_embeddings"):
                    # Concatenate the embeddings
                    complete_input = tf.concat([embedded_spatial_input, embedded_tensor_input], 1)

                # One step of LSTM
                with tf.variable_scope("LSTM") as scope:
                    if seq > 0 or ped > 0:
                        scope.reuse_variables()
                    self.output_states[ped], self.initial_states[ped] = cell(complete_input, self.initial_states[ped])

                # with tf.name_scope("reshape_output"):
                # Store the output hidden state for the current pedestrian
                #    self.output_states[ped] = tf.reshape(tf.concat(1, output), [-1, args.rnn_size])
                #    print self.output_states[ped]

                # Apply the linear layer. Output would be a tensor of shape 1 x output_size
                with tf.name_scope("output_linear_layer"):
                    self.initial_output[ped] = tf.nn.xw_plus_b(self.output_states[ped], self.output_w, self.output_b)

                # with tf.name_scope("store_distribution_parameters"):
                #    # Store the distribution parameters for the current ped
                #    self.initial_output[ped] = output

                with tf.name_scope("extract_target_ped"):
                    # Extract x and y coordinates of the target data
                    # x_data and y_data would be tensors of shape 1 x 1
                    [x_data, y_data, x_prime, y_prime] = tf.split(tf.slice(frame_target_data[seq], [ped, 1], [1,4]), 4, 1)
                    target_pedID = frame_target_data[seq][ped, 0]

                with tf.name_scope("get_coef"):
                    # Extract coef from output of the linear output layer
                    [o_mux, o_muy, o_sx, o_sy, o_corr,o_mux_prime, o_muy_prime, o_sx_prime, o_sy_prime, o_corr_prime, o_rhoxvx, o_rhoxvy, o_rhoyvx, o_rhoyvy] = self.get_coef(self.initial_output[ped])

                with tf.name_scope("calculate_loss"):
                    # Calculate loss for the current ped
                    lossfunc = self.get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, o_mux_prime, o_muy_prime, o_sx_prime, o_sy_prime, o_corr_prime,o_rhoxvx,o_rhoxvy,o_rhoyvx,o_rhoyvy,x_data, y_data,x_prime,y_prime)

                with tf.name_scope("increment_cost"):
                    # If it is a non-existent ped, it should not contribute to cost
                    # If the ped doesn't exist in the next frame, he/she should not contribute to cost as well
                    self.cost = tf.where(tf.logical_or(tf.equal(pedID, nonexistent_ped), tf.equal(target_pedID, nonexistent_ped)), self.cost, tf.add(self.cost, lossfunc))
                    self.counter = tf.where(tf.logical_or(tf.equal(pedID, nonexistent_ped), tf.equal(target_pedID, nonexistent_ped)), self.counter, tf.add(self.counter, self.increment))

        with tf.name_scope("mean_cost"):
            # Mean of the cost
            self.cost = tf.div(self.cost, self.counter)

        # Get all trainable variables
        tvars = tf.trainable_variables()

        # L2 loss
        l2 = args.lambda_param*sum(tf.nn.l2_loss(tvar) for tvar in tvars)
        self.cost = self.cost + l2

        # Get the final LSTM states
        self.final_states = tf.concat(self.initial_states, 0)

        # Get the final distribution parameters
        self.final_output = self.initial_output

        # Compute gradients
        self.gradients = tf.gradients(self.cost, tvars)

        # Clip the gradients
        grads, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        # Define the optimizer
        optimizer = tf.train.RMSPropOptimizer(self.lr)

        # The train operator
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # Merge all summmaries
        # merged_summary_op = tf.merge_all_summaries()

    def define_embedding_and_output_layers(self, args):
        # Define variables for the spatial coordinates embedding layer
        with tf.variable_scope("coordinate_embedding"):
            self.embedding_w = tf.get_variable("embedding_w", [4, args.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.embedding_b = tf.get_variable("embedding_b", [args.embedding_size], initializer=tf.constant_initializer(0.1))

        # Define variables for the social tensor embedding layer
        with tf.variable_scope("tensor_embedding"):
            self.embedding_t_w = tf.get_variable("embedding_t_w", [args.grid_size*args.grid_size*args.rnn_size, args.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.embedding_t_b = tf.get_variable("embedding_t_b", [args.embedding_size], initializer=tf.constant_initializer(0.1))

        # Define variables for the output linear layer
        with tf.variable_scope("output_layer"):
            self.output_w = tf.get_variable("output_w", [args.rnn_size, self.output_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.output_b = tf.get_variable("output_b", [self.output_size], initializer=tf.constant_initializer(0.1))

    def tf_2d_normal(self, x, y, mux, muy, sx, sy, rho):
        '''
        Function that implements the PDF of a 2D normal distribution
        params:
        x : input x points
        y : input y points
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x
        sy : std dev of the distribution in y
        rho : Correlation factor of the distribution
        '''
        # eq 3 in the paper
        # and eq 24 & 25 in Graves (2013)
        # Calculate (x - mux) and (y-muy)
        normx = tf.subtract(x, mux)
        normy = tf.subtract(y, muy)
        # Calculate sx*sy
        sxsy = tf.multiply(sx, sy)
        # Calculate the exponential factor
        z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2*tf.div(tf.multiply(rho, tf.multiply(normx, normy)), sxsy)
        negRho = 1 - tf.square(rho)
        # Numerator
        result = tf.exp(tf.div(-z, 2*negRho))
        # Normalization constant
        denom = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negRho))
        # Final PDF calculation
        result = tf.div(result, denom)
        return result

    def tf_Nd_normal(self,x, y, mux, muy, sx, sy, rho, x_prime, y_prime, mux_prime, muy_prime, sx_prime, sy_prime,corr_prime, rhoxvx, rhoxvy, rhoyvx, rhoyvy):

        obs = tf.concat([x, y, x_prime, y_prime], 1)
        mu = tf.concat([mux, muy, mux_prime, muy_prime], 1)
        # cov = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]

        a11 = tf.multiply(sx, sx)
        a12 = tf.multiply(tf.multiply(sx, sy), rho)
        a13 = tf.multiply(tf.multiply(sx, sx_prime), rhoxvx)
        a14 = tf.multiply(tf.multiply(sx, sy_prime), rhoxvy)

        firstRow = tf.concat([a11, a12, a13, a14], 1)

        a21 = tf.multiply(tf.multiply(sx, sy), rho)
        a22 = tf.multiply(sy, sy)
        a23 = tf.multiply(tf.multiply(sy, sx_prime), rhoyvx)
        a24 = tf.multiply(tf.multiply(sy, sy_prime), rhoyvy)

        secondRow = tf.concat([a21, a22, a23, a24], 1)

        a31 = tf.multiply(tf.multiply(sx, sx_prime), rhoxvx)
        a32 = tf.multiply(tf.multiply(sy, sx_prime), rhoyvx)
        a33 = tf.multiply(sx_prime, sx_prime)
        a34 = tf.multiply(tf.multiply(sx_prime, sy_prime), corr_prime)

        thirdRow = tf.concat([a31, a32, a33, a34], 1)

        a41 = tf.multiply(tf.multiply(sx, sy_prime), rhoxvy)
        a42 = tf.multiply(tf.multiply(sy, sy_prime), rhoyvy)
        a43 = tf.multiply(tf.multiply(sx_prime, sy_prime), corr_prime)
        a44 = tf.multiply(sy_prime, sy_prime)

        fourthRow = tf.concat([a41, a42, a43, a44], 1)

        cov = tf.concat([firstRow, secondRow, thirdRow, fourthRow], 1)
        st = cov.shape[0]
        nst = cov.shape[0]
        st = tf.cast(st, tf.int32)
        ncov = tf.reshape(cov, [st, 4, 4])
        mu = tf.reshape(mu, [st, 4])
        obs = tf.reshape(obs, [st, 4])


        mvn = tf.contrib.distributions.MultivariateNormalFullCovariance(mu, ncov, ncov)

        temp = mvn.prob(obs)
        temp = tf.reshape(temp, [st, 1])
        result=temp


        return result

    def get_lossfunc(self,z_mux, z_muy, z_sx, z_sy, z_corr,z_mux_prime, z_muy_prime, z_sx_prime, z_sy_prime, z_corr_prime ,z_rhoxvx, z_rhoxvy, z_rhoyvx, z_rhoyvy, x_data, y_data, x_prime, y_prime):
        '''
        Function to calculate given a 2D distribution over x and y, and target data
        of observed x and y points
        params:
        z_mux : mean of the distribution in x
        z_muy : mean of the distribution in y
        z_sx : std dev of the distribution in x
        z_sy : std dev of the distribution in y
        z_rho : Correlation factor of the distribution
        x_data : target x points
        y_data : target y points
        '''
        # step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))

        # Calculate the PDF of the data w.r.t to the distribution
        # result0 = self.tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
        # resultTemp = self.tf_2d_normal(x_prime, y_prime, z_mux_prime, z_muy_prime, z_sx_prime, z_sy_prime, z_corr_prime)
        result0 = self.tf_Nd_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr,x_prime, y_prime, z_mux_prime, z_muy_prime, z_sx_prime, z_sy_prime, z_corr_prime,z_rhoxvx,z_rhoxvy,z_rhoyvx,z_rhoyvy)

        # For numerical stability purposes
        epsilon = 1e-20

        # Apply the log operation
        result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability
        # result2 = -tf.log(tf.maximum(resultTemp, epsilon))  # Numerical stability
        #
        # Sum up all log probabilities for each data point
        return tf.reduce_sum(result1)

    def get_coef(self, output):
        # eq 20 -> 22 of Graves (2013)

        z = output
        # Split the output into 5 parts corresponding to means, std devs and corr
        z_mux, z_muy, z_sx, z_sy, z_corr,z_mux_prime, z_muy_prime, z_sx_prime, z_sy_prime, z_corr_prime,z_rhoxvx, z_rhoxvy, z_rhoyvx, z_rhoyvy = tf.split(z, 14, 1)

        # The output must be exponentiated for the std devs
        z_sx = tf.exp(z_sx)
        z_sy = tf.exp(z_sy)

        z_sx_prime = tf.exp(z_sx_prime)
        z_sy_prime = tf.exp(z_sy_prime)
        # Tanh applied to keep it in the range [-1, 1]
        z_corr = tf.tanh(z_corr)
        z_corr_prime = tf.tanh(z_corr_prime)

        z_rhoxvx = tf.tanh(z_rhoxvx)
        z_rhoxvy = tf.tanh(z_rhoxvy)
        z_rhoyvx = tf.tanh(z_rhoyvx)
        z_rhoyvy = tf.tanh(z_rhoyvy)

        return [z_mux, z_muy, z_sx, z_sy, z_corr, z_mux_prime, z_muy_prime, z_sx_prime, z_sy_prime, z_corr_prime, z_rhoxvx, z_rhoxvy, z_rhoyvx, z_rhoyvy]

    def getSocialTensor(self, grid_frame_data, output_states):
        '''
        Computes the social tensor for all the maxNumPeds in the frame
        params:
        grid_frame_data : A tensor of shape MNP x MNP x (GS**2)
        output_states : A list of tensors each of shape 1 x RNN_size of length MNP
        '''
        # Create a zero tensor of shape MNP x (GS**2) x RNN_size
        social_tensor = tf.zeros([self.args.maxNumPeds, self.grid_size*self.grid_size, self.rnn_size], name="social_tensor")
        # Create a list of zero tensors each of shape 1 x (GS**2) x RNN_size of length MNP
        social_tensor = tf.split(social_tensor, self.args.maxNumPeds, 0)
        # Concatenate list of hidden states to form a tensor of shape MNP x RNN_size
        hidden_states = tf.concat(output_states, 0)
        # Split the grid_frame_data into grid_data for each pedestrians
        # Consists of a list of tensors each of shape 1 x MNP x (GS**2) of length MNP
        grid_frame_ped_data = tf.split( grid_frame_data, self.args.maxNumPeds,0)
        # Squeeze tensors to form MNP x (GS**2) matrices
        grid_frame_ped_data = [tf.squeeze(input_, [0]) for input_ in grid_frame_ped_data]

        # For each pedestrian
        for ped in range(self.args.maxNumPeds):
            # Compute social tensor for the current pedestrian
            with tf.name_scope("tensor_calculation"):
                social_tensor_ped = tf.matmul(tf.transpose(grid_frame_ped_data[ped]), hidden_states)
                social_tensor[ped] = tf.reshape(social_tensor_ped, [1, self.grid_size*self.grid_size, self.rnn_size])

        # Concatenate the social tensor from a list to a tensor of shape MNP x (GS**2) x RNN_size
        social_tensor = tf.concat(social_tensor, 0)
        # Reshape the tensor to match the dimensions MNP x (GS**2 * RNN_size)
        social_tensor = tf.reshape(social_tensor, [self.args.maxNumPeds, self.grid_size*self.grid_size*self.rnn_size])
        return social_tensor

    def sample_gaussian_2d(self, mux, muy, muvx, muvy, sx, sy, svx, svy, rhoxy, rhovxvy, rhoxvx, rhoxvy, rhoyvx, rhoyvy):
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
        cov = np.array([[sx * sx, rhoxy * sx * sy, rhoxvx * sx * svx, rhoxvy * sx * svy],
                        [rhoxy * sx * sy, sy * sy, rhoyvx * sy * svx, rhoyvy * sy * svy],
                        [rhoxvx * sx * svx, rhoyvx * sy * svx, svx * svx, rhovxvy * svx * svy],
                        [rhoxvy * sx * svy, rhoyvy * sy * svy, rhovxvy * svx * svy, svy * svy]])
        # Sample a point from the multivariate normal distribution
        pcov = self.nearestPD(cov)
        x = np.random.multivariate_normal(mean, pcov, 1)

        return x[0][0], x[0][1],x[0][2],x[0][3],cov,pcov

    def nearestPD(self,A):
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
        factor = 1e-7
        A3[idx] = A3[idx] + factor

        if self.isPD(A3):
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
        while not self.isPD(A3):
            # print('in while')
            mineig = np.min(np.real(la.eigvals(A3)))
            # mineig=0.01
            factor = (-mineig * k ** 2 + spacing)
            # factor=factor*1000
            A3 += I * factor

            # np.diag(A3)=2
            k += 1

        return A3

    def isPD(self,B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlgError:
            return False

    def sample(self, sess, traj, grid, dimensions, true_traj, num=10):
        # traj is a sequence of frames (of length obs_length)
        # so traj shape is (obs_length x maxNumPeds x 3)
        # grid is a tensor of shape obs_length x maxNumPeds x maxNumPeds x (gs**2)
        states = sess.run(self.LSTM_states)
        # print "Fitting"
        # For each frame in the sequence
        for index, frame in enumerate(traj[:-1]):
            data = np.reshape(frame, (1, self.maxNumPeds, 5))
            target_data = np.reshape(traj[index+1], (1, self.maxNumPeds, 5))
            grid_data = np.reshape(grid[index, :], (1, self.maxNumPeds, self.maxNumPeds, self.grid_size*self.grid_size))

            feed = {self.input_data: data, self.LSTM_states: states, self.grid_data: grid_data, self.target_data: target_data}

            [states, cost] = sess.run([self.final_states, self.cost], feed)
            # print cost

        ret = traj

        last_frame = traj[-1]

        prev_data = np.reshape(last_frame, (1, self.maxNumPeds, 5))
        prev_grid_data = np.reshape(grid[-1], (1, self.maxNumPeds, self.maxNumPeds, self.grid_size*self.grid_size))

        prev_target_data = np.reshape(true_traj[traj.shape[0]], (1, self.maxNumPeds, 5))
        pcov_list = []
        cov_list = []
        # Prediction
        for t in range(num):
            # print "**** NEW PREDICTION TIME STEP", t, "****"
            feed = {self.input_data: prev_data, self.LSTM_states: states, self.grid_data: prev_grid_data, self.target_data: prev_target_data}
            [output, states, cost] = sess.run([self.final_output, self.final_states, self.cost], feed)
            # print "Cost", cost
            # Output is a list of lists where the inner lists contain matrices of shape 1x5. The outer list contains only one element (since seq_length=1) and the inner list contains maxNumPeds elements
            # output = output[0]
            newpos = np.zeros((1, self.maxNumPeds, 5))
            for pedindex, pedoutput in enumerate(output):
                [o_mux, o_muy, o_sx, o_sy, o_corr, o_mux_prime, o_muy_prime, o_sx_prime, o_sy_prime, o_corr_prime,o_rhoxvx, o_rhoxvy, o_rhoyvx, o_rhoyvy] = np.split(pedoutput[0], 14, 0)
                mux, muy, sx, sy, corr = o_mux[0], o_muy[0], np.exp(o_sx[0]), np.exp(o_sy[0]), np.tanh(o_corr[0])
                mux_prime, muy_prime, sx_prime, sy_prime, corr_prime = o_mux_prime[0], o_muy_prime[0], np.exp(o_sx_prime[0]), np.exp(o_sy_prime[0]), np.tanh(o_corr_prime[0])
                rhoxvx, rhoxvy, rhoyvx, rhoyvy= np.tanh(o_rhoxvx[0]), np.tanh(o_rhoxvy[0]), np.tanh(o_rhoyvx[0]), np.tanh(o_rhoyvy[0])

                # next_x, next_y = self.sample_gaussian_2d(mux, muy, sx, sy, corr)
                next_x, next_y, next_x_prime, next_y_prime, cov,pcov = self.sample_gaussian_2d(mux, muy, mux_prime, muy_prime,sx, sy, sx_prime, sy_prime,o_corr, corr_prime,rhoxvx, rhoxvy, rhoyvx, rhoyvy)
                cov_list.append(cov)
                pcov_list.append(pcov)


                newpos[0, pedindex, :] = [prev_data[0, pedindex, 0], next_x, next_y, next_x_prime, next_y_prime]
            ret = np.vstack((ret, newpos))
            prev_data = newpos
            prev_grid_data = getSequenceGridMask(prev_data, dimensions, self.args.neighborhood_size, self.grid_size)
            if t != num - 1:
                prev_target_data = np.reshape(true_traj[traj.shape[0] + t + 1], (1, self.maxNumPeds, 5))

        # The returned ret is of shape (obs_length+pred_length) x maxNumPeds x 3
        return ret, cov_list, pcov_list
