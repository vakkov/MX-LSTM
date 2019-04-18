import os
import pickle
import numpy as np
import random


# TODO : (improve) Add functionality to retrieve data only from specific datasets
class DataLoader():

    def __init__(self, batch_size=50, seq_length=5, datasets=[0], forcePreProcess=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : RNN sequence length
        '''
        # List of data directories where raw data resides
        # self.data_dirs = ['../data/ucy/zara/zara01','../data/ucy/univ']
        # self.data_dirs = ['../data/ucy/zara/z2_fast_slow']
        self.data_dirs = ['../data/ucy/zara/zara02']
        # self.data_dirs = ['../data/ucy/univ','../data/ucy/zara/zara01']
        # self.data_dirs = ['../data/ucy/franz']

        self.used_data_dirs = self.data_dirs

        # Data directory where the pre-processed pickle file resides
        self.data_dir = '../data'

        # Store the batch size and the sequence length arguments
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.myCounter=1
        # self.txtFileName= 'z1_norm_mean_std_vislet.csv'

        # Define the path of the file in which the data needs to be stored
        data_file = os.path.join(self.data_dir, "trajectories.cpkl")
        self.data_file=data_file

        # If the file doesn't exist already or if forcePreProcess is true
        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files
            self.preprocess(self.used_data_dirs, data_file)

        # Load the data from the pickled file
        self.load_preprocessed(data_file)
        # Reset all the pointers
        self.reset_batch_pointer()

    def preprocess(self, data_dirs, data_file):
        '''
        The function that pre-processes the pixel_pos.csv files of each dataset
        into data that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        # all_ped_data would be a dictionary with mapping from each ped to their
        # trajectories given by matrix 3 x numPoints with each column
        # in the order x, y, frameId
        # Pedestrians from all datasets are combined
        # Dataset pedestrian indices are stored in dataset_indices
        all_ped_data = {}
        dataset_indices = []
        current_ped = 0
        # For each dataset
        for directory in data_dirs:
            # Define the path to its respective csv file
            file_path = os.path.join(directory, 'zero_zero_mean_std_vislet.csv')

            # Load data from the csv file
            # Data is a 6 x numTrajPoints matrix
            # where each column is a (frameId, pedId, y, x,y',x') vector
            data = np.genfromtxt(file_path, delimiter=',')

            # Get the number of pedestrians in the current dataset
            numPeds = np.size(np.unique(data[1, :]))

            # For each pedestrian in the dataset
            for ped in range(1, numPeds+1):
                # Extract trajectory of the current ped
                traj = data[:, data[1, :] == ped]
                # Format it as (x, y,x',y', frameId)
                traj = traj[[3, 2, 5, 4, 0, 1], :]

                # Store this in the dictionary
                all_ped_data[current_ped + ped] = traj

            # Current dataset done
            dataset_indices.append(current_ped+numPeds)
            current_ped += numPeds

        # The complete data is a tuple of all pedestrian data, and dataset ped indices
        complete_data = (all_ped_data, dataset_indices)
     #   annoFile=np.transpose(all_ped_data)
        # Store the complete data into the pickle file
        f = open(data_file, "wb")
        pickle.dump(complete_data, f, protocol=2)
        f.close()
        return all_ped_data

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : The path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, "rb")
        self.raw_data = pickle.load(f)
        f.close()

        # Get the pedestrian data from the pickle file
        all_ped_data = self.raw_data[0]
        # Not using dataset_indices for now
        # dataset_indices = self.raw_data[1]

        # Construct the data with sequences(or trajectories) longer than seq_length
        self.data = []
        self.frame_ped_id = []  # to keep track of frame and ped id for visualization purposes
        counter = 0

        # For each pedestrian in the data
        for ped in all_ped_data:
            # Extract his trajectory
            traj = all_ped_data[ped]
            # If the length of the trajectory is greater than seq_length (+2 as we need both source and target data)
            if traj.shape[1] > (self.seq_length+2):
                # TODO: (Improve) Store only the (x,y) coordinates for now
                # I am enforcing x' and y' as well in the data (irtiza)
                self.data.append(traj[[0, 1, 2, 3], :].T)
                self.frame_ped_id.append(traj[[4, 5], :].T)
                # Number of batches this datapoint is worth
                counter += int(traj.shape[1] / ((self.seq_length+2)))

        # Calculate the number of batches (each of batch_size) in the data
        self.num_batches = int(counter / self.batch_size)

    def next_batch(self):
        '''
        Function to get the next batch of point s
        '''
        # List of source and target data for the current batch
        x_batch = []
        y_batch = []

        x_fr_ped_batch = []  # source information
        y_fr_ped_batch = []  # target information
        # For each sequence in the batch
        for i in range(self.batch_size):
            # Extract the trajectory of the pedestrian pointed out by self.pointer
            traj = self.data[self.pointer]
            # Number of sequences corresponding to his trajectory
            pedInfo = self.frame_ped_id[self.pointer]
            n_batch = int(traj.shape[0] / (self.seq_length+2))
            # Randomly sample a index from which his trajectory is to be considered
            idx = random.randint(0, traj.shape[0] - self.seq_length - 2)
            # Append the trajectory from idx until seq_length into source and target data
            x_batch.append(np.copy(traj[idx:idx+self.seq_length, :]))
            y_batch.append(np.copy(traj[idx+1:idx+self.seq_length+1, :]))
            x_fr_ped_batch.append(np.copy(pedInfo[idx:idx + self.seq_length, :]))
            y_fr_ped_batch.append(np.copy(pedInfo[idx + 1:idx + self.seq_length + 1, :]))

            if random.random() < (1.0/float(n_batch)):
                # Adjust sampling probability
                # if this is a long datapoint, sample this data more with
                # higher probability
                self.tick_batch_pointer()

        return x_batch, y_batch, x_fr_ped_batch, y_fr_ped_batch

    def tick_batch_pointer(self):
        '''
        Advance the data pointer
        '''
        self.pointer += 1
        if (self.pointer >= len(self.data)):
            self.pointer = 0

    def reset_batch_pointer(self):
        '''
        Reset the data pointer
        '''
        self.pointer = 0

    def get_ped_batch(self,annoFile,obsLength,predInterval):
        # transforming into a column wise matrix from row wise matyrix
        #annoFile=np.transpose(annoFile)
        total_traj=obsLength+predInterval
        ped_batch=[]
        batch_complete_traj=[]
        information_Batch=[]
        samplingInterval=0
        timeSeries=annoFile.shape[1]
        validPedestrian=0
        targetCounter = 0  # to check if target is within bounds
        # & ((targetCounter + 1 + (self.seq_length) + 1) < len(frame_data)) & (
        # (samplingInterval + 1 + (self.seq_length)) < len(thisPedFrames)):
        while (samplingInterval<timeSeries) & (samplingInterval+obsLength<timeSeries):

            # pick obs trajectory

            diffInterval=timeSeries-(samplingInterval+obsLength)
            if(diffInterval>=predInterval):
                validPedestrian = 1
                ped_batch.append(np.copy(annoFile[[0, 1, 2, 3], samplingInterval:samplingInterval + obsLength]))
                batch_complete_traj.append(np.copy(annoFile[[0,1,2,3],samplingInterval:samplingInterval+total_traj]))
                information_Batch.append(np.copy(annoFile[[4,5], samplingInterval:samplingInterval + total_traj]))
            # else:
            #     batch_complete_traj.append(np.copy(annoFile[[0, 1, 2, 3], samplingInterval:samplingInterval+(obsLength+diffInterval)]))
            #     information_Batch.append(np.copy(annoFile[[4, 5], samplingInterval:samplingInterval+(obsLength+diffInterval)]))
            samplingInterval=samplingInterval+3


        return ped_batch, batch_complete_traj, information_Batch,validPedestrian
        #for i in range



