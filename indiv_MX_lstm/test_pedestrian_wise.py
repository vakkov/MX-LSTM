import numpy as np
import tensorflow as tf

import os
import pickle
import argparse

from utils import DataLoader
from model_irtFranz import Model_irfr
import scipy.io


def get_mean_error(predicted_traj, true_traj, observed_length):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true trajectory
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed
    '''
    # The data structure to store all errors
    error = np.zeros(len(true_traj) - observed_length)
    # For each point in the predicted part of the trajectory
    for i in range(observed_length, len(true_traj)):
        # The predicted position
        pred_pos = predicted_traj[i, [0,1]]
        # The true position
        true_pos = true_traj[i, [0,1]]

        # The euclidean distance is the error
        error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)


def main():
    # Define the parser
    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=8,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=12,
                        help='Predicted length of the trajectory')
    # Test dataset
    parser.add_argument('--test_dataset', type=int, default=1,
                        help='Dataset to be tested on')

    # Read the arguments
    sample_args = parser.parse_args()

    # Load the saved arguments to the model from the config file
    with open(os.path.join('save_lstm', 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    # Initialize with the saved args
    model = Model_irfr(saved_args, True)
    # Initialize TensorFlow session
    sess = tf.InteractiveSession()
    # Initialize TensorFlow saver
    saver = tf.train.Saver()

    # Get the checkpoint state to load the model from
    ckpt = tf.train.get_checkpoint_state('save_lstm')
    print('loading model: ', ckpt.model_checkpoint_path)

    # Restore the model at the checpoint
    saver.restore(sess, ckpt.model_checkpoint_path)
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run((variables_names))

    # Dataset to get data from
    dataset = [sample_args.test_dataset]

    # Initialize the dataloader object to
    # Get sequences of length obs_length+pred_length
    data_loader = DataLoader(1, sample_args.pred_length + sample_args.obs_length, dataset, True)

    # Reset the data pointers of the data loader object
    data_loader.reset_batch_pointer()
    annoFile=data_loader.preprocess(data_loader.data_dirs,data_loader.data_file)
    totalPed=len(annoFile)
    total_error = 0.
    counter=0
    results = []
    for it in range(totalPed):
        thisPed=annoFile[it+1]
        obs, completeTraj, pedInformationBatch, validFlag=data_loader.get_ped_batch(thisPed,sample_args.obs_length,sample_args.pred_length)
        if validFlag:
            print "Pedestrain Number ", it
            for i in range(len(obs)):
                obs_traj=obs[i]
                obs_traj=np.transpose(obs_traj)
                batchComp=completeTraj[i]
                batchComp=np.transpose(batchComp)
                thisBatchInfomration=pedInformationBatch[i]
                thisBatchInfomration=np.transpose(thisBatchInfomration)
                complete_traj,params, cov_list,mean_list = model.sample(sess, obs_traj, batchComp, num=sample_args.pred_length)
                total_error += get_mean_error(complete_traj, batchComp, sample_args.obs_length)

                results.append((batchComp, complete_traj, sample_args.obs_length, thisBatchInfomration,cov_list,mean_list))
                filename = 'VisualizeUtils/pedSpecifiCovMU/' + str(counter)+'_z1_covmat.mat'
                # scipy.io.savemat(filename, mdict={'covMat': cov_list,'muMat':mean_list})
                counter = counter + 1
               # print "iteration",i
    print "Total mean error of the model is ", total_error / counter
    with open(os.path.join('save_lstm', 'social_results.pkl'), 'wb') as f:
         pickle.dump(results, f)
    path = 'save_lstm/'
    filename = path + 'social_results.pkl'
    f = open(filename, 'rb')
    results = pickle.load(f)
    print "Saving results as .mat"
    filesave = 'VisualizeUtils/' + 'z2_.mat'
    scipy.io.savemat(filesave, mdict={'data': results})


if __name__ == '__main__':
    main()
