
This is work is available for research purposes. If you are using this code for your work, please cite the following paper

@InProceedings{Hasan_2018_CVPR,
author = {Hasan, Irtiza and Setti, Francesco and Tsesmelis, Theodore and Del Bue, Alessio and Galasso, Fabio and Cristani, Marco},
title = {MX-LSTM: Mixing Tracklets and Vislets to Jointly Forecast Trajectories and Head Poses},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}




------------------------------------------------------------------------------------------------------------------------------------
Requiremnets python 2.7

easydict==1.7


matplotlib==2.1.2


numpy==1.14.1


scikit-image==0.13.1


scikit-learn==0.19.1


scipy==1.0.0


sympy==1.2


tensorflow==1.5.0


tensorflow-tensorboard==1.5.1

------------------------------------------------------------------------------------------------------------------------------------

In order to train the model run 

1) social_train.py, you can identify which data to train on in the file social_utils.py


2) Saved model will be in save irectory


3) In order to evaluate, run test_pedestrian_wise_working.py. Again specify the model and dataset (social_utils).


4) Final output of the model will be saved in "filesave". 


5) The code is not clean nor optimize Evaluation is done using matlab script evallstm.m specify the filename(output of test_pedestrian_wise_working.py)


