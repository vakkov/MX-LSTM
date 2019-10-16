
This work is available for research purposes. If you are using this code for your work, please cite the following papers

@InProceedings{Hasan_2018_CVPR,
author = {Hasan, Irtiza and Setti, Francesco and Tsesmelis, Theodore and Del Bue, Alessio and Galasso, Fabio and Cristani, Marco},
title = {MX-LSTM: Mixing Tracklets and Vislets to Jointly Forecast Trajectories and Head Poses},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}

@article{hasan2019forecasting,
  title={Forecasting People Trajectories and Head Poses by Jointly Reasoning on Tracklets and Vislets},
  author={Hasan, Irtiza and Setti, Francesco and Tsesmelis, Theodore and Belagiannis, Vasileios and Amin, Sikandar and Del Bue, Alessio and Cristani, Marco and Galasso, Fabio},
  journal={arXiv preprint arXiv:1901.02000},
  year={2019}
}

We have provided the benchmarking code for MX-LSTM. Compete training and testing code is not provided at the moment. We provide raw trajectories for all 3-UCY sequences along with homography matrix and a script to plot trajectories on the image plane

Instructions:-
Evaluation Script can be seen in MX-LSTM/VisualizeUtils/socialLSTMEvaluate.m

1) In roder to run that script, please donwload data files and output of MX-LSTM from the link below
(https://drive.google.com/open?id=153s1mLDOBGjO25bHv2I5x_xptX0k4jyE)

2) Copy all files in the dataFiles to MX-LSTM/VisualizeUtils/dataFiles

3) Run socialLSTMEvaluate.m to evaluate

4) You can also set the flag genVisualization to 1 inorder to plot trajectories on the images (you need to download images).

5) As we are conitniously updating code and models, you migth see some discrepenacy in the numbers, it is due to the sampling from gaussian.










