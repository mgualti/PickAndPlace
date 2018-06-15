This is a research code for simulated experiments in the paper, "Pick and Place Without Geometric Object Models". As a research code, it may be difficult to get running on your machine. It may be poorly documented. Also, it is not under active development, and nobody may remember exactly how it works. It may be necessary to change some hard-coded paths. Be sure to install all of the prerequisites. Originally was run on Ubuntu 14.04.

Prerequisites:
 - OpenRAVE
 - Caffe
 - Matlab with Python interface

Contents:
 - simulation/caffe: model and weight files for trained pick-place agents.
 - simulation/matlab: the matlab part of the code for detecting mechanically stable grasps.
 - simulation/matlab/gpd2: Matlab version of GPD.
 - simulation/matlab/gpd2/data/CAFFEfiles: model and weight files trained for 3DNet bottles and mugs.
 - simulation/openrave: openrave environment files.
 - simulation/python0: for training the real robot on pick-place (similar to python2 but with a different reward function).
 - simulation/python1: for training the multi-step scenario.
 - simulation/python2: for reproducing the two-step simulation results.
 - RunBatchSimulation.sh: script for running several simulations sequentially.
 
