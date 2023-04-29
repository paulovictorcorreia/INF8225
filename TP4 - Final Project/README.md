# EEG Signal Classification

The final project consists of comparing multiple sequence processing models to classify samples of EEG signals as either being from an epileptic seizure or not.

The repport describing and explaining the experiments is in the file final_repport.pdf.


The models are:
* Transformer Encoders with Unsupervised Pretraining
* LSTM
* 1D-CNN + LSTM
* MLP
* Gated Transformer Network

And also some hybrid models used in literature.

Data source: https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition

# Files

The main files in this project are:

* main.ipynb
* exploratory_data_analysis.ipynb
* preprocessing.py

The main file contains the code for all of the models used in the project, built with pytorch framework. It also displays the experiment results of the project.

The exploratory data analysis file simply contains basic exploration over the EEG signals.

The preprocessing file simply contains all the preprocessing steps to prepare the data and turn them into the input of our model. Also, we run this file inside the main file.