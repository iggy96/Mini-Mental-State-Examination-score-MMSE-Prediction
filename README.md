# Mini-Mental-State-Examination-score-MMSE-Prediction
Development of an Artificial Neural Network (ANN) model for the prediction of MMSE score from mid line scalp EEG
Objective: To examine the possibility of predicting MMSE scores from midline EEG
Methods: an artificial neural network was developed and trained on a dataset containing spectral features (alpha, beta, theta, delta and gamma-band power, peak power, and frequency) and eventrelated potential features (LPC, N200, P300, N400 Amplitudes). These features were extracted from EEG collected from midline electrodes (Fz, Cz, and Pz).
Results: The model with all EEG features utilized for training (i.e., model did not undergo feature selection), performed better than the model with 75% of the features from the dataset (i.e., model underwent feature selection).
Conclusion: EEG features from midline electrodes can be combined with an artificial neural network to predict the MMSE score to eliminate the downsides experienced during the traditional methods of deriving an MMSE score
