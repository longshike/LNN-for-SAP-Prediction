# LNN-for-SAP-Prediction
This repository is the code and data for the thesis “Constructing a prediction model for acute pancreatitis severity based on liquid neural network”, which constructs a prediction model for acute pancreatitis severity based on liquid neural network. Constructing a prediction model for acute pancreatitis severity based on liquid neural network.

1.Heatmap_test.py is the program that reads the data and generates a heat map for correlation analysis.

2.LR_features_test.py plots the relationship between feature combinations and AUC in increasing order of feature size(logistic regression).

3.DCT_features_test.py plots the relationship between feature combinations and AUC in increasing order of feature size(decision tree).

4.RF_features_test.py plots the relationship between feature combinations and AUC in increasing order of feature size(random forest).

5.XGB_features_test.py plots the relationship between feature combinations and AUC in increasing order of feature size(XGBoost).

6.Features_compare_models.py describes the best combination of features and all combinations of features in different models, resulting in an AUC worth comparing.

7.Features_bar.py is designed to generate a histogram of AUC comparisons of feature combinations between different models

8.SMOTE_compare_models.py describes the SMOTE use and no use of SMOTE in different models, resulting in an AUC worth comparing.

9.ROC_different_model.py generates a comparison of ROC plots for different models.

10.LNN_yizhi3.py is the python code for porting the LNN.

11.test_data.xlsx is the data after preprocessing, totaling 64 features

12.zhenglishuju_v1.0.xlsx is the data before preprocessing, totaling 105 features.
