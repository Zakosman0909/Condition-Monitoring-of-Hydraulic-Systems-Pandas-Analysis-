Condition Monitoring of Hydraulic Systems

Overview
This project focuses on the analysis of a dataset obtained from a hydraulic test rig, designed for condition monitoring and fault detection in hydraulic systems. The primary objective is to classify the conditions of four hydraulic components (cooler, valve, pump, and accumulator) using advanced machine learning techniques.

Dataset Description
Source: The data was created by ZeMA gGmbH for condition assessment of hydraulic test rigs. The dataset is provided by M. Bastuck, T. Schneider, and Nikolai Helwig.
Data Type: Multivariate, Time-Series
Task: Classification, Regression
Number of Instances: 2205 cycles
Number of Attributes: 43,680 attributes (across multiple sensors with different sampling rates)

Sensors and Sampling Rates
Pressure Sensors (PS1-6): 100 Hz, 6000 attributes per sensor
Motor Power Sensor (EPS1): 100 Hz, 6000 attributes
Volume Flow Sensors (FS1, FS2): 10 Hz, 600 attributes per sensor
Temperature Sensors (TS1-4): 1 Hz, 60 attributes per sensor
Vibration Sensor (VS1): 1 Hz, 60 attributes
Efficiency Factor (SE): 1 Hz, 60 attributes
Virtual Cooling Efficiency (CE): 1 Hz, 60 attributes
Virtual Cooling Power (CP): 1 Hz, 60 attributes
Target Variables
The dataset includes condition annotations for each cycle in the profile.txt file, covering:

Cooler Condition:
3%: Near failure
20%: Reduced efficiency
100%: Full efficiency

Valve Condition:
100%: Optimal
90%: Small lag
80%: Severe lag
73%: Near failure

Internal Pump Leakage:
0: No leakage
1: Weak leakage
2: Severe leakage

Hydraulic Accumulator Pressure:
130 bar: Optimal
115 bar: Slightly reduced
100 bar: Severely reduced
90 bar: Near failure

Stable Flag:
0: Stable conditions
1: Static conditions not yet reached

Project Goals
The main goal of this project is to classify the condition of the cooler using the available multivariate sensor data. The project workflow involves the following steps:

1. Data Loading
Read and parse the sensor data files and the profile.txt file to prepare the dataset.
2. Data Preprocessing
Resampling: Downsample all sensor data to a common frequency (1 Hz) to align the data from different sensors.
Feature Aggregation: Compute rolling statistics (mean and standard deviation) for each sensor's data to reduce dimensionality and noise.
Ensure Numeric: Convert all features to a numeric format to ensure compatibility with machine learning algorithms.
3. Model Training and Evaluation
Train-Test Split: Split the data into training and test sets to prevent data leakage and ensure robust evaluation.
Pipeline Construction: Utilize a machine learning pipeline that includes scaling, feature selection, and classification.
Models: Train and evaluate models using RandomForestClassifier and XGBoostClassifier.
Grid Search: Use GridSearchCV to optimize hyperparameters for the models within the pipeline.
Performance Metrics: Evaluate models using accuracy, precision, recall, and F1-score. Generate classification reports to summarize model performance.
4. Feature Analysis
Feature Selection: Apply SelectKBest based on mutual information regression to identify the most important features.
Correlation Analysis: Visualize the correlation between selected features using a heatmap to understand feature significance and relationships.


