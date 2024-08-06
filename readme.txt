Condition Monitoring of Hydraulic Systems

Overview

This project involves the analysis of a dataset obtained from a hydraulic test rig. The dataset is designed for condition monitoring and fault detection in hydraulic systems using multivariate sensor data. The primary objective is to classify the conditions of four hydraulic components (cooler, valve, pump, and accumulator) using machine learning techniques.

Dataset Description

Source: The data was created by ZeMA gGmbH and is used for condition assessment of hydraulic test rigs. The dataset is provided by M. Bastuck, T. Schneider, and the creator, Nikolai Helwig.
Data Type: Multivariate, Time-Series
Task: Classification, Regression
Number of Instances: 2205
Number of Attributes: 43680 (across multiple sensors with different sampling rates)

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

The main goal of this project is to classify the condition of the cooler using the available multivariate sensor data. The code implements the following steps:

Data Loading: Read and parse the sensor data and the profile file to prepare the dataset.
Data Preprocessing:
Remove constant columns.
Standardize the data using StandardScaler.
Perform feature selection using SelectKBest based on mutual information regression scores.
Model Training and Evaluation:
Split the data into training and test sets to prevent data leakage.
Train and evaluate models using RandomForestClassifier, SVC (Support Vector Machine), and KNeighborsClassifier.
Utilize GridSearchCV to optimize hyperparameters for SVC and KNN using a pipeline to ensure preprocessing is applied consistently.
Feature Analysis: Analyze the selected features and visualize their correlation to understand their significance.
