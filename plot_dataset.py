import numpy as np
import pandas as pd
import glob
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import data_transforms as dt  

def file_read(file_path):
    """Reads the data file into a Pandas DataFrame."""
    try:
        return pd.read_csv(file_path, sep='\t', header=None)
    except pd.errors.EmptyDataError:
        print(f"Warning: The file {file_path} is empty and will be skipped.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is empty

def main():
    print("Running Test on Hydraulic Systems dataset")

    # Set the directory containing the dataset files
    dataset_dir = os.path.abspath(os.path.dirname(__file__))
    print(f"Dataset directory: {dataset_dir}") 
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.txt")))

    if not files:
        sys.exit('No dataset found')

    # Initialize dictionary to store raw and scaled data
    data_dict = {}
    scaled_data_dict = {} 
    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f'Loading {file_name}...')

        # Skip non-data files
        if 'description' in file_name or 'documentation' in file_name or 'profile' in file_name or 'readme' in file_name:
            continue
        else:
            data = file_read(file_path)
            if data.empty:  # Skip the file if it's empty
                continue
            key = file_name.split('.')[0]
            print(f'{key}, Shape: {data.shape}')
            data_dict[key] = data
            print(f"Data from {file_name}:\n", data.head())

            # Remove constant columns
            var_thres = VarianceThreshold(threshold=0)
            var_thres.fit(data)
            constant_columns = [column for column in data.columns if column not in data.columns[var_thres.get_support()]]
            data = data.drop(columns=constant_columns)

            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            scaled_data_dict[key] = pd.DataFrame(scaled_data, columns=data.columns)
                # print(scaled_data_dict[key].head())

            # print('Dataframe Info/Description')
            # print(data.info())
            # print(data.describe())

    # Load the target variable data
    ans_df = pd.read_csv(os.path.join(dataset_dir, 'profile.txt'), sep='\t', header=None,
                         names=['Cooler', 'Valve', 'Pump', 'Accumulator', 'Stable'])
    y = ans_df['Cooler'].values  # Target variable: Cooler condition

    # Concatenate all scaled data into a single DataFrame
    X = pd.concat([scaled_data_dict[key] for key in scaled_data_dict.keys() if not scaled_data_dict[key].empty], axis=1)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')

    # Data preprocessing and feature selection pipeline
    preprocessing_pipeline = Pipeline(steps=[
        ('var_thres', VarianceThreshold(threshold=0)),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(score_func=mutual_info_regression, k=10))
    ])

    # Fit the preprocessing pipeline on the training data
    X_train_transformed = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_test_transformed = preprocessing_pipeline.transform(X_test)

    # Extract selected feature names
    selected_features = X.columns[preprocessing_pipeline.named_steps['feature_selection'].get_support()]
    selected_features_df = pd.DataFrame({
        'Feature': selected_features,
        'Score': preprocessing_pipeline.named_steps['feature_selection'].scores_[preprocessing_pipeline.named_steps['feature_selection'].get_support()]
    })

    print(f"Selected features:\n{selected_features_df}")

    # Train and evaluate Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_transformed, y_train)
    y_pred_rf = rf_model.predict(X_test_transformed)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf)

    print(f'RandomForestClassifier Accuracy: {accuracy_rf}')
    print('RandomForestClassifier Classification Report:')
    print(report_rf)

  # Define parameter grid and pipeline for SVM
    svc_pipeline = make_pipeline(preprocessing_pipeline, SVC())
    param_grid_svc = {
        'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
        'svc__degree': [2, 3, 4]
    }
    grid_search_svc = GridSearchCV(svc_pipeline, param_grid_svc, refit=True, cv=5)
    grid_search_svc.fit(X_train, y_train)
    best_svm = grid_search_svc.best_estimator_

    y_pred_svm = best_svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    report_svm = classification_report(y_test, y_pred_svm)

    print(f'Best SVM parameters: {grid_search_svc.best_params_}')
    print(f'SVM Accuracy: {accuracy_svm}')
    print('SVM Classification Report:')
    print(report_svm)

    # Define parameter grid and pipeline for KNN
    knn_pipeline = make_pipeline(preprocessing_pipeline, KNeighborsClassifier())
    param_grid_knn = {'kneighborsclassifier__n_neighbors': np.arange(1, 31)}
    grid_search_knn = GridSearchCV(knn_pipeline, param_grid_knn, cv=5)
    grid_search_knn.fit(X_train, y_train)
    best_k = grid_search_knn.best_params_['kneighborsclassifier__n_neighbors']
    
    y_pred_knn = grid_search_knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    report_knn = classification_report(y_test, y_pred_knn)

    print(f'Optimal number of neighbors: {best_k}')
    print(f'KNN Accuracy: {accuracy_knn}')
    print('KNN Classification Report:')
    print(report_knn)

    # Correlation matrix for selected features
    X_train_transformed = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_selected = pd.DataFrame(X_train_transformed, columns=selected_features_df['Feature'])

    corr_matrix = X_selected.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Heatmap of Cooler Condition')
    # plt.show()


#     unstables = np.where(ans_df['Stable'].values == 0)[0]
#     stables = np.where(ans_df['Stable'].values == 1)[0]

#     num = 250

#     # Plot Time series
#     num_keys = len(data_dict.keys())
#     cols = 4
#     rows = int(np.ceil(num_keys / cols))

#     plt.figure(figsize=(20, rows * 4))
#     for i, key in enumerate(data_dict.keys()):
#         plt.subplot(rows, cols, i + 1)
#         plt.title(f'{key} - Time Series', fontsize=6)
#         for v in stables[:num - 1]:
#             plt.plot(data_dict[key].iloc[v], color='green')
#         plt.plot(data_dict[key].iloc[stables[num]], label='stable', color='green')
#         for v in unstables[:num]:
#             plt.plot(data_dict[key].iloc[v], color='red', alpha=0.5)
#         plt.plot(data_dict[key].iloc[unstables[num]], label='unstable', color='red', alpha=0.5)
#         plt.legend(loc='upper right', fontsize=8)
#         plt.xticks(fontsize=6)
#         plt.yticks(fontsize=6)

#     plt.tight_layout(pad=3.0)
#     plt.show()

#     # Plot Fourier transform
#     plt.figure(figsize=(20, rows * 4))
#     for i, key in enumerate(data_dict.keys()):
#         df_fft_mag, df_fft_real, df_fft_imag = dt.time_to_frequency(data_dict[key])
#         plt.subplot(rows, cols, i + 1)
#         plt.title(f'{key} - Fourier Transform', fontsize=6)
#         for v in stables[:num]:
#             plt.plot(df_fft_mag.iloc[v], color='green')
#         for v in unstables[:num]:
#             plt.plot(df_fft_mag.iloc[v], color='red', alpha=0.5)
#         plt.xticks([])
#         plt.yticks(fontsize=6)

#     plt.tight_layout(pad=3.0)
#     plt.show()

#     # PCA Visualization - Time Domain
#     pca_max = min([len(df.columns) for df in data_dict.values()])  

#     plt.figure()
#     for key in data_dict.keys():
#         engine_df = data_dict[key]

#         std_scaler = StandardScaler()  
#         engine_df_scaled = std_scaler.fit_transform(engine_df)

#         if pca_max > len(engine_df.columns): 
#             pca_max = len(engine_df.columns)
#         pca_range = np.arange(start=1, stop=pca_max + 1)
#         var_ratio = []
#         for num in pca_range:
#             pca = PCA(n_components=num, whiten=False)
#             pca.fit(engine_df_scaled)
#             var_ratio.append(np.sum(pca.explained_variance_ratio_))

#         plt.plot(pca_range, var_ratio, marker='o', label=f'{key}')

#     plt.grid()
#     plt.xlabel('# Components')
#     plt.ylabel('Explained Variance Ratio')
#     plt.title('Time Domain PCA')
#     plt.legend()
#     plt.show()

#     # PCA Visualization - Frequency Domain
#     plt.figure()
#     for key in data_dict.keys():
#         engine_df = data_dict[key]

#         engine_data_fft = engine_df.apply(dt.fft_df)
#         engine_data_fft_real = pd.DataFrame(np.real(engine_data_fft),
#                                             index=engine_data_fft.index,
#                                             columns=engine_data_fft.columns)
#         engine_data_fft_real.iloc[0] = engine_data_fft_real.iloc[1]
#         engine_data_fft_imag = pd.DataFrame(np.imag(engine_data_fft),
#                                             index=engine_data_fft.index,
#                                             columns=engine_data_fft.columns)
#         engine_data_fft_imag.iloc[0] = engine_data_fft_imag.iloc[1]
#         engine_df = np.sqrt(np.square(engine_data_fft_real).add(np.square(engine_data_fft_imag)))

#         std_scaler = StandardScaler()  
#         engine_df_scaled = std_scaler.fit_transform(engine_df)

#         if pca_max > len(engine_df.columns):
#             pca_max = len(engine_df.columns)
#         pca_range = np.arange(start=1, stop=pca_max + 1)
#         var_ratio = []
#         for num in pca_range:
#             pca = PCA(n_components=num)
#             pca.fit(engine_df_scaled)
#             var_ratio.append(np.sum(pca.explained_variance_ratio_))

#         plt.plot(pca_range, var_ratio, marker='o', label=f'{key}')

#     plt.grid()
#     plt.xlabel('# Components')
#     plt.ylabel('Explained Variance Ratio')
#     plt.title('Frequency Domain PCA')
#     plt.legend()
#     plt.show()

if __name__ == "__main__":
    main()
