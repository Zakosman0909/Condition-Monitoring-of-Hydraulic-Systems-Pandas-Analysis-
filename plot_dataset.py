import numpy as np
import pandas as pd
import glob
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import data_transforms as dt  

def file_read(file_path):
    """Reads the data file into a Pandas DataFrame."""
    try:
        return pd.read_csv(file_path, sep='\t', header=None)
    except pd.errors.EmptyDataError:
        print(f"Warning: The file {file_path} is empty and will be skipped.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is empty

def resample_data(data, target_frequency=1):
    """Resamples the data to the target frequency (in Hz)."""
    current_frequency = 100 
    resampling_factor = current_frequency // target_frequency
    return data.iloc[::resampling_factor, :]

def aggregate_features(data, window_size=10):
    """Aggregates the data by calculating rolling statistics (mean and std) over the window size."""
    if data.shape[0] < window_size:
        print(f"Data too short to aggregate with window size {window_size}.")
        return pd.DataFrame()  # Return empty DataFrame if not enough data to aggregate
    
    rolling_mean = data.rolling(window=window_size, min_periods=1).mean()
    rolling_std = data.rolling(window=window_size, min_periods=1).std()
    aggregated_data = pd.concat([rolling_mean, rolling_std], axis=1)
    aggregated_data.columns = [f'{col}_mean' for col in data.columns] + [f'{col}_std' for col in data.columns]
    return aggregated_data.dropna()

def ensure_numeric(df):
    """Ensure all columns in the DataFrame are numeric."""
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def main():
    print("Running Test on Hydraulic Systems dataset")

    # Set the directory containing the dataset files
    dataset_dir = os.path.abspath(os.path.dirname(__file__))
    print(f"Dataset directory: {dataset_dir}") 
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.txt")))

    if not files:
        sys.exit('No dataset found')

    # Initialize dictionary to store processed data
    data_dict = {}
    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f'Loading {file_name}...')

        # Skip non-data files
        if 'description' in file_name or 'documentation' in file_name or 'profile' in file_name or 'readme' in file_name:
            continue
        else:
            data = file_read(file_path)
            if data.empty:  
                print(f'{file_name} is empty after reading.')
                continue
            
            key = file_name.split('.')[0]
            print(f'{key}, Initial Shape: {data.shape}')
            data = resample_data(data, target_frequency=1)  # Resample data to 1 Hz
            if data.empty:
                print(f'{file_name} is empty after resampling.')
                continue
            
            data = aggregate_features(data)  # Aggregate features with rolling statistics
            if data.empty:
                print(f'{file_name} is empty after aggregation.')
                continue
            
            data_dict[key] = data
            print(f'{key}, Final Shape: {data.shape}')

    if not data_dict:
        sys.exit('No valid data was loaded after processing.')

    # Concatenate all processed data into a single DataFrame
    X = pd.concat([data_dict[key] for key in data_dict.keys()], axis=1)

    if X.empty:
        sys.exit('The concatenated DataFrame X is empty.')

    # Load the target variable data
    ans_df = pd.read_csv(os.path.join(dataset_dir, 'profile.txt'), sep='\t', header=None,
                         names=['Cooler', 'Valve', 'Pump', 'Accumulator', 'Stable'])

    # Adjust the target variable length to match the aggregated data
    y = ans_df['Cooler'].iloc[X.index].values

    # Encode the labels to sequential integers starting from 0
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')

    # Ensure all data is numeric
    X_train = ensure_numeric(X_train)
    X_test = ensure_numeric(X_test)

    # Convert DataFrame to NumPy array before passing to XGBoost
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()

    # Define the pipeline with preprocessing, feature selection, and model
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(score_func=mutual_info_regression, k=10)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'feature_selection__k': [10, 20, 30],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20]
    }

    # Initialize GridSearchCV with the pipeline and parameter grid
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best pipeline and parameters
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predict on the test set using the best pipeline
    y_pred = best_pipeline.predict(X_test)

    y_pred_original = label_encoder.inverse_transform(y_pred)
    y_test_original = label_encoder.inverse_transform(y_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test_original, y_pred_original)
    precision = precision_score(y_test_original, y_pred_original, average='macro')
    recall = recall_score(y_test_original, y_pred_original, average='macro')
    f1 = f1_score(y_test_original, y_pred_original, average='macro')
    report = classification_report(y_test_original, y_pred_original)

    print(f'Best pipeline parameters: {best_params}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print('Classification Report:')
    print(report)

    # Train and evaluate XGBoost model
    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train_np, y_train)
    y_pred_xgb = xgb_model.predict(X_test_np)

    
    y_pred_xgb_original = label_encoder.inverse_transform(y_pred_xgb)

    accuracy_xgb = accuracy_score(y_test_original, y_pred_xgb_original)
    report_xgb = classification_report(y_test_original, y_pred_xgb_original, zero_division=0)

    print(f'XGBoost Accuracy: {accuracy_xgb}')
    print('XGBoost Classification Report:')
    print(report_xgb)

    # Correlation matrix for selected features
    X_train_transformed = best_pipeline.named_steps['feature_selection'].transform(X_train_np)
    selected_features = X.columns[best_pipeline.named_steps['feature_selection'].get_support()]
    X_selected = pd.DataFrame(X_train_transformed, columns=selected_features)

    corr_matrix = X_selected.corr()

    plt.figure(figsize=(15, 12))  
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 8}, linewidths=.5) 
    plt.title('Heatmap of Cooler Condition', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)  
    plt.yticks(rotation=0, fontsize=10)  
    plt.show()

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
