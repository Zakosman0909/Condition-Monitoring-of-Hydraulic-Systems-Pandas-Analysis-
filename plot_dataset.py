import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import data_transforms as dt


def file_read(file_path):
    return pd.read_csv(file_path, sep='\t', header=None)


def main():
    print("Running Test on Hydraulic Systems dataset")
    
    dataset_dir = os.path.abspath(os.path.dirname(__file__))
    print(f"Dataset directory: {dataset_dir}") 
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.txt")))

    if not files:
        sys.exit('No dataset found')

    data_dict = {}
    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f'Loading {file_name}...')

        if 'description' in file_name or 'documentation' in file_name or 'profile' in file_name:
            continue
        else:
            data = file_read(file_path)
            key = file_name.split('.')[0]
            print(f'{key}, Shape: {data.shape}')
            data_dict[key] = data

            print('Dataframe Info/Description')
            print(data.info())
            print(data.describe())

    ans_df = pd.read_csv(os.path.join(dataset_dir, 'profile.txt'), sep='\t', header=None,
                         names=['Cooler', 'Valve', 'Pump', 'Accumulator', 'Stable'])

    unstables = np.where(ans_df['Stable'].values == 0)[0]
    stables = np.where(ans_df['Stable'].values == 1)[0]

    num_keys = len(data_dict.keys())
    cols = 4
    rows = int(np.ceil(num_keys / cols))

    num = 250
    plt.figure(figsize=(20, rows * 4))
    for i, key in enumerate(data_dict.keys()):
        plt.subplot(rows, cols, i + 1)
        plt.title(f'{key}', fontsize=6)
        for v in stables[:num - 1]:
            plt.plot(data_dict[key].iloc[v], color='green')
        plt.plot(data_dict[key].iloc[stables[num]], label='stable', color='green')
        for v in unstables[:num]:
            plt.plot(data_dict[key].iloc[v], color='red', alpha=0.5)
        plt.plot(data_dict[key].iloc[unstables[num]], label='unstable', color='red', alpha=0.5)
        plt.legend(loc='upper right', fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

    plt.tight_layout(pad=3.0)
    plt.show()

    # Apply FFT transformation
    plt.figure(figsize=(20, rows * 4))
    for i, key in enumerate(data_dict.keys()):
        plt.subplot(rows, cols, i + 1)
        plt.title(f'{key} FFT', fontsize=6)
        data_fft = data_dict[key].apply(dt.fft_df)
        data_fft_real = pd.DataFrame(np.real(data_fft), index=data_fft.index, columns=data_fft.columns)
        data_fft_real.iloc[0] = data_fft_real.iloc[1]
        data_fft_imag = pd.DataFrame(np.imag(data_fft), index=data_fft.index, columns=data_fft.columns)
        data_fft_imag.iloc[0] = data_fft_imag.iloc[1]
        data_fft_mag = np.sqrt(np.square(data_fft_real).add(np.square(data_fft_imag)))
        for v in stables[:num - 1]:
            plt.plot(data_fft_mag.iloc[v], color='blue')
        plt.plot(data_fft_mag.iloc[stables[num]], label='stable', color='blue')
        for v in unstables[:num]:
            plt.plot(data_fft_mag.iloc[v], color='orange', alpha=0.5)
        plt.plot(data_fft_mag.iloc[unstables[num]], label='unstable', color='orange', alpha=0.5)
        plt.legend(loc='upper right', fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

    plt.tight_layout(pad=3.0)
    plt.show()

    # Apply PCA transformation on the time domain
    pca_max = min([len(data_dict[key].columns) for key in data_dict.keys()])  # Adjust pca_max to the minimum number of columns
    plt.figure()
    for key in data_dict.keys():
        data = data_dict[key]
        std_scaler = StandardScaler()
        data_scaled = std_scaler.fit_transform(data)
        pca_range = np.arange(start=1, stop=pca_max + 1)
        var_ratio = []
        for num in pca_range:
            pca = PCA(n_components=num)
            pca.fit(data_scaled)
            var_ratio.append(np.sum(pca.explained_variance_ratio_))
        plt.plot(pca_range, var_ratio, marker='o', label=f'{key}')
    
    plt.grid()
    plt.xlabel('# Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Time Domain PCA')
    plt.legend(loc='lower right')
    plt.show()

    # Apply PCA transformation on the frequency domain
    plt.figure()
    for key in data_dict.keys():
        data = data_dict[key]
        data_fft = data.apply(dt.fft_df)
        data_fft_real = pd.DataFrame(np.real(data_fft), index=data_fft.index, columns=data_fft.columns)
        data_fft_real.iloc[0] = data_fft_real.iloc[1]
        data_fft_imag = pd.DataFrame(np.imag(data_fft), index=data_fft.index, columns=data_fft.columns)
        data_fft_imag.iloc[0] = data_fft_imag.iloc[1]
        data_fft_mag = np.sqrt(np.square(data_fft_real).add(np.square(data_fft_imag)))
        std_scaler = StandardScaler()
        data_scaled = std_scaler.fit_transform(data_fft_mag)
        pca_range = np.arange(start=1, stop=pca_max + 1)
        var_ratio = []
        for num in pca_range:
            pca = PCA(n_components=num)
            pca.fit(data_scaled)
            var_ratio.append(np.sum(pca.explained_variance_ratio_))
        plt.plot(pca_range, var_ratio, marker='o', label=f'{key}')
    
    plt.grid()
    plt.xlabel('# Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Frequency Domain PCA')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()
