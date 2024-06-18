import numpy as np
import pandas as pd


def fft_df(d):
    return np.fft.fft(d)

def time_to_frequency(df: pd.DataFrame):
    df_fft = df.apply(np.fft.fft)

    # Real component
    df_fft_real = pd.DataFrame(np.real(df_fft), index=df_fft.index, columns=df_fft.columns)
    df_fft_real.iloc[0] = df_fft_real.iloc[1]

    df_fft_imag = pd.DataFrame(np.imag(df_fft), index=df_fft.index, columns=df_fft.columns)
    df_fft_imag.iloc[0] = df_fft_imag.iloc[1]

    df_fft_mag = np.sqrt(np.square(df_fft_real).add(np.square(df_fft_imag)))

    # Add _fft to the column names for future concat
    col_names = [str(s) + '_fft' for s in df_fft_mag.columns]
    df_fft_mag.columns = col_names
    df_fft_real.columns = col_names
    df_fft_imag.columns = col_names

    return df_fft_mag, df_fft_real, df_fft_imag

def rolling_ranges(df: pd.DataFrame, min_window, max_window=None,
                   measures=("mean", "var"), lag=1, dropna=True):
    """
    Calculate rolling window statistics for a range of window sizes.
    Concatenate these with unique names to the input dataframe.
    :param df: pd.Dataframe
    :param min_window: starting window size
    :param max_window: ending window size
    :param measures: statistical measures to calculate (..., "min", "max")
    :param lag: Shift to add to data (t-1, ...)
    :param dropna: remove initial nan rows created by the rolling windows
    :return: Concatenated original dataframe with all the rolling windows
    """
    window_size = 0

    if max_window:
        # Typical window sizes are odd
        window_range = np.arange(start=min_window, stop=max_window + 1, step=2)
    else:
        window_range = [max_window]

    df_concat = df
    for window_size in window_range:
        df_roll = df.shift(periods=lag).rolling(window=window_size).agg(measures)

        # create discrete names based on window size
        col_names = [(s + f"_wnd{window_size}", u) for s, u in df_roll.columns]
        df_roll.columns = col_names

        df_concat = pd.concat([df_concat,
                               df_roll.set_axis(df_concat.index)], axis=1)

    if dropna:
        df_concat = df_concat.iloc[window_size:].reset_index(drop=True)

    return df_concat


def zmuv_norm_df(df):
    """
    Calculate zero-mean unit-variance shift for columns in dataframe
    Constants are transformed to 0.0
    Called with pd.apply(func)
    :param df: passed by pandas
    :return:
    """

    if df.std() == 0.:
        return df - df.mean()
    else:
        return (df - df.mean()) / df.std()


def minmax_norm_df(df: pd.DataFrame):
    return (df - df.min()) / (df.max() - df.min())
