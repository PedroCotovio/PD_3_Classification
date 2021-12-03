###
### Pre-Processing Functions
###

import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas_profiling
from scipy import nanmedian as median, nanmean as mean, nanstd as std
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

def profile(df, title='EDA', file='Report_df'):
    
    """
    Pandas Profiling
    """

    profile = pandas_profiling.ProfileReport(df, title='Basic EDA', correlations={"cramers": False})
    file = str(file)+'.html'
    profile.to_file(file)
    
    return profile

def std_outlier_detection(data):
    """
    Outlier Detection based on standart deviation
    """
    
    data_mean = mean(data)
    anomaly_cut_off = std(data) * 3
    
    lower_limit  = data_mean - anomaly_cut_off 
    upper_limit = data_mean + anomaly_cut_off

    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            return True
    return False

def DBSCAN_outlier_detection(data):
    """
    Outlier detection through DBSCAN noise
    """
    data = data.reshape(-1,1)
    model = DBSCAN(min_samples = 2, eps = 3)
    clusters = model.fit_predict(data)
    return (list(clusters).count(-1) > 0)

def outlier_detection(data):
    """
    Outlier Detection ensemble
    """
    data = data.copy()
    data.dropna(inplace=True)
    data = np.array(data)
    
    return DBSCAN_outlier_detection(data) and std_outlier_detection(data)

def feature_transform(df, threshold=0.8, bin_threshold=0.3, rem_low_var=True):
    
    """
    Pipeline to Preprocess Dataset
    """
    
    _len = df.shape[1]
    df = pd.DataFrame(df)
    df = df.copy()
    missing_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    zeros_percent = ((df == 0).astype(int).sum()/(df == 0).astype(int).count()).sort_values(ascending=False)
    cols_drop = []
    cols_cat = []

    for col in df.columns:
        if zeros_percent[col] >= threshold or missing_percent[col] >= threshold:
            cols_drop.append(col)
            
        elif zeros_percent[col] >= bin_threshold:
            df[col][df[col] != 0] = 1
        
        elif missing_percent[col] >= bin_threshold:
            
            df[col][~df[col].isnull()] = 1
            df[col][df[col].isnull()] = 0
            
        elif missing_percent[col] > 0.0:
            try:
                if outlier_detection(df[col]):
                    df[col][df[col].isnull()] = median(df[col])

                else:
                    df[col][df[col].isnull()] = mean(df[col])
            except ValueError:
                cols_cat.append(col)
                df[col][df[col].isnull()] = 'None'
        else:
            try: int(df[col][0])
            except:
                cols_cat.append(col)
            
    df.drop(cols_drop, axis=1, inplace=True)
    
    cols_drop2 = []
    
    if rem_low_var is True:
        for col in df.columns:
            if list((df[col].value_counts()/df[col].value_counts().sum()).sort_values(ascending=False))[0] > threshold:
                cols_drop2.append(col)
        
        df.drop(cols_drop2, axis=1, inplace=True)
        
        cols_cat = [item for item in cols_cat if item not in cols_drop2]
        
    _del = pd.DataFrame(cols_drop+cols_drop2, columns=['Columns'])
        
    print('{} Deleted Features of {}'.format(_del.shape[0], _len))
    
    return df, _del, cols_cat

def generate_representation(X, train=True, path='', names=['encoder', 'autoencoder'], encoding_dim=20, batch=None, epochs=100, split=0.05, verbose=0):
    
    """
    Generate representations for the Features Matrix by applying an Autoencoder Neural Network
    """
    
    X = np.array(X)
    names = list(names)
    
    encoder_path = os.path.join(names[0], str(path))
    autoencoder_path = os.path.join(names[1], str(path))
    
    if train is True:
        
        n_columns = X.shape[1]
        encoding_dim = int(encoding_dim)
        epochs = int(epochs)
        split = float(split)

        if not batch:
            batch = int(round(X.shape[0]/100, 0))

        input_df = Input(shape=(n_columns,))
        encoded = Dense(encoding_dim, activation='relu')(input_df)
        decoded = Dense(n_columns, activation='sigmoid')(encoded)

        # Full Model (decoder)
        autoencoder = Model(input_df, decoded)

        # Generates representations (encoder)
        encoder = Model(input_df, encoded)

        autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

        autoencoder.fit(X, X,
                        epochs=epochs,
                        batch_size=batch,
                        shuffle=True,
                        validation_split=split,
                        verbose=verbose)
        
        autoencoder.save(autoencoder_path)
        encoder.save(encoder_path)
        
    else:
        
        autoencoder = load_model(autoencoder_path)
        encoder = load_model(encoder_path)

    loss = autoencoder.evaluate(X, X, verbose=verbose)
    print('Evaluation loss: ', round(loss, 4), ' (MSE)')
    
    X_encoded = encoder.predict(X)
    print ('{} Representations Generated from {} Features'.format(X_encoded.shape[1], X.shape[1]))
    
    return X_encoded

def ohe(data):
    """
    One Hot Encoder
    """
    model = OneHotEncoder(sparse=False)
    X = data
    X = model.fit_transform(X)
    return X

def encoder(data, categoric):
    """
    Main Encoder
    """
    data = data.copy()
    
    #One Hot Encoding
    X = ohe(data[categoric].values)
    print('From {} categorical features, {} binary features generated'.format(len(categoric), X.shape[1]))
    
    # Concatenate
    data.drop(categoric, axis=1, inplace=True)
    X = pd.DataFrame(X)
    data = pd.concat([data, X], axis=1)
    
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(data.values)
    
    print('Total features {}'.format(data.shape[1]))
    return X_scale


# Bar Chart
def make_bar_chart(x):
    """
    Return Bar Chart

    :param x: array, dataset to plot

    """
    plt.figure(figsize=(14, 6));
    g = sns.countplot(x)
    ax = g.axes
    for p in ax.patches:
        ax.annotate(f"{p.get_height() * 100 / x.shape[0]:.2f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
                    textcoords='offset points')
