import pandas as pd
import numpy as np
import wfdb
import ast

def min_max_scaling(data):
    min_val = np.min(data, axis=1, keepdims=True)
    max_val = np.max(data, axis=1, keepdims=True)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data
class ECGDataLoader:
    def __init__(self, path, sampling_rate=100):
        self.path = path
        self.sampling_rate = sampling_rate

    def load_raw_data(self, df):
        if self.sampling_rate == 100:
            data = [wfdb.rdsamp(self.path + f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(self.path + f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def load_annotation_data(self, csv_path):
        Y = pd.read_csv(csv_path, index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        return Y

    def load_aggregation_data(self, csv_path):
        agg_df = pd.read_csv(csv_path, index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        return agg_df

    def aggregate_diagnostic(self, y_dic, agg_df):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def split_train_test_data(self, X, Y, test_fold):
        # X_train = X[np.where(Y.strat_fold != test_fold)]
        # y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
        # X_test = X[np.where(Y.strat_fold == test_fold)]
        # y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
        X_train = X[np.where(Y.strat_fold != test_fold)][:, :, 0]
        y_train = X[np.where(Y.strat_fold != test_fold)][:, :, 1:]
        X_test = X[np.where(Y.strat_fold == test_fold)][:, :, 0]
        y_test = X[np.where(Y.strat_fold == test_fold)][:, :, 1:]
        return X_train, y_train, X_test, y_test

    def preprocess_data(self, test_fold=10):
        # Load and convert annotation data
        Y = self.load_annotation_data(self.path + 'ptbxl_database.csv')

        # Load raw signal data
        X = self.load_raw_data(Y)
        print('shape: ', X.shape)
        # Load scp_statements.csv for diagnostic aggregation
        agg_df = self.load_aggregation_data(self.path + 'scp_statements.csv')

        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: self.aggregate_diagnostic(x, agg_df))

        # Split data into train and test
        X_train, y_train, X_test, y_test = self.split_train_test_data(X, Y, test_fold)

        return X_train, y_train, X_test, y_test