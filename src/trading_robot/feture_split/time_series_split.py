import pandas as pd
import numpy as np

from trading_robot.data_collection.data_collector import DataCollector

class TimeSeriesSplits:
    """
    Class for preparing data and dividing it into training and test samples.

    Methods:
    Train_test_split (x, y, test_size):
    Divides data into training and test samples.
    Create_sequences (Data, Sequence_LENGTH):
    Creates data sequences and appropriate tags for prediction tasks.
    """
       
    def train_test_split(self, data: pd.DataFrame, y_col: str, test_size: float = 0.2):
        """
        Separates data into training and test samples, maintaining a temporary sequence.

        Options:
        Data: PD.Dataframe - data containing signs and target variable.
        y_col: str - the name of the column with the target variable.
        Test_Size: Float - the share of data that will be used for test sample (default 0.2).

        Returns:
        X_train, y_train, X_test, y_test - divided data.
        """
        split = int((1 - test_size) * len(data))
        
        X = data.drop(columns=[y_col])
        y = data[y_col]
        
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        return X_train, y_train, X_test, y_test

    def create_sequences(self, data: pd.DataFrame, y_col: str = "Close", sequence_length: int = 32):
        """
        Creates data sequences and tags from Dataframe.

        Options:
        - Data: Pd.Dataframe - data containing signs and target variable.
        - y_col: str - the name of the column with the target variable.
        - Sequence_LENGTH: int - the length of the sequence.

        Returns:
        - sequences: np.array - an array of sequences of signs.
        - Labels: NP.array - an array of marks.
        """

        sequences = []
        labels = []

        X = data.drop(columns=[y_col]).values
        y = data[y_col].values

        for i in range(len(data) - sequence_length):
            sequences.append(X[i:i + sequence_length])
            labels.append(y[i + sequence_length])
        
        return np.array(sequences), np.array(labels)
    


if __name__ == "__main__":

    data_col = DataCollector()
    data = data_col.get_historical_data(symbol="EURUSD")

    tss = TimeSeriesSplits()

    X_train, y_train, X_test, y_test = tss.train_test_split(data, "Close")
    print("data shape: ", data.shape)
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

