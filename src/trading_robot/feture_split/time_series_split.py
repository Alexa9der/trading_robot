import pandas as pd
import numpy as np

from trading_robot.data_collection.data_collector import DataCollector
from trading_robot.utils.logger import log_message

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
        
        log_message(f"Starting train-test split with test_size={test_size} for data with {len(data)} rows.")

        try:
            split = int((1 - test_size) * len(data))
            log_message(f"Data will be split at index {split}. Training data will have {split} rows, test data will have {len(data) - split} rows.")
            
            X = data.drop(columns=[y_col])
            y = data[y_col]
            
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            log_message("Train-test split completed successfully.")
            
            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            log_message(f"Error occurred during train-test split: {e}")
            return None, None, None, None

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
        log_message("Starting to create sequences.")

        try:
            log_message(f"Creating sequences with sequence_length={sequence_length} and target column='{y_col}'.")

            sequences = []
            labels = []

            X = data.drop(columns=[y_col]).values
            y = data[y_col].values

            for i in range(len(data) - sequence_length):
                sequences.append(X[i:i + sequence_length])
                labels.append(y[i + sequence_length])

            sequences = np.array(sequences)
            labels = np.array(labels)

            log_message(f"Created {len(sequences)} sequences with length {sequence_length}.")
            return sequences, labels
        
        except Exception as e:
            log_message(f"Error occurred while creating sequences: {e}")
            return np.array([]), np.array([])
    


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

