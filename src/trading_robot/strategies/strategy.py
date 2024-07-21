import pandas as pd
import numpy as np
import talib as tl

class Strategy:

    def rebound_signal(self, data: pd.DataFrame, 
                       window_size: int = 14, 
                       bias: int = 1) -> pd.DataFrame:
        """
        Conducts analysis of price rebounds from support and resistance levels in the DataFrame.

        Args:
            data (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Updated DataFrame containing rebound analysis.
        """
        data = self.__define_line(data.copy(), window_size = window_size, bias=bias)
        # Analiza odbić cen od poziomów oporu
        data['SELL'] = (
            (data["WindowsMax"] < data["High"]) &
            (data["WindowsMax"] > data["Close"])
        ).map({True: 1, False: 0}) 
    
        # Analiza odbić cen od poziomów wsparcia
        data['Buy'] = (
            (data["WindowsMin"] > data["Low"]) &
            (data["WindowsMin"] < data["Close"])
        ).map({True: 2, False: 0})
        
        # Tworzenie sygnału na podstawie wykrytych odbić
        data["Signal"] = data['Buy'] + data['SELL']
        data["Signal"] = data["Signal"].astype(str)
        data.loc[data["Signal"] == '1', "Signal"] = "sell"
        data.loc[data["Signal"] == '2', "Signal"] = "buy"
        data.loc[data["Signal"] == '0', "Signal"] = "No signal"
        
        # Uzupełnianie pustych wartości sygnału zgodnie z poprzednimi wartościami
        data = data.drop(["SELL", "Buy", "WindowsMin", "WindowsMax" ], axis=1)
    
        # Usunięcie wierszy zawierających wartości NaN
        values = {"Signal": "no info"}
        data = data.fillna(value=values).reset_index(drop=True)
        
        # Zresetowanie indeksu ramki danych
        data.reset_index(drop=True, inplace=True)
    
        return data

    def breakdown_signal(self, data: pd.DataFrame,
                         window_size: int = 14, 
                         bias: int = 1) -> pd.DataFrame:
        """
        Conducts analysis of price breakdowns from support and resistance levels in the DataFrame.

        Args:
            data (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Updated DataFrame with breakdown analysis.
        """
        data = self.__define_line(data.copy(), window_size = window_size, bias=bias)
        # Analiza odbić cen od poziomów oporu
        data['SELL'] = (
            (data["WindowsMin"] < data["Open"]) &
            (data["WindowsMin"] > data["Close"])
        ).map({True: 1, False: 0})
    
        # Analiza odbić cen od poziomów wsparcia
        data['Buy'] = (
            (data["WindowsMax"] > data["Open"]) &
            (data["WindowsMax"] < data["Close"])
        ).map({True: 2, False: 0})
        
        # Tworzenie sygnału na podstawie wykrytych odbić
        data["Signal"] = data['Buy'] + data['SELL']
        data["Signal"] = data["Signal"].astype(str)
        data.loc[data["Signal"] == '1', "Signal"] = "sell"
        data.loc[data["Signal"] == '2', "Signal"] = "buy"
        data.loc[data["Signal"] == '0', "Signal"] = "No signal"
        
        # Uzupełnianie pustych wartości sygnału zgodnie z poprzednimi wartościami
        data = data.drop(["SELL", "Buy", "WindowsMin", "WindowsMax" ], axis=1)
    
        # Usunięcie wierszy zawierających wartości NaN
        values = {"Signal": "no info"}
        data = data.fillna(value=values).reset_index(drop=True)
        
        # Zresetowanie indeksu ramki danych
        data.reset_index(drop=True, inplace=True)

        return data

    def candle_signal(self, data, slope_period=10):
        """
        Generate signals based on candlestick analysis.
    
        Parameters:
        - data: DataFrame, input data
        - atr_period: int, period for calculating Average True Range (ATR)
        - slope_period: int, period for calculating slope
    
        Returns:
        - DataFrame, updated data with signals
        """
    
        # Copying data to avoid changes in the original data
        data = data.copy()
        
        # Determine price direction: 1 - up, -1 - down
        data['price_direction'] = np.where(data['Open'] < data['Close'], 1, -1)
    
        # Group candles by direction for subsequent counting
        data['сount_candles'] = data.groupby((data['price_direction'] != data['price_direction'].shift(1)).cumsum()).cumcount() + 1
    
        # Determine the first candle in a sequence of the same direction
        data['first_unidirect_candles'] = np.where(
            (data["сount_candles"] >= 2) & 
            (data["сount_candles"].shift(-1) == 1), 1, 0
        )
    
        # Determine two consecutive candles with a change in direction afterward
        data['two_candles_row'] = np.where(
            (data["first_unidirect_candles"] == 1) & 
            (data["first_unidirect_candles"].shift(2) == 1),
            1, 0
        )
        # Precondition for generating signals based on previous candles
        data['precondition'] = np.where(
              ( data['two_candles_row'].shift(1)  == 1) 
            & ( data['Open'] > data['Close'])
            , -1, 0
        )

        data.loc[
              ( data['two_candles_row'].shift(1)  == 1) 
            & ( data['Open'] < data['Close'])
            , "precondition" ] = 1

    
        # Generate "buy" and "No signal" signals based on conditions
        data['pre-signal'] = np.where(
              ( data['precondition'].shift(1)  == -1)
            & ( data['Open'] < data['Close'] )
            & ( data['Open'].shift(1) < data['Close'] )
            ,"buy", "No signal"
        )
    
        # Generate "sell" signal based on conditions
        data.loc[
              ( data['precondition'].shift(1)  == 1)
            & ( data['Open'] > data['Close'] )
            & ( data['Open'].shift(1) > data['Close'] ),
            "pre-signal" ] = "sell"

        # Signal
        data["Signal"] = data["pre-signal"].shift(1)
        data["Signal"].fillna ("No signal", inplace = True)
    
        # Remove temporary columns
        data.drop(['price_direction','сount_candles','first_unidirect_candles',
                   'two_candles_row','precondition', "pre-signal"], axis=1, inplace=True)
    
        return data

    def sma(self, data, slow=843, fast=5, intersection=True, count_intersection=False):
        """
        Detects crossings between two simple moving averages (SMA) in the provided dataset.
    
        Parameters:
        - data (DataFrame): The input DataFrame containing the 'Close' prices.
        - fast (int): The period for the fast SMA. Default is 5.
        - slow (int): The period for the slow SMA. Default is 843.
        - intersection (bool): Whether to detect crossings between the SMAs. Default is True.
        - count_intersection (bool): Whether to count the number of intersections. Default is True.
    
        Returns:
        - DataFrame: The input DataFrame with additional columns indicating crossings and intersection counts.
        """
    
        # Make a copy of the input data
        data = data.copy()
    
        # Calculate fast and slow SMAs
        data["fast"] = tl.SMA(data.Close, timeperiod=fast)
        data["slow"] = tl.SMA(data.Close, timeperiod=slow)
    
        # Fill missing values
        data = data.fillna(0)
    
        # If count_intersection is True, set intersection to True
        if count_intersection:
            intersection = True
    
        # Detect crossings between SMAs if intersection is True
        if intersection:
            data['Signal'] = "No signal"
            
            # Define conditions for upward and downward crossings
            conditions = [
                (data['fast'] > data['slow']) & (data['fast'].shift(1) <= data['slow'].shift(1)),  # Upward crossing
                (data['fast'] < data['slow']) & (data['fast'].shift(1) >= data['slow'].shift(1))   # Downward crossing
            ]
            
            # Mark crossings with 1 for upward crossing and -1 for downward crossing
            data.loc[conditions[0], 'Signal'] = "buy"
            data.loc[conditions[1], 'Signal'] = "sell"

            data['Signal'] = data['Signal'].shift(1).fillna("No signal")
        
        # Count the number of intersections if count_intersection is True
        if count_intersection:
            data['count_intersection'] = data.groupby((data['Signal'] != data['Signal'].shift(1)).cumsum()).cumcount() + 1
        
        return data
        
    def __define_line(self, data: pd.DataFrame,  window_size: int = 14, 
                     bias: int = 1) -> pd.DataFrame:
        """
        Adds moving maximum and minimum values to the input DataFrame based on specified parameters.

        Args:
            data (pd.DataFrame): Input DataFrame.
            window_size (int): Window size used for calculating maximum and minimum values. Default is set to 14.
            bias (int): Shift value for calculations. Default is set to 1.

        Returns:
            pd.DataFrame: Updated DataFrame containing maximum and minimum values.
        """
        data = data.copy()
        
        # Obliczenie wartości maksimum i minimum za pomocą przesuwającego się okna
        data.loc[:, 'WindowsMax'] = data['High'].rolling(window=window_size).max().shift(periods=bias)
        data.loc[:, 'WindowsMin'] = data['Low'].rolling(window=window_size).min().shift(periods=bias)
    
        # Usunięcie wierszy zawierających wartości NaN
        values = {"WindowsMax": data.loc[ : window_size + 1, 'High'].max(), 
                  "WindowsMin": data.loc[ : window_size + 1, 'Low'].min()}

        
        data = data.fillna(value=values).reset_index(drop=True)
                
        return data