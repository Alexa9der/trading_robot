import pandas as pd
import numpy as np
import os
from docker_connector import mt5

from data_collection.data_collector import DataCollector
from trading_robot.strategies.strategy import Strategy


class SMATuner(Strategy, DataCollector):
    """
    Class for tuning Simple Moving Average (SMA) trading strategies.

    Attributes:
        None

    Methods:
        calculate_accumulated_price_changes(data, column="Signal", buy_symbol="buy", sell_symbol="sell", 
                                            return_data=False, sl=100, tp=500):
            Calculates accumulated price changes based on buy and sell signals.
        
        collect_analyze_data(symbols, strategi, min_period=150, max_period=1000, 
                             save=False, folder_path="data/input_data", 
                             name_file="analyze_sma_data", **kwargs):
            Collects and analyzes data for SMA trading strategies.
        
        select_max_sma_per_symbol(data):
            Selects the maximum SMA per symbol based on specified criteria.

    """

    def calculate_accumulated_price_changes(self, data, column="Signal",
                                            buy_symbol="buy", sell_symbol="sell", 
                                            return_data=False, sl=100, tp=500):
        """
        Calculates accumulated price changes based on buy and sell signals.

        Args:
            data (DataFrame): Historical price data.
            column (str): Column containing signal information. Default is "Signal".
            buy_symbol (str): Symbol representing a buy signal. Default is "buy".
            sell_symbol (str): Symbol representing a sell signal. Default is "sell".
            return_data (bool): Whether to return the modified DataFrame. Default is False.
            sl (int): Stop loss value. Default is 100.
            tp (int): Take profit value. Default is 500.

        Returns:
            If return_data is True, returns the modified DataFrame.
            Otherwise, returns tuple containing sum_buy, sum_sell, median_max_profit, and median_min_profit.
        """

        df = data.copy()

        indices = data[data[column].isin([buy_symbol, sell_symbol])].index
        diff_price = data['Close'].values

        df['AccumulatedPriceChange'] = 0.0
        df['max_profit'] = 0.0
        df['min_profit'] = 0.0

        # Iterate through signal changes and calculate accumulated price changes
        for i in range(0, len(indices) - 1):
            if df.loc[indices[i], column] == sell_symbol and df.loc[indices[i + 1], column] == buy_symbol:
                price_change_sell_to_buy, max_profit, min_profit = self.__find_princes(diff_price[indices[i]:indices[i+1]],
                                                                                       df.loc[indices[i], column])

                df.loc[indices[i], 'AccumulatedPriceChange'] = price_change_sell_to_buy
                df.loc[indices[i], 'max_profit'] = max_profit
                df.loc[indices[i], 'min_profit'] = min_profit


            elif df.loc[indices[i], column] == buy_symbol and df.loc[indices[i + 1], column] == sell_symbol:
                price_change_buy_to_sell, max_profit, min_profit = self.__find_princes(diff_price[indices[i]:indices[i+1]],
                                                                                       df.loc[indices[i], column])
                df.loc[indices[i], 'AccumulatedPriceChange'] = price_change_buy_to_sell 
                df.loc[indices[i], 'max_profit'] = max_profit 
                df.loc[indices[i], 'min_profit'] = min_profit 

        sl = self.__sl_normalizer(df, sl)
        df["pozitiv_traid"] = np.where((df["min_profit"] > - sl) & (df["max_profit"] > tp), 1, 
                                       np.where(df["min_profit"] == 0, 0, -1 ))

        # Return either the calculated sums or the modified DataFrame, depending on the return_data parameter
        if return_data:
            return df
        else:
            # Calculate total accumulated price changes for buy and sell signals
            sum_sell = round( df.loc[df[column] == sell_symbol, "AccumulatedPriceChange"].sum(), 5)
            sum_buy = round( df.loc[df[column] == buy_symbol, "AccumulatedPriceChange"].sum(), 5)
            median_max_profit =  round(np.median(df.loc[df['max_profit'] != 0, "max_profit"]), 5)
            median_min_profit = round(np.median(df.loc[df['min_profit'] != 0, "min_profit"]), 5)

            return sum_buy, sum_sell, median_max_profit, median_min_profit

    def collect_analyze_data(self, symbols, timeframe = mt5.TIMEFRAME_H1,
                             min_period=150, max_period=1000, 
                             save=False, folder_path="data/input_data", 
                             name_file="analyze_sma_data", **kwargs):
        """
        Collects and analyzes data for SMA trading strategies.

        Args:
            symbols (list): List of symbols to analyze.
            strategi (function): Function for strategy calculation.
            min_period (int): Minimum period for analysis. Default is 150.
            max_period (int): Maximum period for analysis. Default is 1000.
            save (bool): Whether to save the results to a file. Default is False.
            folder_path (str): Path to the folder for saving files. Default is "data/input_data".
            name_file (str): Name of the output file. Default is "analyze_sma_data".
            **kwargs: Additional keyword arguments for calculate_accumulated_price_changes method.

        Returns:
            DataFrame: Analyzed data.

        """
        data = {}
        for symbol in symbols:
            data[symbol] = self.get_historical_data(symbol, timeframe)

        result = []
        for sma in range(min_period, max_period):
            for symbol in data.keys():
                buy, sell, median_max_profit, median_min_profit = self.calculate_accumulated_price_changes(self.sma(data[symbol], slow=sma), **kwargs)
                result.append((sma, symbol, buy, sell, median_max_profit, median_min_profit))

        df = pd.DataFrame(result, columns=["sma", "Symbol", "buy", "sell", "median_max_profit", "median_min_profit"])

        if save:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            df.to_csv(f"{folder_path}/{name_file}.csv", index=False)

        return df

    def select_max_sma_per_symbol(self, df):
        """
        Selects the maximum SMA per symbol based on specified criteria.
    
        Args:
            data (DataFrame): Dataframe containing analyzed data.
    
        Returns:
            DataFrame: Selected SMA per symbol.
    
        """
        positive_masc = ((df["buy"] > 0) & (df["sell"] > 0)) # & (df["median_max_profit"] >  np.median(df["median_max_profit"])))
        
        count = df.loc[positive_masc,'sma'].value_counts().reset_index()
        sma_lens = count.loc[count["count"] == count["count"].max(), "sma"].to_list()
        selected = df[positive_masc & df["sma"].isin(sma_lens)]
        
        
        best_indices = selected.groupby("Symbol")[["buy", "sell", "median_max_profit", "median_min_profit"]].idxmax()
        best_rows = selected.loc[best_indices["median_max_profit"], ["Symbol", "sma"]]
    
        return best_rows

    def __find_princes(self, prices, signal):
        """
        Finds prices based on the given signal.

        Args:
            prices (array-like): Prices array.
            signal (str): Signal indicating buy or sell.

        Returns:
            Tuple: exit_profit, max_profit, min_profit.

        """
        if len(prices) > 1:
            diff = (prices[0] - prices[0:]) * -1 if signal == "buy" else prices[0] - prices[0:]

            max_profit = np.max(diff)
            min_profit = np.min(diff)
            exit_profit = prices[0] - prices[-1]
        else:
            max_profit = 0
            min_profit = 0
            exit_profit = 0

        return exit_profit, max_profit, min_profit  

    def __sl_normalizer(self, data, sl):
        """
        Normalizes stop loss value.

        Args:
            data (DataFrame): DataFrame containing price data.
            sl (int): Stop loss value.

        Returns:
            float: Normalized stop loss value.

        """
        count_z = len(str(data.iloc[-1]['Close']).split('.')[-1]) - 1
        sl = sl * float('0.' +  f'{ "0" * count_z}' + '1')
        return sl
    

if __name__ == "__main":
    ...