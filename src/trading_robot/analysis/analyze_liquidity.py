import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# from connectors.docker_connector  import mt5
from connectors.data_collector import DataCollector, mt5

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz




class AnalyzeLiquidity(DataCollector):
    """
    Description:
    This class performs market liquidity analysis and selects the best symbols for trading based on specified criteria.

    Attributes:
    - timeframe (int): The timeframe for historical data. Default is mt5.TIMEFRAME_M1.
    - wick_more_on (int): The multiplier for determining the threshold for long wicks. Default is 2.
    """

    def __init__(self, timeframe=mt5.TIMEFRAME_D1, wick_more_on=2, 
                 period=250, permissible_clearance_gaps = 10):
        """
        Description: Initializes the AnalyzeLiquidity object with specified parameters.
        Parameters:
            - timeframe (int): The timeframe for historical data. Default is mt5.TIMEFRAME_M1.
            - wick_more_on (int): The multiplier for determining the threshold for long wicks. Default is 2.
        """
        super().__init__()
        self.connect = DataCollector()
        self.timeframe = timeframe
        self.wick_more_on = wick_more_on 
        self.period = period
        self.permissible_clearance_gaps = permissible_clearance_gaps

    def market_analysis(self, symbol, timeframe=mt5.TIMEFRAME_M5):
        """
        Description: Performs market analysis for a specific symbol.
        Parameters:
            - symbol (str): The symbol for which market analysis is to be performed.
        Returns:
            - dict: A dictionary containing market analysis metrics.
        """

        data = self.get_historical_data(symbol=symbol, timeframe=self.timeframe, count=self.period)

        decimal_length = len( str(data["Close"][1]).split(".")[-1])
        permissible_clearance = self.permissible_clearance_gaps * float(f"0.{decimal_length * '0'}1")

        data["Volume"] = abs(data["High"] - data["Low"])
        data.loc[ (data["Open"] + permissible_clearance <= data["Close"].shift(1)) 
                & (data["Close"].shift(1) >= data["Open"] - permissible_clearance), "Gep"] = 1
        
        data.loc[abs(data["Open"] - data["Close"]) * self.wick_more_on < abs(data["Open"] - data["High"]), "wick_buy"] = 1
        data.loc[abs(data["Open"] - data["Close"]) * self.wick_more_on < abs(data["Close"] - data["Low"]), "wick_sell"] = 1

        data['Daily_Return'] = (data['Close'] - data['Open']) / data['Open']

        
        
        return {
            "spread": self.__spread(symbol, timeframe) ,
            "geps": data["Gep"].sum() / len(data),
            "mean_volume": data["Volume"].sum() / len(data),
            "wick_buy": data["wick_buy"].sum() / len(data),
            "wick_sell": data["wick_sell"].sum() / len(data), 
            "std": round( np.sqrt(np.var(data['Daily_Return'])) * np.sqrt(len(data)), 5),
        }

    def choose_best_symbols(self, symbols: list, return_info="symbols",
                            timeframe=mt5.TIMEFRAME_M5,
                            i_spread="25%",i_geps="75%",i_wick="75%",
                            i_mean_volume="75%",i_std="50%", f_spread=True, 
                            f_geps=True, f_wick=True, f_mean_volume=True, f_std=False):
        """
        Choose the best symbols based on various criteria with customizable percentiles.
    
        Parameters:
        - symbols (list): List of symbols to analyze.
        - return_info (str): Determines what information to return ("symbols" or "data", default is "symbols").
        - i_spread (str or float): Percentile for spread filtering (default is "25%").
        - i_geps (str or float): Percentile for geps filtering (default is "75%").
        - i_wick (str or float): Percentile for wick filtering (default is "50%").
        - i_mean_volume (str or float): Percentile for mean volume filtering (default is "50%").
        - i_std (str or float): Percentile for standard deviation filtering (default is "50%").
        - f_spread (bool): Whether to filter by spread (default is True).
        - f_geps (bool): Whether to filter by geps (default is True).
        - f_wick (bool): Whether to filter by wick (default is True).
        - f_mean_volume (bool): Whether to filter by mean volume (default is True).
        - f_std (bool): Whether to filter by standard deviation (default is False).
    
        Returns:
        - list or DataFrame: List of symbols (if return_info is "symbols") or DataFrame with filtered data (if return_info is "data").
    
        This method analyzes the specified symbols based on various criteria and filters them accordingly.
        It allows customizing the percentiles used for filtering each criterion. 
        It returns either a list of symbols that meet the criteria (if return_info is "symbols") or a DataFrame
        containing the filtered data (if return_info is "data").
        """
        analyze_symbols = {}
        percentiles = [.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95]
        
        # Analyze symbols and store results in a DataFrame
        for symbol in symbols:
            analyze_symbols[symbol] = self.market_analysis(symbol=symbol, timeframe=timeframe)
        data_analyze_symbols = pd.DataFrame(analyze_symbols).T

        # Filter symbols based on specified criteria
        if f_spread:
            spread = data_analyze_symbols.describe(percentiles=percentiles).loc[i_spread, "spread"]
            data_analyze_symbols = data_analyze_symbols[data_analyze_symbols["spread"] <= spread]
    
        if f_geps:
            geps = data_analyze_symbols.describe(percentiles=percentiles).loc[i_geps, "geps"]
            data_analyze_symbols = data_analyze_symbols[data_analyze_symbols["geps"] <= geps]
    
        if f_wick:
            wick = data_analyze_symbols.describe(percentiles=percentiles).loc[i_wick, ["wick_buy", "wick_sell"]].to_numpy()
            data_analyze_symbols = data_analyze_symbols[(data_analyze_symbols[["wick_buy", "wick_sell"]] <= wick).any(axis=1)]
    
        if f_mean_volume:
            mean_volume = data_analyze_symbols.describe(percentiles=percentiles).loc[i_mean_volume, "mean_volume"]
            data_analyze_symbols = data_analyze_symbols[data_analyze_symbols["mean_volume"] >= mean_volume]
    
        if f_std:
            std = data_analyze_symbols.describe(percentiles=percentiles).loc[i_std, "std"]
            data_analyze_symbols = data_analyze_symbols[data_analyze_symbols["std"] <= std]
    
        # Return either symbols or filtered data based on return_info parameter
        if return_info == "symbols":
            return data_analyze_symbols.index.to_list()
        elif return_info == "data":
            return data_analyze_symbols

    def __spread(self, symbol, timeframe):

        try:
     
            timezone = pytz.timezone("Etc/UTC")
            utc_to = datetime.now(tz=timezone)
            utc_from = utc_to - timedelta(days=3)
            
            self.connect._connect_to_mt5()
            ticks = mt5.copy_rates_range(symbol=symbol, date_from=utc_from, 
                                         date_to=utc_to, timeframe=timeframe)

            ticks_frame = pd.DataFrame(ticks)
            ticks_frame["spread"] = ticks_frame.ask - ticks_frame.bid

            return ticks_frame["spread"].mean()

        except NameError as e:
            print(f"NameError Error while calculating spread: {e}")

        except Exception as e:
            print(f"Error while calculating spread: {e}")

        finally:
            mt5.shutdown()



if __name__ == "__main__":

    analyze_liq = AnalyzeLiquidity()
    analize_result = analyze_liq.market_analysis(symbol="EURUSD")
    print(analize_result)
    

