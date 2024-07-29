import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from connectors.data_collector import DataCollector, mt5

import talib as tl
import pandas as pd 
import numpy as np


class Indicators:
    """
    A class to preprocess stock data by applying various technical indicators and mathematical transformations.
    """
    def __init__ (self, df: pd.DataFrame= None, periods: list[int] = None):
        """
        Initializes the Preprocessing_stock_data class.

        Args:
            data (pd.DataFrame): Input data to be processed.
            periods (list, optional): List of periods for calculations. Defaults to None.

        This function initializes the Preprocessing_stock_data class and sets the dataframe, columns, and periods to be used for processing.
        """
        if df is not None:
            self.df = df.copy()
        self.periods = periods if periods is None else [23,115,220]

    def price_changes(self, df=None):
        """
        Calculate various price change characteristics for a given DataFrame.
    
        Args:
            df (pd.DataFrame, optional): Input DataFrame containing price-related columns.
                If not provided, the DataFrame passed during class initialization will be used.
    
        Returns:
            pd.DataFrame: Updated DataFrame with additional columns for price changes.
    
        This function calculates several price change characteristics based on the input DataFrame. The following
        columns are added to the DataFrame:
    
        - 'PriceRange': The difference between the High and Low prices, representing the price range or volatility.
        - 'MaxPositivePriceChange': The difference between the High price and the Open price, indicating the maximum
          positive price change during the period.
        - 'MaxNegativePriceChange': The difference between the Open price and the Low price, indicating the maximum
          negative price change during the period.
        - 'PriceChange': The change in closing price compared to the previous period.
    
        These additional columns provide insights into the volatility, maximum positive and negative price changes,
        and overall price dynamics over time.
        """
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
    
        # Calculate the difference between High and Low prices (Volatility)
        df['PriceRange'] = df['High'] - df['Low']
    
        # Calculate the difference between High and Open prices (Max Positive Price Change)
        df['MaxPositivePriceChange'] = df['High'] - df['Open']
    
        # Calculate the difference between Open and Low prices (Max Negative Price Change)
        df['MaxNegativePriceChange'] = df['Open'] - df['Low']
    
        # Calculate the change in closing price compared to the previous period
        df['PriceChange'] = (df['Close'] - df['Close'].shift(1)).fillna(0)
    
        return df

    def indicators_pattern_recognition_functions(self, df=None):
        """
        Adds pattern recognition indicators to the dataframe.
    
        Args:
            df (pd.DataFrame, optional): Input DataFrame containing price-related columns.
                If not provided, the DataFrame passed during class initialization will be used.
    
        Returns:
            pd.DataFrame: Dataframe with added pattern recognition indicators.
    
        This function creates copies of the input data and adds pattern recognition indicators to the dataframe.
        """
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
        
        df["CDL2CROWS"] = tl.CDL2CROWS(df["Open"], df["High"], df["Low"], df["Close"]) # type: ignore
        df["CDL3BLACKCROWS"] = tl.CDL3BLACKCROWS(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDL3INSIDE"] = tl.CDL3INSIDE(df["Open"], df["High"], df["Low"], df["Close"]) # type: ignore
        df["CDL3LINESTRIKE"] = tl.CDL3LINESTRIKE(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDL3OUTSIDE"] = tl.CDL3OUTSIDE(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDL3STARSINSOUTH"] = tl.CDL3STARSINSOUTH(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDL3WHITESOLDIERS"] = tl.CDL3WHITESOLDIERS(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLABANDONEDBABY"] = tl.CDLABANDONEDBABY(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLADVANCEBLOCK"] = tl.CDLADVANCEBLOCK(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLBELTHOLD"] = tl.CDLBELTHOLD(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLBREAKAWAY"] = tl.CDLBREAKAWAY(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLCLOSINGMARUBOZU"] = tl.CDLCLOSINGMARUBOZU(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLCONCEALBABYSWALL"] = tl.CDLCONCEALBABYSWALL(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLCOUNTERATTACK"] = tl.CDLCOUNTERATTACK(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLDARKCLOUDCOVER"] = tl.CDLDARKCLOUDCOVER(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLDOJI"] = tl.CDLDOJI(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLDOJISTAR"] = tl.CDLDOJISTAR(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLDRAGONFLYDOJI"] = tl.CDLDRAGONFLYDOJI(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLENGULFING"] = tl.CDLENGULFING(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLEVENINGDOJISTAR"] = tl.CDLEVENINGDOJISTAR(df["Open"], df["High"], df["Low"], df["Close"], penetration=0)
        df["CDLEVENINGSTAR"] = tl.CDLEVENINGSTAR(df["Open"], df["High"], df["Low"], df["Close"], penetration=0)
        df["CDLGAPSIDESIDEWHITE"] = tl.CDLGAPSIDESIDEWHITE(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLGRAVESTONEDOJI"] = tl.CDLGRAVESTONEDOJI(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLHAMMER"] = tl.CDLHAMMER(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLHANGINGMAN"] = tl.CDLHANGINGMAN(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLHARAMI"] = tl.CDLHARAMI(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLHARAMICROSS"] = tl.CDLHARAMICROSS(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLHIGHWAVE"] = tl.CDLHIGHWAVE(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLHIKKAKE"] = tl.CDLHIKKAKE(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLHIKKAKEMOD"] = tl.CDLHIKKAKEMOD(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLHOMINGPIGEON"] = tl.CDLHOMINGPIGEON(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLIDENTICAL3CROWS"] = tl.CDLIDENTICAL3CROWS(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLINNECK"] = tl.CDLINNECK(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLINVERTEDHAMMER"] = tl.CDLINVERTEDHAMMER(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLKICKING"] = tl.CDLKICKING(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLKICKINGBYLENGTH"] = tl.CDLKICKINGBYLENGTH(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLLADDERBOTTOM"] = tl.CDLLADDERBOTTOM(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLLONGLEGGEDDOJI"] = tl.CDLLONGLEGGEDDOJI(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLLONGLINE"] = tl.CDLLONGLINE(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLMARUBOZU"] = tl.CDLMARUBOZU(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLMATCHINGLOW"] = tl.CDLMATCHINGLOW(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLMATHOLD"] = tl.CDLMATHOLD(df["Open"], df["High"], df["Low"], df["Close"], penetration=0)
        df["CDLMORNINGDOJISTAR"] = tl.CDLMORNINGDOJISTAR(df["Open"], df["High"], df["Low"], df["Close"], penetration=0)
        df["CDLMORNINGSTAR"] = tl.CDLMORNINGSTAR(df["Open"], df["High"], df["Low"], df["Close"], penetration=0)
        df["CDLONNECK"] = tl.CDLONNECK(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLPIERCING"] = tl.CDLPIERCING(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLRICKSHAWMAN"] = tl.CDLRICKSHAWMAN(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLRISEFALL3METHODS"] = tl.CDLRISEFALL3METHODS(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLSEPARATINGLINES"] = tl.CDLSEPARATINGLINES(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLSHOOTINGSTAR"] = tl.CDLSHOOTINGSTAR(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLSHORTLINE"] = tl.CDLSHORTLINE(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLSPINNINGTOP"] = tl.CDLSPINNINGTOP(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLSTALLEDPATTERN"] = tl.CDLSTALLEDPATTERN(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLSTICKSANDWICH"] = tl.CDLSTICKSANDWICH(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLTAKURI"] = tl.CDLTAKURI(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLTASUKIGAP"] = tl.CDLTASUKIGAP(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLTHRUSTING"] = tl.CDLTHRUSTING(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLTRISTAR"] = tl.CDLTRISTAR(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLUNIQUE3RIVER"] = tl.CDLUNIQUE3RIVER(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLUPSIDEGAP2CROWS"] = tl.CDLUPSIDEGAP2CROWS(df["Open"], df["High"], df["Low"], df["Close"])
        df["CDLXSIDEGAP3METHODS"] = tl.CDLXSIDEGAP3METHODS(df["Open"], df["High"], df["Low"], df["Close"])

        return df.fillna(0)

    def calculate_overlap_studies(self, df=None, periods=[23,115,220]):
        """
        Calculates various overlap studies for the input data.
    
        Args:
            df (pd.DataFrame, optional): Input DataFrame containing price-related columns.
                If not provided, the DataFrame passed during class initialization will be used.
            periods (list[int], optional): List of periods for overlap studies calculations.
                If not provided, the periods specified during class initialization will be used.
    
        Returns:
            pd.DataFrame: DataFrame with calculated overlap studies.
    
        This function calculates various overlap studies based on the provided periods.
        """
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
        if periods is None:
            periods = self.periods
        
        for i in periods:
            df["DEMA"+str(i)] = tl.DEMA(df["Close"], timeperiod=i)
            df["EMA"+str(i)] = tl.EMA(df["Close"], timeperiod=i)
            df["KAMA"+str(i)] = tl.KAMA(df["Close"], timeperiod=i)
            df["MIDPOINT"+str(i)] = tl.MIDPOINT(df["Close"], timeperiod=i)
            df["SMA"+str(i)] = tl.SMA(df["Close"], timeperiod=i)
            df["TRIMA"+str(i)] = tl.TRIMA(df["Close"], timeperiod=i)
            df["WMA"+str(i)] = tl.WMA(df["Close"], timeperiod=i)
            df["T3"+str(i)] = tl.T3(df["Close"], timeperiod=i, vfactor=0)
            df["TEMA"+str(i)] = tl.TEMA(df["Close"], timeperiod=i)
            df["MA"+str(i)] = tl.MA(df["Close"], timeperiod=i, matype=0)
            df["HT_TRENDLINE"+str(i)] = tl.HT_TRENDLINE(df["Close"])
        
        return df.fillna(0)

    def math_transform_functions(self, df=None):
        """
        Applies various mathematical transformation functions to the input data.
    
        Args:
            df (pd.DataFrame, optional): Input DataFrame containing price-related columns.
                If not provided, the DataFrame passed during class initialization will be used.
    
        Returns:
            pd.DataFrame: DataFrame with applied mathematical transformation functions.
    
        This function applies various mathematical transformation functions to the 'close' column.
        """
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
    
        with np.errstate(invalid='ignore'):
            df["ACOS"] = np.where(np.abs(df["Close"]) > 1, np.nan, np.arccos(df["Close"]))
            df["ASIN"] = np.where(np.abs(df["Close"]) > 1, np.nan, np.arcsin(df["Close"]))
    
        df["ATAN"] = np.arctan(df["Close"])
        df["CEIL"] = np.ceil(df["Close"])
        df["COS"] = np.cos(df["Close"])
        df["FLOOR"] = np.floor(df["Close"])
        df["LN"] = np.log(df["Close"])
        df["LOG10"] = np.log10(df["Close"])
        df["SIN"] = np.sin(df["Close"])
        df["SQRT"] = np.sqrt(df["Close"])
        df["TAN"] = np.tan(df["Close"])
        df["TANH"] = np.tanh(df["Close"])
    
        return df.fillna(0)

    def momentum_indicator_functions(self, df=None, periods=[23,115,220]):
        """
        Applies various momentum indicator functions to the input data.
    
        Args:
            df (pd.DataFrame, optional): Input DataFrame containing price-related columns.
                If not provided, the DataFrame passed during class initialization will be used.
    
        Returns:
            pd.DataFrame: DataFrame with applied momentum indicator functions.
    
        This function applies various momentum indicator functions to the columns such as 'open', 'high', 'low', 'close', and 'real_volume'.
        """
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
        if periods is None:
            periods = self.periods
        
        for i in periods:
            df["ADX" + str(i)] = tl.ADX(df["High"], df["Low"], df["Close"], timeperiod=i)
            df["ADXR" + str(i)] = tl.ADXR(df["High"], df["Low"], df["Close"], timeperiod=i)
            df["AROONOSC" + str(i)] = tl.AROONOSC(df["High"], df["Low"], timeperiod=i)
            df["CCI" + str(i)] = tl.CCI(df["High"], df["Low"], df["Close"], timeperiod=i)
            df["CMO" + str(i)] = tl.CMO(df["Close"], timeperiod=i)
            df["DX" + str(i)] = tl.DX(df["High"], df["Low"], df["Close"], timeperiod=i)
            df["MFI" + str(i)] = tl.MFI(df["High"], df["Low"], df["Close"], df["Volume"], timeperiod=i)
            df["MINUS_DI" + str(i)] = tl.MINUS_DI(df["High"], df["Low"], df["Close"], timeperiod=i)
            df["MINUS_DM" + str(i)] = tl.MINUS_DM(df["High"], df["Low"], timeperiod=i)
            df["MOM" + str(i)] = tl.MOM(df["Close"], timeperiod=i)
            df["PLUS_DI" + str(i)] = tl.PLUS_DI(df["High"], df["Low"], df["Close"], timeperiod=i)
            df["PLUS_DM" + str(i)] = tl.PLUS_DM(df["High"], df["Low"], timeperiod=i)
            df["ROC" + str(i)] = tl.ROC(df["Close"], timeperiod=i)
            df["ROCP" + str(i)] = tl.ROCP(df["Close"], timeperiod=i)
            df["ROCR" + str(i)] = tl.ROCR(df["Close"], timeperiod=i)
            df["ROCR100" + str(i)] = tl.ROCR100(df["Close"], timeperiod=i)
            df["RSI" + str(i)] = tl.RSI(df["Close"], timeperiod=i)
            df["WILLR" + str(i)] = tl.WILLR(df["High"], df["Low"], df["Close"], timeperiod=i)
            df["TRIX" + str(i)] = tl.TRIX(df["Close"], timeperiod=i)
    
        df["APO"] = tl.APO(df["Close"], fastperiod=12, slowperiod=26, matype=0)
        df["PPO"] = tl.PPO(df["Close"], fastperiod=12, slowperiod=26, matype=0)
        df["real"] = tl.ULTOSC(df["High"], df["Low"], df["Close"], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    
        return df.fillna(0)

    def statistic_functions(self, df=None, periods=[23,115,220]):
        """
        Applies various statistical functions to the input data.
    
        Args:
            df (pd.DataFrame, optional): Input DataFrame containing price-related columns.
                If not provided, the DataFrame passed during class initialization will be used.
            periods (list of int, optional): List of periods for calculations. 
                If not provided, the periods set during class initialization will be used.
    
        Returns:
            pd.DataFrame: Dataframe with applied statistical functions.
    
        This function applies various statistical functions to the columns such as 'high', 'low', and 'close'.
        """
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
        if periods is None:
            periods = self.periods
        
        for i in periods:
            df["BETA" + str(i)] = tl.BETA(df["High"], df["Low"], timeperiod=i)
            df["CORREL" + str(i)] = tl.CORREL(df["High"], df["Low"], timeperiod=i)
            df["LINEARREG" + str(i)] = tl.LINEARREG(df["Close"], timeperiod=i)
            df["LINEARREG_ANGLE" + str(i)] = tl.LINEARREG_ANGLE(df["Close"], timeperiod=i)
            df["LINEARREG_INTERCEPT" + str(i)] = tl.LINEARREG_INTERCEPT(df["Close"], timeperiod=i)
            df["LINEARREG_SLOPE" + str(i)] = tl.LINEARREG_SLOPE(df["Close"], timeperiod=i)
            df["STDDEV" + str(i)] = tl.STDDEV(df["Close"], timeperiod=i, nbdev=1)
            df["TSF" + str(i)] = tl.TSF(df["Close"], timeperiod=i)
            df["VAR" + str(i)] = tl.VAR(df["Close"], timeperiod=i, nbdev=1)
            df["median" + str(i)] = df["Close"].rolling(window=i, min_periods=1).median()
            df["mode" + str(i)] = df["Close"].rolling(window=i, min_periods=1).apply(lambda x: x.mode()[0])
            df["std" + str(i)] = df["median" + str(i)].rolling(window=i, min_periods=1).std()
    
        return df.fillna(0)

    def math_operator_functions(self, df=None, periods=[23,115,220]):
        """
        Applies various mathematical operator functions to the input data.
    
        Args:
            df (pd.DataFrame, optional): Input DataFrame containing price-related columns.
                If not provided, the DataFrame passed during class initialization will be used.
            periods (list of int, optional): List of periods for calculations. 
                If not provided, the periods set during class initialization will be used.
    
        Returns:
            pd.DataFrame: Dataframe with applied mathematical operator functions.
    
        This function applies various mathematical operator functions to the columns such as 'high', 'low', and 'close'.
        """
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
        if periods is None:
            periods = self.periods
            
        for i in periods:
            df["MAX"+str(i)] = tl.MAX(df["Close"], timeperiod=i)
            df["MAXINDEX"+str(i)] = tl.MAXINDEX(df["Close"], timeperiod=i)
            df["MIN"+str(i)] = tl.MIN(df["Close"], timeperiod=i)
            df["MININDEX"+str(i)] = tl.MININDEX(df["Close"], timeperiod=i)
            df["SUM"+str(i)] = tl.SUM(df["Close"], timeperiod=i)
    
        df["ADD"] = df["High"] + df["Low"]
        df["DIV"] = df["High"] / df["Low"]
        df["SUB"] = df["High"] - df["Low"]
    
        return df.fillna(0)

    def all_talib_indicators(self, df=None):
        """
        Adds all available indicators to the input DataFrame.
    
        Args:
            df (pd.DataFrame, optional): Input DataFrame to which the indicators will be added.
                If not provided, the DataFrame passed during class initialization will be used.
    
        Returns:
            pd.DataFrame: DataFrame with added indicators.
    
        This function iterates over all available indicator methods in the class and adds their outputs to the input DataFrame.
        """
        try:
            if df is None:
                df = self.df.copy()
        except AttributeError as e:
            print(f"Exept: {e}\nThe user allegedly did not submit data for processing")
            return None
    
        indicator_methods = ['price_changes','indicators_pattern_recognition_functions',
                             'calculate_overlap_studies','math_transform_functions',
                             'momentum_indicator_functions','statistic_functions',
                             'math_operator_functions']

        merged_data = df.copy() if df is not None else None
        
        for method in indicator_methods:
            m = getattr(self, method)
            method_result = m(df)
            merged_data = pd.merge(merged_data, method_result, 
                                   on=df.columns.to_list(),
                                   how='inner',
                                   suffixes=('_orig', '_added'))

        return merged_data
    




if __name__ == "__main__":
    indicators = Indicators()
    data_col = DataCollector()
    
    data = data_col.get_historical_data(symbol="EURUSD")
    data = indicators.indicators_pattern_recognition_functions(data)
    print(data)