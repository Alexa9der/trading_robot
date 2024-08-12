from trading_robot.data_collection.data_collector import DataCollector, mt5
from trading_robot.utils.logger import log_message
import talib as tl
import pandas as pd 
import numpy as np



class TLIndicators:
    """
    A class to preprocess stock data by applying various technical tl_indicators and mathematical transformations.
    """
    def __init__ (self, data: pd.DataFrame, Open:str="FLOpen", Close:str="FLClose",
                  High:str="FLHigh", Low:str="FLLow", Volume:str="FLVolume"):
        """
        Initializes the Preprocessing_stock_data class.

        Args:
            data (pd.DataFrame): Input data to be processed.
            periods (list, optional): List of periods for calculations. Defaults to None.

        This function initializes the Preprocessing_stock_data class and sets the dataframe, columns, and periods to be used for processing.
        """

        self._df = data.copy()
        self._Open = data[Open].copy()
        self._Close = data[Close].copy()
        self._High = data[High].copy() 
        self._Low = data[Low].copy() 
        self._Volume = data[Volume].copy() 



    def indicators_pattern_recognition_functions(self):
        """
        Adds pattern recognition tl_indicators to the dataframe.
    
        Returns:
            pd.DataFrame: Dataframe with added pattern recognition tl_indicators.
    
        This function creates copies of the input data and adds pattern recognition tl_indicators to the dataframe.
        """
        
        log_message("Starting pattern recognition indicator calculations.")

        self._df["CDL2CROWS"] = tl.CDL2CROWS(self._Open, self._High, self._Low, self._Close) # type: ignore
        self._df["CDL3BLACKCROWS"] = tl.CDL3BLACKCROWS(self._Open, self._High, self._Low, self._Close)
        self._df["CDL3INSIDE"] = tl.CDL3INSIDE(self._Open, self._High, self._Low, self._Close) # type: ignore
        self._df["CDL3LINESTRIKE"] = tl.CDL3LINESTRIKE(self._Open, self._High, self._Low, self._Close)
        self._df["CDL3OUTSIDE"] = tl.CDL3OUTSIDE(self._Open, self._High, self._Low, self._Close)
        self._df["CDL3STARSINSOUTH"] = tl.CDL3STARSINSOUTH(self._Open, self._High, self._Low, self._Close)
        self._df["CDL3WHITESOLDIERS"] = tl.CDL3WHITESOLDIERS(self._Open, self._High, self._Low, self._Close)
        self._df["CDLABANDONEDBABY"] = tl.CDLABANDONEDBABY(self._Open, self._High, self._Low, self._Close)
        self._df["CDLADVANCEBLOCK"] = tl.CDLADVANCEBLOCK(self._Open, self._High, self._Low, self._Close)
        self._df["CDLBELTHOLD"] = tl.CDLBELTHOLD(self._Open, self._High, self._Low, self._Close)
        self._df["CDLBREAKAWAY"] = tl.CDLBREAKAWAY(self._Open, self._High, self._Low, self._Close)
        self._df["CDLCLOSINGMARUBOZU"] = tl.CDLCLOSINGMARUBOZU(self._Open, self._High, self._Low, self._Close)
        self._df["CDLCONCEALBABYSWALL"] = tl.CDLCONCEALBABYSWALL(self._Open, self._High, self._Low, self._Close)
        self._df["CDLCOUNTERATTACK"] = tl.CDLCOUNTERATTACK(self._Open, self._High, self._Low, self._Close)
        self._df["CDLDARKCLOUDCOVER"] = tl.CDLDARKCLOUDCOVER(self._Open, self._High, self._Low, self._Close)
        self._df["CDLDOJI"] = tl.CDLDOJI(self._Open, self._High, self._Low, self._Close)
        self._df["CDLDOJISTAR"] = tl.CDLDOJISTAR(self._Open, self._High, self._Low, self._Close)
        self._df["CDLDRAGONFLYDOJI"] = tl.CDLDRAGONFLYDOJI(self._Open, self._High, self._Low, self._Close)
        self._df["CDLENGULFING"] = tl.CDLENGULFING(self._Open, self._High, self._Low, self._Close)
        self._df["CDLEVENINGDOJISTAR"] = tl.CDLEVENINGDOJISTAR(self._Open, self._High, self._Low, self._Close, penetration=0)
        self._df["CDLEVENINGSTAR"] = tl.CDLEVENINGSTAR(self._Open, self._High, self._Low, self._Close, penetration=0)
        self._df["CDLGAPSIDESIDEWHITE"] = tl.CDLGAPSIDESIDEWHITE(self._Open, self._High, self._Low, self._Close)
        self._df["CDLGRAVESTONEDOJI"] = tl.CDLGRAVESTONEDOJI(self._Open, self._High, self._Low, self._Close)
        self._df["CDLHAMMER"] = tl.CDLHAMMER(self._Open, self._High, self._Low, self._Close)
        self._df["CDLHANGINGMAN"] = tl.CDLHANGINGMAN(self._Open, self._High, self._Low, self._Close)
        self._df["CDLHARAMI"] = tl.CDLHARAMI(self._Open, self._High, self._Low, self._Close)
        self._df["CDLHARAMICROSS"] = tl.CDLHARAMICROSS(self._Open, self._High, self._Low, self._Close)
        self._df["CDLHIGHWAVE"] = tl.CDLHIGHWAVE(self._Open, self._High, self._Low, self._Close)
        self._df["CDLHIKKAKE"] = tl.CDLHIKKAKE(self._Open, self._High, self._Low, self._Close)
        self._df["CDLHIKKAKEMOD"] = tl.CDLHIKKAKEMOD(self._Open, self._High, self._Low, self._Close)
        self._df["CDLHOMINGPIGEON"] = tl.CDLHOMINGPIGEON(self._Open, self._High, self._Low, self._Close)
        self._df["CDLIDENTICAL3CROWS"] = tl.CDLIDENTICAL3CROWS(self._Open, self._High, self._Low, self._Close)
        self._df["CDLINNECK"] = tl.CDLINNECK(self._Open, self._High, self._Low, self._Close)
        self._df["CDLINVERTEDHAMMER"] = tl.CDLINVERTEDHAMMER(self._Open, self._High, self._Low, self._Close)
        self._df["CDLKICKING"] = tl.CDLKICKING(self._Open, self._High, self._Low, self._Close)
        self._df["CDLKICKINGBYLENGTH"] = tl.CDLKICKINGBYLENGTH(self._Open, self._High, self._Low, self._Close)
        self._df["CDLLADDERBOTTOM"] = tl.CDLLADDERBOTTOM(self._Open, self._High, self._Low, self._Close)
        self._df["CDLLONGLEGGEDDOJI"] = tl.CDLLONGLEGGEDDOJI(self._Open, self._High, self._Low, self._Close)
        self._df["CDLLONGLINE"] = tl.CDLLONGLINE(self._Open, self._High, self._Low, self._Close)
        self._df["CDLMARUBOZU"] = tl.CDLMARUBOZU(self._Open, self._High, self._Low, self._Close)
        self._df["CDLMATCHINGLOW"] = tl.CDLMATCHINGLOW(self._Open, self._High, self._Low, self._Close)
        self._df["CDLMATHOLD"] = tl.CDLMATHOLD(self._Open, self._High, self._Low, self._Close, penetration=0)
        self._df["CDLMORNINGDOJISTAR"] = tl.CDLMORNINGDOJISTAR(self._Open, self._High, self._Low, self._Close, penetration=0)
        self._df["CDLMORNINGSTAR"] = tl.CDLMORNINGSTAR(self._Open, self._High, self._Low, self._Close, penetration=0)
        self._df["CDLONNECK"] = tl.CDLONNECK(self._Open, self._High, self._Low, self._Close)
        self._df["CDLPIERCING"] = tl.CDLPIERCING(self._Open, self._High, self._Low, self._Close)
        self._df["CDLRICKSHAWMAN"] = tl.CDLRICKSHAWMAN(self._Open, self._High, self._Low, self._Close)
        self._df["CDLRISEFALL3METHODS"] = tl.CDLRISEFALL3METHODS(self._Open, self._High, self._Low, self._Close)
        self._df["CDLSEPARATINGLINES"] = tl.CDLSEPARATINGLINES(self._Open, self._High, self._Low, self._Close)
        self._df["CDLSHOOTINGSTAR"] = tl.CDLSHOOTINGSTAR(self._Open, self._High, self._Low, self._Close)
        self._df["CDLSHORTLINE"] = tl.CDLSHORTLINE(self._Open, self._High, self._Low, self._Close)
        self._df["CDLSPINNINGTOP"] = tl.CDLSPINNINGTOP(self._Open, self._High, self._Low, self._Close)
        self._df["CDLSTALLEDPATTERN"] = tl.CDLSTALLEDPATTERN(self._Open, self._High, self._Low, self._Close)
        self._df["CDLSTICKSANDWICH"] = tl.CDLSTICKSANDWICH(self._Open, self._High, self._Low, self._Close)
        self._df["CDLTAKURI"] = tl.CDLTAKURI(self._Open, self._High, self._Low, self._Close)
        self._df["CDLTASUKIGAP"] = tl.CDLTASUKIGAP(self._Open, self._High, self._Low, self._Close)
        self._df["CDLTHRUSTING"] = tl.CDLTHRUSTING(self._Open, self._High, self._Low, self._Close)
        self._df["CDLTRISTAR"] = tl.CDLTRISTAR(self._Open, self._High, self._Low, self._Close)
        self._df["CDLUNIQUE3RIVER"] = tl.CDLUNIQUE3RIVER(self._Open, self._High, self._Low, self._Close)
        self._df["CDLUPSIDEGAP2CROWS"] = tl.CDLUPSIDEGAP2CROWS(self._Open, self._High, self._Low, self._Close)
        self._df["CDLXSIDEGAP3METHODS"] = tl.CDLXSIDEGAP3METHODS(self._Open, self._High, self._Low, self._Close)

        log_message("Completed pattern recognition indicator calculations.")

        return self._df.fillna(0)

    def calculate_overlap_studies(self, periods=[23,115,220]):
        """
        Calculates various overlap studies for the input data.
    
        Args:
            periods (list[int], optional): List of periods for overlap studies calculations.
                If not provided, the periods specified during class initialization will be used.
    
        Returns:
            pd.DataFrame: DataFrame with calculated overlap studies.
    
        This function calculates various overlap studies based on the provided periods.
        """
        log_message("Starting overlap studies calculations.")
        
        new_data = {}
        for i in periods:
            new_data["DEMA"+str(i)] = tl.DEMA(self._Close, timeperiod=i)
            new_data["EMA"+str(i)] = tl.EMA(self._Close, timeperiod=i)
            new_data["KAMA"+str(i)] = tl.KAMA(self._Close, timeperiod=i)
            new_data["MIDPOINT"+str(i)] = tl.MIDPOINT(self._Close, timeperiod=i)
            new_data["SMA"+str(i)] = tl.SMA(self._Close, timeperiod=i)
            new_data["TRIMA"+str(i)] = tl.TRIMA(self._Close, timeperiod=i)
            new_data["WMA"+str(i)] = tl.WMA(self._Close, timeperiod=i)
            new_data["T3"+str(i)] = tl.T3(self._Close, timeperiod=i, vfactor=0)
            new_data["TEMA"+str(i)] = tl.TEMA(self._Close, timeperiod=i)
            new_data["MA"+str(i)] = tl.MA(self._Close, timeperiod=i, matype=0)
            new_data["HT_TRENDLINE"+str(i)] = tl.HT_TRENDLINE(self._Close)

        new_data_df = pd.DataFrame(new_data, index=self._df.index)
        self._df = pd.concat([self._df, new_data_df], axis=1)

        log_message("Completed overlap studies calculations.")

        return self._df.fillna(0)

    def math_transform_functions(self):
        """
        Applies various mathematical transformation functions to the input data.
    
        Returns:
            pd.DataFrame: DataFrame with applied mathematical transformation functions.
    
        This function applies various mathematical transformation functions to the 'close' column.
        """
        log_message("Starting mathematical transformation functions.")

        with np.errstate(invalid='ignore'):
            self._df["ACOS"] = np.where(np.abs(self._Close) > 1, np.nan, np.arccos(self._Close))
            self._df["ASIN"] = np.where(np.abs(self._Close) > 1, np.nan, np.arcsin(self._Close))
    
        self._df["ATAN"] = np.arctan(self._Close)
        self._df["CEIL"] = np.ceil(self._Close)
        self._df["COS"] = np.cos(self._Close)
        self._df["FLOOR"] = np.floor(self._Close)
        self._df["LN"] = np.log(self._Close)
        self._df["LOG10"] = np.log10(self._Close)
        self._df["SIN"] = np.sin(self._Close)
        self._df["SQRT"] = np.sqrt(self._Close)
        self._df["TAN"] = np.tan(self._Close)
        self._df["TANH"] = np.tanh(self._Close)

        log_message("Completed mathematical transformation functions.")
    
        return self._df.fillna(0)

    def momentum_indicator_functions(self, periods=[23,115,220]):
        """
        Applies various momentum indicator functions to the input data.
    
        Args:
            periods (list[int], optional): List of periods for overlap studies calculations.
                If not provided, the periods specified during class initialization will be used.
    
        Returns:
            pd.DataFrame: DataFrame with applied momentum indicator functions.
    
        This function applies various momentum indicator functions to the columns such as 'open', 'high', 'low', 'close', and 'real_volume'.
        """
        log_message("Starting momentum indicator calculations.")

        new_data = {}
        for i in periods:
            new_data["ADX" + str(i)] = tl.ADX(self._High, self._Low, self._Close, timeperiod=i)
            new_data["ADXR" + str(i)] = tl.ADXR(self._High, self._Low, self._Close, timeperiod=i)
            new_data["AROONOSC" + str(i)] = tl.AROONOSC(self._High, self._Low, timeperiod=i)
            new_data["CCI" + str(i)] = tl.CCI(self._High, self._Low, self._Close, timeperiod=i)
            new_data["CMO" + str(i)] = tl.CMO(self._Close, timeperiod=i)
            new_data["DX" + str(i)] = tl.DX(self._High, self._Low, self._Close, timeperiod=i)
            new_data["MFI" + str(i)] = tl.MFI(self._High, self._Low, self._Close, self._df["Volume"], timeperiod=i)
            new_data["MINUS_DI" + str(i)] = tl.MINUS_DI(self._High, self._Low, self._Close, timeperiod=i)
            new_data["MINUS_DM" + str(i)] = tl.MINUS_DM(self._High, self._Low, timeperiod=i)
            new_data["MOM" + str(i)] = tl.MOM(self._Close, timeperiod=i)
            new_data["PLUS_DI" + str(i)] = tl.PLUS_DI(self._High, self._Low, self._Close, timeperiod=i)
            new_data["PLUS_DM" + str(i)] = tl.PLUS_DM(self._High, self._Low, timeperiod=i)
            new_data["ROC" + str(i)] = tl.ROC(self._Close, timeperiod=i)
            new_data["ROCP" + str(i)] = tl.ROCP(self._Close, timeperiod=i)
            new_data["ROCR" + str(i)] = tl.ROCR(self._Close, timeperiod=i)
            new_data["ROCR100" + str(i)] = tl.ROCR100(self._Close, timeperiod=i)
            new_data["RSI" + str(i)] = tl.RSI(self._Close, timeperiod=i)
            new_data["WILLR" + str(i)] = tl.WILLR(self._High, self._Low, self._Close, timeperiod=i)
            new_data["TRIX" + str(i)] = tl.TRIX(self._Close, timeperiod=i)
    
        new_data_df = pd.DataFrame(new_data, index=self._df.index)
        self._df = pd.concat([self._df, new_data_df], axis=1)

        self._df["APO"] = tl.APO(self._Close, fastperiod=12, slowperiod=26, matype=0)
        self._df["PPO"] = tl.PPO(self._Close, fastperiod=12, slowperiod=26, matype=0)
        self._df["real"] = tl.ULTOSC(self._High, self._Low, self._Close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        log_message("Completed momentum indicator calculations.")

        return self._df.fillna(0)

    def statistic_functions(self, periods=[23,115,220]):
        """
        Applies various statistical functions to the input data.
    
        Args:
            periods (list of int, optional): List of periods for calculations. 
                If not provided, the periods set during class initialization will be used.
    
        Returns:
            pd.DataFrame: Dataframe with applied statistical functions.
    
        This function applies various statistical functions to the columns such as 'high', 'low', and 'close'.
        """
        log_message("Starting statistical functions calculations.")
        
        new_data = {}
        for i in periods:
            new_data[f"BETA{i}"] = tl.BETA(self._High, self._Low, timeperiod=i)
            new_data[f"CORREL{i}"] = tl.CORREL(self._High, self._Low, timeperiod=i)
            new_data[f"LINEARREG{i}"] = tl.LINEARREG(self._Close, timeperiod=i)
            new_data[f"LINEARREG_ANGLE{i}"] = tl.LINEARREG_ANGLE(self._Close, timeperiod=i)
            new_data[f"LINEARREG_INTERCEPT{i}"] = tl.LINEARREG_INTERCEPT(self._Close, timeperiod=i)
            new_data[f"LINEARREG_SLOPE{i}"] = tl.LINEARREG_SLOPE(self._Close, timeperiod=i)
            new_data[f"STDDEV{i}"] = tl.STDDEV(self._Close, timeperiod=i, nbdev=1)
            new_data[f"TSF{i}"] = tl.TSF(self._Close, timeperiod=i)
            new_data[f"VAR{i}"] = tl.VAR(self._Close, timeperiod=i, nbdev=1)

        new_data_df = pd.DataFrame(new_data, index=self._df.index)
        self._df = pd.concat([self._df, new_data_df], axis=1)

        for i in periods:
            self._df[f"median{i}"] = self._Close.rolling(window=i, min_periods=1).median()
            self._df[f"mode{i}"] = self._Close.rolling(window=i, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
            self._df[f"std{i}"] = self._df[f"median{i}"].rolling(window=i, min_periods=1).std()
    

        self._df = self._df.fillna(0)

        log_message("Completed statistical functions calculations.")

        return self._df
    
    def math_operator_functions(self, periods=[23,115,220]):
        """
        Applies various mathematical operator functions to the input data.
    
        Args:
            periods (list of int, optional): List of periods for calculations. 
                If not provided, the periods set during class initialization will be used.
    
        Returns:
            pd.DataFrame: Dataframe with applied mathematical operator functions.
    
        This function applies various mathematical operator functions to the columns such as 'high', 'low', and 'close'.
        """
        log_message("Starting mathematical operator functions calculations.")
        
        new_data = {}
        for i in periods:
            new_data["MAX"+str(i)] = tl.MAX(self._Close, timeperiod=i)
            new_data["MAXINDEX"+str(i)] = tl.MAXINDEX(self._Close, timeperiod=i)
            new_data["MIN"+str(i)] = tl.MIN(self._Close, timeperiod=i)
            new_data["MININDEX"+str(i)] = tl.MININDEX(self._Close, timeperiod=i)
            new_data["SUM"+str(i)] = tl.SUM(self._Close, timeperiod=i)
    
        new_data_df = pd.DataFrame(new_data, index=self._df.index)
        self._df = pd.concat([self._df, new_data_df], axis=1)

        self._df["ADD"] = self._High + self._Low
        self._df["DIV"] = self._High / self._Low
        self._df["SUB"] = self._High - self._Low

        log_message("Completed mathematical operator functions calculations.")

        return self._df.fillna(0)

    def all_talib_indicators(self ):
        """
        Adds all available tl_indicators to the input DataFrame.
    
        Returns:
            pd.DataFrame: DataFrame with added tl_indicators.
    
        This function iterates over all available indicator methods in the class and adds their outputs to the input DataFrame.
        """
        log_message("Starting to add all TA-Lib indicators.")

        class_methods = [ methods for methods in dir(self) if not methods.startswith("_") and "all_talib_indicators" not in methods]

        for method in class_methods:
            try:
                m = getattr(self, method)
                m()
            except Exception as e:
                log_message(f"Error applying indicator method {method}: {e}")
        
        log_message("Completed adding all TA-Lib indicators.")

        return self._df
    




if __name__ == "__main__":
    data_col = DataCollector()
    data = data_col.get_historical_data(symbol="EURUSD")

    tl_indicators = TLIndicators(data)
    data = tl_indicators.all_talib_indicators()
    print(data)
