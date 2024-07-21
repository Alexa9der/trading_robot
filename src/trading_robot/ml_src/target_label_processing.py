import numpy as np 
import pandas as pd 
import random

from src.data_collector import *
from src.machine_learning.data_balancer import *
from src.talib_indicators import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class TargetLabelProcessing(DataCollector):

    def train_test_data_collector(self, symbols=None, train_size=0.8, save = True, 
                             path="data/input_data", file_names=["train_data", "test_data"]):
        
        indicators = Indicators()
        data_balancer = DataBalancer()
        
        random.seed(42)
        
        all_symbols = self.get_all_symbols() if symbols is None else symbols  
        
        train_symbols = []
        test_symbols = []
        train_data = [] 
        test_data = [] 
        symbols_not_included = []
        
        market_type = ["z", "us", "eu", "uk", "other"]
        
        characters_sorted = {"z":  [symbol for symbol in all_symbols if "z" in symbol], 
                             "us": [symbol for symbol in all_symbols if "us" in symbol], 
                             "eu": [symbol for symbol in all_symbols if "eu" in symbol], 
                             "uk": [symbol for symbol in all_symbols if "uk" in symbol], 
                             "other": [symbol for symbol in all_symbols if not any(substring in symbol for substring in market_type)]
                            }
        
        for market in tqdm(market_type, desc="Division of characters into train and test"):
            symbols = characters_sorted[market]
            train_symbols.append(random.sample(symbols, int(train_size * len(symbols))))
            test_symbols.append(list(set(symbols) - set(train_symbols[-1])))
        
        train_symbols = sum(train_symbols, [])
        test_symbols = sum(test_symbols, [])
        
        random.shuffle(all_symbols)
        for symbol in  tqdm(all_symbols, desc="Collecting data for training and testing datasets"):
            try:
                df = self.merge_data_with_target(symbol = symbol) 
            except TypeError as e:
                print(f"Except: {e}\nFailed to get symbol data: {symbol}.")
                symbols_not_included.append(symbol)
                continue
            df = indicators.all_talib_indicators(df) 
            try:
                df = self.normalize_data(df, target="direction", method="standard" )
            except ValueError as e:
                print(f"Except: {e}\nFailed to normalize symbol data: {symbol} and therefore was skipped.")
                symbols_not_included.append(symbol)
                continue
            df = data_balancer.undersample(df, target="direction", method='random', all_small_classes=True, coefficient=5)  
            
            if symbol in train_symbols:
                train_data.append(df)
            elif symbol in test_symbols:
                test_data.append(df)
        
        train_data = pd.concat(train_data, axis=0, ignore_index= True)
        test_data = pd.concat(test_data, axis=0, ignore_index= True)
    
        if save:
            train_data.to_csv(f"{path}/{file_names[0]}.csv", index=False)
            test_data.to_csv(f"{path}/{file_names[1]}.csv", index=False)

        if symbols_not_included:
            print(f"The characters {symbols_not_included} were not included in any data set.")
            
        return train_data, test_data

    def merge_data_with_target(self, data=None, symbol=None, 
                               data_period=mt5.TIMEFRAME_H1, 
                               senior_period=mt5.TIMEFRAME_D1):

        # Dictionary for timeframe correspondence
        TIMEFRAME = {  
            mt5.TIMEFRAME_M5: mt5.TIMEFRAME_M30,  
            mt5.TIMEFRAME_M15: mt5.TIMEFRAME_H1,  
            mt5.TIMEFRAME_M30: mt5.TIMEFRAME_H4,  
            mt5.TIMEFRAME_H1: mt5.TIMEFRAME_D1,   
            mt5.TIMEFRAME_D1: mt5.TIMEFRAME_W1,   
            mt5.TIMEFRAME_W1: mt5.TIMEFRAME_MN1   
        }
        
        if data is not None:  # Checking if data is passed
            data = data.copy()
        elif data is None:  # If data is not passed, retrieve it based on symbol
            if symbol:
                data = self.get_historical_data(symbol, data_period)
            else:
                raise ValueError("The required 'symbol' variable was not passed to the function.")
        
        if symbol:  # Checking if symbol is passed
            if senior_period:
                highest_data = self.get_historical_data(symbol, senior_period)  # Getting data for higher timeframe
            else: 
                highest_data = self.get_historical_data(symbol, TIMEFRAME[data_period])
        else:
            raise ValueError("The required 'symbol' variable was not passed to the function.")
        
        # Normalizing direction and computing price differences for future calculations
        highest_data["dif"] = (highest_data['Open'] - highest_data['Close']) * -1 
        highest_data["h_c_dif"] =  highest_data['Close'] - highest_data['High']   
        highest_data["l_c_dif"] = highest_data['Close'] - highest_data['Low']     
        
        # Setting price movement direction
        highest_data["direction"] = np.where(highest_data["dif"] >= 0, 1, -1) 
        
        # Computing candle count in movement and summing price changes
        highest_data['candle_count'] = highest_data.groupby((highest_data['direction'] != highest_data['direction'].shift(1)).cumsum()).cumcount() + 1 
        highest_data['profit'] = np.where( highest_data["direction"] > 0,
                highest_data.groupby((highest_data['direction'] != highest_data['direction'].shift(1)).cumsum())['l_c_dif'].cumsum(), 
                highest_data.groupby((highest_data['direction'] != highest_data['direction'].shift(1)).cumsum())['h_c_dif'].cumsum()) 
        
        # Forming data on movement start and end
        start_movement_masc = (
            (highest_data['candle_count'] == 1 )            
          & (highest_data['candle_count'].shift(-1) != 1))
        end_movement_masc = (
            (highest_data['candle_count'] > 1 )            
          & (highest_data['candle_count'].shift(-1) == 1))
        
        start = highest_data[start_movement_masc ]
        end = highest_data[end_movement_masc]
        end = end.rename(columns = {'direction':'direction_end', 
                                             'candle_count':'candle_count_end', 
                                             'profit':'profit_end'})
        # Combining movement data
        start_movement = pd.concat([start[['Date','Open','Close','High','Low','Volume','direction','candle_count']].reset_index(),
                              end[['Date','direction_end', 'candle_count_end', 'profit_end']].reset_index()], 
                              axis=0 )
        start_movement = start_movement.sort_values(["Date", "index"])
        start_movement[["direction_end","candle_count_end","profit_end"]] = start_movement[["direction_end","candle_count_end","profit_end"]].shift(-1)
        start_movement = start_movement[start_movement["Open"].notna()].drop(columns = "index")
        
    
        # Normalizing and merging data with main data
        data["Year_Month_Day"] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
        start_movement = start_movement.rename(columns={"Date":"Year_Month_Day"})
        start_movement["Year_Month_Day"] = pd.to_datetime(start_movement['Year_Month_Day']).dt.strftime('%Y-%m-%d')
        
        # Creating masks for merging
        buy_masc = (start_movement["direction"] == 1 ) 
        short_masc = (start_movement["direction"] == -1 ) 
        
        # Merging data
        merge_data = pd.merge(data, 
                              start_movement.loc[buy_masc, ["Low","Year_Month_Day","direction",'direction_end', "candle_count", 'candle_count_end', 'profit_end']], 
                              how='left', 
                              on=["Low","Year_Month_Day"] )
        
        merge_data = pd.merge(merge_data, 
                          start_movement.loc[short_masc, ["High","Year_Month_Day","direction", 'direction_end', "candle_count", 'candle_count_end', 'profit_end']], 
                          how='left', 
                          on=["High","Year_Month_Day"], 
                          suffixes=('_b', '_s'))
        
        # Filling missing values and combining columns
        merge_data = merge_data.fillna(0)
        
        merge_data["direction"] = merge_data["direction_b"] + merge_data["direction_s"]
        merge_data["direction_end"] = merge_data["direction_end_b"] + merge_data["direction_end_s"]
        merge_data["candle_count"] = merge_data["candle_count_b"] + merge_data["candle_count_s"]
        merge_data["candle_count_end"] = merge_data["candle_count_end_b"] + merge_data["candle_count_end_s"]
        
        merge_data["profit"] = merge_data["profit_end_b"] + merge_data["profit_end_s"]
        
        # Dropping unnecessary columns
        merge_data = merge_data.drop(columns=["direction_b","direction_s",
                                              "direction_end_b","direction_end_s",
                                              "candle_count_b","candle_count_s",
                                              "candle_count_end_b","candle_count_end_s",
                                              "profit_end_b","profit_end_s",
                                              "Year_Month_Day"])
        
        return merge_data

    def merge_symbols_with_target(self, symbols, data_period=mt5.TIMEFRAME_H1, 
                                  senior_period=mt5.TIMEFRAME_D1):
        
        data = pd.DataFrame()
        for symbol in symbols:
            print(symbol)
            intermediate_data = self.merge_data_with_target(symbol=symbol, data_period=data_period, 
                                                            senior_period=senior_period)

            if data.empty:
                data = intermediate_data
            else :
                data = pd.concat([data, intermediate_data])

        return data

    def normalize_data(self, data, target, method='standard', return_date_columns=False):
        df = data.copy()

        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        drop_col =  list(datetime_columns) + [target]
        numerical_columns = df.columns.difference(drop_col)
        df = df.drop(columns=drop_col)
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return data

        scaled_data = scaler.fit_transform(df[numerical_columns].to_numpy())

        df[numerical_columns] = scaled_data

        if return_date_columns:
            df[drop_col] = data[drop_col]
        else:
            df[target] = data[target]
            

        return df