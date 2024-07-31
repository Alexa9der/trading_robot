import pandas as pd
import numpy as np
from docker_connector import mt5

from trading_robot.connectors.connect_mt5 import ConnectMT5
from data_collection.data_collector import DataCollector 



class RiskManager(ConnectMT5):
    
    def __init__(self, balance=None, benefit_ratio=5, 
                  risk_per_day=10, risk_per_trade=2, 
                  money_risk_per_day=None, 
                  money_risk_per_trade=None):
        
        super().__init__()
        try:
            self._connect_to_mt5()
            self.balance = balance if balance else mt5.account_info().balance
        finally:
            mt5.shutdown()
            
        self.benefit_ratio = benefit_ratio
        self.risk_per_day = risk_per_day
        self.risk_per_trade = risk_per_trade
        
        self.money_risk_per_day = money_risk_per_day if money_risk_per_day else self.balance * (risk_per_day / 100)
        self.money_risk_per_trade = money_risk_per_trade if money_risk_per_trade else self.money_risk_per_day * (risk_per_trade / 100)
        
    def calculate_trade_parameters(self, symbol, signal,
                                   st_loss_points=100, min_stop_los = None ,
                                   verbose = False):
        """
        Calculates trade parameters such as lot size, stop-loss, and take-profit.

        Parameters:
        - symbol (str): The trading symbol.
        - direction (str): The trading direction ("buy" or "sell").
        - percent_risk (float): The percentage of balance at risk in each trade.
        - st_loss_points (int): The number of points for setting stop-loss (default is 100).
        - benefit_ratio (int): The benefit ratio for setting take-profit (default is 5).
        - balance (float): The account balance (default is None, which uses the balance from the trading account).
        - verbose (bool): If True, print detailed information about the trade parameters (default is False).
        
        Returns:
        - lot_size_to_use (float): The calculated lot size to use in the trade.
        - stop_loss_price (float): The calculated stop-loss price.
        - take_profit_price (float): The calculated take-profit price.
        """ 
        try:
            # Connect to MT5
            self._connect_to_mt5()
            # Get account information
            account_info = mt5.account_info()
            # Get symbol information
            symbol_info = mt5.symbol_info(symbol)
        finally:
            # Shutdown MT5
            mt5.shutdown()

        st_loss_points = self.__stop_loss(st_loss_points, min_stop_los,
                                 symbol_info.trade_tick_size, 
                                 symbol_info.spread )
            
        # Get entry price based on trading direction
        entry_price = symbol_info.ask if signal == "buy" else symbol_info.bid

         # Check for division by zero
        if entry_price == 0:
            raise ValueError("Function calculate_trade_parameters. Cannot calculate risk per trade due to division by zero.")

        lot_size_to_use = self.__calculate_lot_size(symbol= symbol,
                                                  symbol_info=symbol_info, 
                                                  price=entry_price, 
                                                  stop_loss=st_loss_points, 
                                                  signal=signal )
        
        # Calculate stop-loss and take-profit prices
        decimal_length = len(str(symbol_info.trade_tick_size).split('.')[-1])
        
        if signal == "buy":
            stop_loss_price = round(entry_price - st_loss_points * symbol_info.point, decimal_length)
            take_profit_price = round(entry_price + self.benefit_ratio * st_loss_points * symbol_info.point, decimal_length)
            
        elif signal == "sell":
            stop_loss_price = round(entry_price + st_loss_points * symbol_info.point, decimal_length)
            take_profit_price = round(entry_price - self.benefit_ratio * st_loss_points * symbol_info.point, decimal_length)
        
        else :
            lot_size_to_use = 0
            stop_loss_price = 0
            take_profit_price = 0
            
        # Output results
        try:
            self._connect_to_mt5()
            if self.balance <= mt5.account_info().balance - self.money_risk_per_day:
                lot_size_to_use = 0.0
        finally:
            mt5.shutdown()

        if verbose:
            print(f"Risk per trade: {self.money_risk_per_day * (self.risk_per_trade / 100)} in currency")
            print(f"Lot size to use: {lot_size_to_use}")
            print(f"Entry price: {entry_price}")
            print(f"Stop-loss price: {stop_loss_price}")
            print(f"Take-profit price: {take_profit_price}")

        return lot_size_to_use, stop_loss_price, take_profit_price

    def calculate_atr(self, symbol, timeframe=mt5.TIMEFRAME_D1, period=22):
        """
        Calculate the Average True Range (ATR) for the specified symbol and timeframe.
    
        Parameters:
        - symbol (str): The symbol for which to calculate the ATR.
        - timeframe (int): The timeframe for the ATR calculation (default is mt5.TIMEFRAME_D1).
        - period (int): The period over which to calculate the ATR (default is 250).
    
        Returns:
        - float: The last calculated ATR value.
    
        This method calculates the Average True Range (ATR) for the specified symbol and timeframe.
        It retrieves historical data for the symbol, calculates the true range for each period,
        calculates the mean true range over the specified period, and returns the last calculated
        ATR value.
        """
    
        # Retrieve historical data for the symbol and timeframe
        data_collector = DataCollector()
        data = data_collector.get_historical_data(symbol, timeframe)
    
        # Calculate the mean ATR
        data['Mean_ATR'] = abs(data['Close'] - data['Close'].shift(periods=1)) \
            .fillna(0.01) \
            .rolling(window=period, min_periods=1).mean()
    
        # Round the last ATR value to the same precision as the close price
        last_atr = round(data.iloc[-1, -1], len(str(data.iloc[-1, 2]).split(".")[-1]))
    
        return last_atr

    def test_calculate_trade_parameters(self, data, symbol, signal,
                                       st_loss_points=100, min_stop_los = None ,
                                       verbose = False):
        
        try:
            # Connect to MT5
            self._connect_to_mt5()
            # Get account information
            account_info = mt5.account_info()
            # Get symbol information
            symbol_info = mt5.symbol_info(symbol)
        finally:
            # Shutdown MT5
            mt5.shutdown()

        st_loss_points = self.__stop_loss(st_loss_points, min_stop_los,
                                 symbol_info.trade_tick_size, 
                                 symbol_info.spread )
            
        # Get entry price based on trading direction
  
        entry_price = data["Close"][-1]
        
        lot_size_to_use = self.__calculate_lot_size(symbol= symbol,
                                                  symbol_info=symbol_info, 
                                                  price=entry_price, 
                                                  stop_loss=st_loss_points, 
                                                  signal=signal )
        
        # Calculate stop-loss and take-profit prices
        decimal_length = len(str(symbol_info.trade_tick_size).split('.')[-1])
        
        if signal == "buy":
            stop_loss_price = round(entry_price - st_loss_points * symbol_info.point, decimal_length)
            take_profit_price = round(entry_price + self.benefit_ratio * st_loss_points * symbol_info.point, decimal_length)
            
        elif signal == "sell":
            stop_loss_price = round(entry_price + st_loss_points * symbol_info.point, decimal_length)
            take_profit_price = round(entry_price - self.benefit_ratio * st_loss_points * symbol_info.point, decimal_length)
        
        elif signal == 'No signal':
            lot_size_to_use = 0
            stop_loss_price = 0
            take_profit_price = 0
            
        # Output results
        try:
            self._connect_to_mt5()
            if self.balance <= mt5.account_info().balance - self.money_risk_per_day:
                lot_size_to_use = 0.0
        finally:
            mt5.shutdown()

        if verbose:
            print(f"Risk per trade: {self.money_risk_per_day * (self.risk_per_trade / 100)} in currency")
            print(f"Lot size to use: {lot_size_to_use}")
            print(f"Entry price: {entry_price}")
            print(f"Stop-loss price: {stop_loss_price}")
            print(f"Take-profit price: {take_profit_price}")

        return lot_size_to_use, stop_loss_price, take_profit_price

    def __stop_loss(self, st_loss_points, min_stop_loss, trade_tick_size, spread):
        """
        Calculate the stop loss value based on the specified stop loss points.
    
        Parameters:
        - st_loss_points (float): The stop loss value in points.
        - min_stop_loss (float): The minimum stop loss value to be applied.
        - trade_tick_size (float): The tick size for the trade.
        - spread (float): The current spread for the trading pair.
    
        Returns:
        - float: The adjusted stop loss value.
    
        This method calculates the stop loss value based on the specified stop loss points. 
        If the stop loss points are provided as a float, it adjusts them based on the trade 
        tick size. If a minimum stop loss value is provided and it is greater than the adjusted 
        stop loss points, the minimum stop loss value is used instead. The resulting stop loss 
        value is then returned.
        """
        
        if isinstance(st_loss_points, float):
            # Adjust stop loss points based on trade tick size
            st_loss_points = st_loss_points / trade_tick_size
            
            # Determine minimum stop loss value
            min_stop_loss = min_stop_loss if min_stop_loss else spread * 3
            
            # If minimum stop loss is greater than adjusted stop loss points, use minimum stop loss
            if min_stop_loss > st_loss_points:
                st_loss_points = min_stop_loss
                    
        return st_loss_points
                
    def __calculate_lot_size(self, symbol, symbol_info, price, stop_loss, signal):
        """
        Calculate the lot size for a trade based on the provided parameters.
    
        Parameters:
        - symbol (str): The trading symbol.
        - symbol_info (object): Information about the trading symbol.
        - price (float): The current price of the trading symbol.
        - stop_loss (float): The stop loss value.
        - signal (str): The trading signal ("buy" or "sell").
    
        Returns:
        - float: The calculated lot size for the trade.
    
        This method calculates the lot size for a trade based on the provided parameters.
        It takes into account various factors such as the risk per trade, stop loss value,
        and currency pair information. The lot size is calculated differently for Forex pairs
        and non-Forex symbols. For Forex pairs, it considers the currency profit currency.
        For non-Forex symbols, it may use an auxiliary currency pair for calculations.
    
        Note:
        This method assumes that the necessary data, such as symbol information and
        auxiliary currency pair data, are available and correctly provided.
    
        """
    
        # Check if the symbol is a Forex pair
        if "forex" in symbol_info.path.lower():
            if "USD" in symbol:
                if "USD" == symbol_info.currency_profit:
                    # Calculate lot size for USD-based Forex pairs
                    lot = (self.money_risk_per_trade / (stop_loss * symbol_info.point)) / symbol_info.trade_contract_size
                elif "USD" != symbol_info.currency_profit:
                    # Calculate lot size for non-USD-based Forex pairs
                    lot = ((self.money_risk_per_trade * price) / (stop_loss * symbol_info.point)) / symbol_info.trade_contract_size
            elif "USD" not in symbol:
                # Find an auxiliary currency pair with USD for calculation
                data_collector = DataCollector()
                auxiliary_course = [symbol for symbol in data_collector.get_all_symbols()
                                    if symbol_info.currency_profit in symbol and "USD" in symbol][0]
                try:
                    # Connect to MT5
                    self._connect_to_mt5()
                    # Get symbol information for the auxiliary currency pair
                    auxiliary_course_symbol_info = mt5.symbol_info(auxiliary_course)
                finally:
                    # Shutdown MT5
                    mt5.shutdown()
    
                entry_price = auxiliary_course_symbol_info.ask if signal == "buy" else auxiliary_course_symbol_info.bid
                if entry_price == 0:
                    raise ValueError("Function calculate_trade_parameters. Cannot calculate risk per trade due to division by zero.")
    
                # Calculate lot size using the auxiliary currency pair
                if "USD" == auxiliary_course_symbol_info.currency_profit:
                    lot = (self.money_risk_per_trade / ((stop_loss * symbol_info.point) * entry_price)) * symbol_info.point
                elif "USD" != auxiliary_course_symbol_info.currency_profit:
                    lot = (self.money_risk_per_trade * entry_price) / (stop_loss * symbol_info.point) * symbol_info.point
    
        else:
            # Calculate lot size for non-Forex symbols
            lot = ((self.money_risk_per_day * (self.risk_per_trade / 100)) / (stop_loss * symbol_info.trade_contract_size))
    
        # Round the lot size and ensure it is not less than the minimum volume
        lot = round(lot, len(str(symbol_info.volume_min)) - 2)
        lot = max(lot, symbol_info.volume_min)
    
        return lot
        