import pandas as pd
from docker_connector import mt5

from trading_robot.connectors.connect_mt5 import ConnectMT5
from traiding.risk_manager import RiskManager



class Trading(ConnectMT5):
    
    def __init__(self,account=None, 
                 password=None, server=None, 
                 path_conect_data='data/input_data/private_data.json'):
        
        super().__init__(account, password, server, path_conect_data)
        self.risk_manager = RiskManager()
        

    def execute_trading(self, df, symbol, lot=None,  sl=None, tp=None,
                        deviation=10, close_previous_deal=True, 
                        order_type="market", order_offset=100 ):
        """
        Executes trading operations based on the provided DataFrame and trading parameters.

        Args:
        - df (DataFrame): DataFrame containing trading signals.
        - symbol (str): Trading symbol.
        - lot (float): Trading lot size.
        - sl (float): Stop loss level.
        - tp (float): Take profit level.
        - deviation (int): Maximum deviation allowed when executing a trade.
        - close_previous_deal (bool): Whether to close previous deals before executing a new one. Defaults to True.
        - order_type (str): Type of order to execute. Options are "market" or "limit". Defaults to "market".
        - order_offset (int): Number of shifts in points to adjust the price, stop loss, and take profit levels. Defaults to 100.

        Returns:
        - bool: True if the trading operation was successful, False otherwise.
        """

        try:
            # Get the latest signal from the DataFrame
            signal = df["Signal"].iloc[-1]
            
            if signal in ["buy", "sell"]:

                type_signal = {0: "buy", 1: "sell", 2: "buy", 3: "sell"}
                
                # Establish a connection to MetaTrader 5
                self._connect_to_mt5()
                
                # Get symbol, position and orders info  
                symbol_info = mt5.symbol_info(symbol)
                positions = mt5.positions_get(symbol=symbol)
                orders = mt5.orders_get(symbol=symbol)

                if close_previous_deal and (positions or orders):
                    trigger = False
                    
                    if positions:
                        trigger = type_signal[positions[0].type] != signal
                    if orders:
                        trigger = type_signal[orders[0].type] != signal if not trigger else trigger
                    if trigger: 
                        self.close(symbol=symbol, positions=positions, symbol_info=symbol_info,
                                    orders=orders , order_type=order_type )
                        
                if lot is None and sl is None and tp is None:
                    lot_size, stop_loss_price, take_profit_price = self.risk_manager.calculate_trade_parameters(symbol = symbol, signal=signal) 
                        
                lot = lot if lot else lot_size
                sl = sl if sl else stop_loss_price
                tp = tp if tp else take_profit_price
    
                # Create an order request based on the signal and other parameters
                if order_type == "market":
                    request = self.__market_order( signal=signal, symbol=symbol, symbol_info=symbol_info, 
                                           lot=lot, sl=sl, tp=tp, deviation=deviation)
                elif order_type == "limit":
                    request = self.__limit_order(symbol_info=symbol_info, signal=signal, symbol=symbol, 
                                               lot=lot, sl=sl, tp=tp, deviation=deviation, shifts= order_offset)
                    
                # Send the order request to execute the trade
                
                result = mt5.order_send(request)
                # Check the result of the order execution
                if result is not None:
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        # Print an error message if the order execution fails
                        print(f"2. order_send failed, retcode={result.retcode}, comment={result.comment}")
                        result_dict = result._asdict()
                        for field in result_dict.keys():
                            if field == "request":
                                traderequest_dict = result_dict[field]._asdict()
                                for tradereq_field in traderequest_dict:
                                    print(f"   traderequest: {tradereq_field}={traderequest_dict[tradereq_field]}")
                    else:
                        print("Order executed successfully.")
                else:
                    print(f"Order send failed result None.")
        
        except ValueError as e:
            # Handle any exceptions that may occur during the trading process
            print(f"execute_trading function error: {e}")
    
        finally:
            # Shut down the MetaTrader 5 connection
            mt5.shutdown()

        return True

    def close(self, symbol, symbol_info=None, positions=None, 
              orders=None, order_type="market"):
        """
        Closes positions and orders for the specified symbol.

        Args:
        - symbol (str): Trading symbol.
        - symbol_info (mt5 symbol info): Information about the trading symbol.
        - positions (list): List of open positions for the specified symbol.
        - orders (list): List of pending orders for the specified symbol.
        - order_type (str): Type of order to close. Options are "market" or "limit". Defaults to "market".
        
        Returns:
        - bool: True if the trading operation was successful, False otherwise.
        """
        
        if symbol_info :
                symbol_info = mt5.symbol_info(symbol)
        if positions :
            self.__close_positions(symbol, positions, symbol_info)
        if orders :
            self.__close_orders(symbol, orders )
            
        return True

    def __limit_order(self, symbol_info, signal, symbol, lot, sl, tp, deviation, shifts=100):
        """
        Private method to create a limit order request.

        Args:
        - symbol_info (mt5 symbol info): Information about the trading symbol.
        - signal (str): Trading signal ("buy" or "sell").
        - symbol (str): Trading symbol.
        - lot (float): Trading lot size.
        - sl (float): Stop loss level.
        - tp (float): Take profit level.
        - deviation (int): Maximum deviation allowed when executing a trade.
        - shifts (int): Number of shifts in points to adjust the price, stop loss, and take profit levels. Defaults to 100.

        Returns:
        - dict: Order request dictionary.
        """
        
        if isinstance(shifts, int ): 
            shifts = shifts * symbol_info.point
        try:
            self._connect_to_mt5()
            trade_type = mt5.ORDER_TYPE_BUY_LIMIT if signal == "buy" else mt5.ORDER_TYPE_SELL_LIMIT
            price = symbol_info.ask - shifts if signal == "buy" else symbol_info.bid + shifts
        finally:
            mt5.shutdown
            
        sl = sl - shifts if signal == "buy" else sl + shifts
        tp = tp - shifts if signal == "buy" else tp + shifts

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(lot),
            "type": trade_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": 234000,
            "comment": "Limit Python order",
            "type_time": mt5.ORDER_TIME_GTC,  
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        return request

    def __market_order(self, symbol_info, signal, symbol, lot, sl, tp, deviation):
        """
        Private method to create a market order request.

        Args:
        - symbol_info (mt5 symbol info): Information about the trading symbol.
        - signal (str): Trading signal ("buy" or "sell").
        - symbol (str): Trading symbol.
        - lot (float): Trading lot size.
        - sl (float): Stop loss level.
        - tp (float): Take profit level.
        - deviation (int): Maximum deviation allowed when executing a trade.

        Returns:
        - dict: Order request dictionary.
        """
        
        # Define the order parameters based on the given signal
        try:
            self._connect_to_mt5()
            # Define the order parameters based on the given signal
            price = mt5.symbol_info(symbol).ask if signal == "buy" else mt5.symbol_info(symbol).bid
            trade_type = mt5.ORDER_TYPE_BUY if signal == "buy" else mt5.ORDER_TYPE_SELL
        finally:
            mt5.shutdown

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot), 
            "type": trade_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": 234000,
            "comment": "Market Python order",
            "type_time": mt5.ORDER_TIME_DAY,
            "type_filling": mt5.TRADE_ACTION_DEAL,
            "request_actions": mt5.TRADE_ACTION_DEAL
        }
        
        return request

    def __close_positions(self, symbol, positions, symbol_info):
        """
        Private method to close positions for the specified symbol.

        Args:
        - symbol (str): Trading symbol.
        - positions (list): List of open positions for the specified symbol.
        - symbol_info (mt5 symbol info): Information about the trading symbol.
        """
        
        if positions:
            position = positions[0]
        else:
            return None

        if position.type == 0:
            type = mt5.ORDER_TYPE_SELL
            price = symbol_info.bid
        else:
            type = mt5.ORDER_TYPE_BUY
            price = symbol_info.ask
            
            
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL, 
            "position": position.ticket,
            "symbol": symbol,
            "volume": position.volume,
            "deviation": 10,
            "magic": position.magic,
            "price": price,
            "type": type,
            "comment": "python script close",
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_send(close_request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order closing failed: {result.retcode}")
        else:
            print("closed correctly")

    def __close_orders(self, symbol, orders ):
        """
        Private method to close orders for the specified symbol.

        Args:
        - symbol (str): Trading symbol.
        - orders (list): List of pending orders for the specified symbol.
        """
        
        for order in orders:
            if order.symbol == symbol:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                    "magic": 234000,
                    "comment": "Delete order"
                }
                
                result = mt5.order_send(request)
            
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Order {order.ticket} for symbol {symbol} was successfully deleted.")
                else:
                    print(f"Error when deleting order {order.ticket} by symbol {symbol}: {result.comment}")
    