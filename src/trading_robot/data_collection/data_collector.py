import pandas as pd
from trading_robot.connectors.docker_connector import mt5
from trading_robot.connectors.connect_mt5 import ConnectMT5
from trading_robot.utils.logger import log_message



class DataCollector(ConnectMT5):
    """
    DataCollector class extends ConnectMT5 to collect historical data and retrieve available symbols from MetaTrader 5.

    Parameters:
    - account (str): MT5 account number.
    - password (str): MT5 account password.
    - server (str): MT5 server address.

    Methods:
    - __init__: Class constructor. Inherits from ConnectMT5 constructor.
    - get_historical_data: Retrieves historical data for a specified symbol and time frame.
    - get_all_symbols: Gets a list of all available symbols in MetaTrader 5.
    """

    def __init__(self, account=None, password=None, server=None):
        super().__init__(account, password, server)

    def get_historical_data(self, symbol, timeframe=mt5.TIMEFRAME_D1, count=30_000):
        """
        Retrieves historical data from MetaTrader 5.

        Arguments:
        - symbol (str): The symbol for which you want to get historical data.
        - timeframe (int): Time frame for historical data.
        - count (int): Number of candles requested.

        Returns:
        - pd.DataFrame: DataFrame containing the received historical data.
        """

        if not self._connect_to_mt5():
            log_message("Failed to connect to MetaTrader 5. Please check the connection settings.")
            return None

        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                log_message(f"Received no data for symbol: {symbol}, timeframe: {timeframe}.")
                return None
        
        except Exception as e:
            log_message(f"Exception occurred while receiving data for symbol: {symbol}, timeframe: {timeframe}. Exception: {str(e)}")
            log_message(f"MetaTrader 5 last error: {mt5.last_error()}")
            return None

        finally:
            mt5.shutdown()

        if rates is not None and len(rates) > 0:
            log_message(f"Data retrieved successfully for symbol: {symbol}, timeframe: {timeframe}, count: {count}")
            rates_frame = pd.DataFrame(rates)
            rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
            rates_frame = rates_frame.rename(columns={"time": "Date", 'tick_volume': 'Volume'})
            rates_frame = rates_frame.rename(columns=lambda x: x.capitalize())
            rates_frame = rates_frame.set_index("Date")
            return rates_frame[['Open', 'Close', 'High', 'Low', 'Volume']]
        else:
            if rates is None:
                log_message(f"Failed to retrieve data: 'rates' is None for symbol: {symbol}, timeframe: {timeframe}")
            else:
                log_message(f"Failed to retrieve data: 'rates' is an empty list for symbol: {symbol}, timeframe: {timeframe}")

    def get_all_symbols(self):
        """
        Retrieves a list of all available symbols in MetaTrader 5.

        Returns:
        - list: List of symbol names, or None if an error occurred.
        """
        if not self._connect_to_mt5():
            log_message("Failed to connect to MetaTrader 5. Please check the connection settings.")
            return None

        try:
            symbols_info = mt5.symbols_get()
            symbols = [symbol.name for symbol in symbols_info]
            
            if not symbols:
                log_message("No symbols found in MetaTrader 5.")
                return None

            return symbols
        
        except Exception as e:
            log_message(f"Error occurred while retrieving the list of symbols: {e}")
            return None

        finally:
            mt5.shutdown()



if __name__ == "__main__":

    dc = DataCollector()

    data = dc.get_historical_data('EURUSD')
    print(data )

    # print(dc.get_all_symbols())

