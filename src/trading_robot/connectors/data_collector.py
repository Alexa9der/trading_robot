import pandas as pd
import MetaTrader5 as mt5

from trading_robot.connectors.connect_mt5 import ConnectMT5

class DataCollector(ConnectMT5):
    """
    DataCollector class extends ConnectMT5 to collect historical data and retrieve available symbols from MetaTrader 5.

    Parameters:
    - account (str): MT5 account number.
    - password (str): MT5 account password.
    - server (str): MT5 server address.
    - path_connect_data (str): Path to a JSON file containing connection data (optional).

    Methods:
    - __init__: Class constructor. Inherits from ConnectMT5 constructor.
    - get_historical_data: Retrieves historical data for a specified symbol and time frame.
    - get_all_symbols: Gets a list of all available symbols in MetaTrader 5.

    Usage:
    Create an instance of DataCollector by providing connection parameters or a path to a JSON file.
    Call get_historical_data() to retrieve historical data for a specific symbol.
    Call get_all_symbols() to get a list of all available symbols.
    """

    def __init__(self, account=None, password=None, server=None, 
                 path_conect_data='data/input_data/private_data.json'):
        super().__init__(account, password, server, path_conect_data)

    def get_historical_data(self, symbol, timeframe=mt5.TIMEFRAME_D1, count=30_000 ):
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
            print("Error connecting to MetaTrader 5.")
            return None
    
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        except Exception as e:
            print("initialize() failed, error code =",mt5.last_error())
            print(f"Function get_historical_data Error while receiving data: {e}")
            return None
        finally:
            mt5.shutdown()
    
        # Check the success of receiving data
        if rates is not None:
            # Convert data to pandas DataFrame
            rates_frame = pd.DataFrame(rates)
            rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
            rates_frame = rates_frame.rename(columns={"time": "Date", 'tick_volume':'volume'})
            rates_frame = rates_frame.rename(columns=lambda x: x.capitalize())
            
            return rates_frame[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']]
        else:
            print("Error while receiving data.")
            return None

    def get_all_symbols(self):
        """
         Gets a list of all available symbols in MetaTrader 5.
    
         Returns:
         list: List of characters.
         """
        # It is assumed that connect_to_mt5 is a function that connects to MetaTrader 5
        if not self._connect_to_mt5():
            print("Error connecting to MetaTrader 5.")
            return None
    
        try:
            symbols_info = mt5.symbols_get()
            symbols = [symbol.name for symbol in symbols_info]
            return symbols
        except Exception as e:
            print(f"Error when retrieving a list of symbols:{e}")
            return None
        finally:
            mt5.shutdown()

if __name__ == "__main__":

    dc = DataCollector()
    data = dc.get_historical_data('EURUSDz')
    print(data)