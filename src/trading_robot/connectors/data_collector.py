import sys 
import os 

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
import pandas as pd
from connectors.docker_connector import mt5
from connectors.connect_mt5 import ConnectMT5




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
            print("Error connecting to MetaTrader 5.")
            return None

        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        except Exception as e:
            print("Error while receiving data:", mt5.last_error())
            print(f"Function get_historical_data Error while receiving data: {e}")
            return None
        finally:
            mt5.shutdown()

        if rates is not None and len(rates) > 0:
            rates_frame = pd.DataFrame(rates)
            rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
            rates_frame = rates_frame.rename(columns={"time": "Date", 'tick_volume': 'Volume'})
            rates_frame = rates_frame.rename(columns=lambda x: x.capitalize())
            return rates_frame[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']]
        else:
            print("Error while receiving data.")
            return None

    def get_all_symbols(self):
        """
        Gets a list of all available symbols in MetaTrader 5.

        Returns:
        - list: List of characters.
        """
        if not self._connect_to_mt5():
            print("Error connecting to MetaTrader 5.")
            return None

        try:
            symbols_info = mt5.symbols_get()
            symbols = [symbol.name for symbol in symbols_info]
            return symbols
        except Exception as e:
            print(f"Error when retrieving a list of symbols: {e}")
            return None
        finally:
            mt5.shutdown()



if __name__ == "__main__":

    dc = DataCollector()
    data = dc.get_historical_data('EURUSD')
    print(data )
