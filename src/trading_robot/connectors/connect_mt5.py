from trading_robot.connectors.docker_connector import mt5
import pandas as pd
import os 


class ConnectMT5:
    """
    ConnectMT5 class provides methods to establish a connection to MetaTrader 5 (MT5) server.

    Parameters:
    - account (str): MT5 account number.
    - password (str): MT5 account password.
    - server (str): MT5 server address.
    - path_connect_data (str): Path to a JSON file containing connection data (optional).

    Methods:
    - __init__: Class constructor. Initializes the connection parameters either through provided arguments or from a JSON file.
    - __get_connect_data: Private method to extract connection data from a JSON file.
    - _connect_to_mt5: Protected method to connect to the MT5 server using the provided or loaded credentials.

    Usage:
    Initialize the class with account, password, and server parameters or provide the path to a JSON file containing the data.
    Call _connect_to_mt5() method to establish a connection to the MT5 server.
    """
    
    def __init__(self, account=None, password=None, server=None):
        try:
            if account and password and server:
                self.__account = int(account)
                self.__password = password
                self.__server = server
            else:
                self.__account, self.__password, self.__server = self.__get_connect_data()
        except Exception as e:
            print("ConnectMT5: Error setting up data to connect to the MT5 server.")
            raise e
    
    def __get_connect_data(self):
        """
        Private method to extract connection data from environment variables.

        Returns:
        - Tuple (str, str, str): Account, password, and server extracted from environment variables.
        """
        account = os.getenv('MT5_ACCOUNT')
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')

        if not account or not password or not server:
            raise ValueError("Missing connection data from environment variables")

        return account, password, server

    def _connect_to_mt5(self):
        """
        Protected method to connect to the MT5 server using the provided or loaded credentials.

        Returns:
        - bool: True if connection is successful, False otherwise.
        """
        if not mt5.initialize():
            print("Initialization error")
            mt5.shutdown()
            return False

        authorized = mt5.login(login=int(self.__account), 
                            password=self.__password, 
                            server=self.__server)

        if authorized:
            return True
        else:
            print("Failed to connect. Error code:", mt5.last_error())
            mt5.shutdown()
            return False
        

    