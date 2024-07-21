import pandas as pd 

class Metrics:
    
    def _calmar_ratio(self, df):
        """
        Calculate the average Calmar Ratio for a given DataFrame.
    
        Args:
            df (pd.DataFrame): The input DataFrame containing trading signals and accumulated price changes.
    
        Returns:
            float: The average Calmar Ratio.
    
        This function calculates the average Calmar Ratio for a given DataFrame. It considers the average return
        divided by the maximum drawdown as a measure of strategy performance.
    
        Note: The average Calmar Ratio provides insights into the average performance of the strategy over time.
    
        """
        max_drawdown = df["AccumulatedPriceChange"].min()
    
        if max_drawdown != 0:
            # Calculate the sum of accumulated price changes for sell signals
            result = df.loc[(df["Signal"] == "sell") | (df["Signal"] == "sell"),
                            "AccumulatedPriceChange"].sum()
    
            # Count the number of trades
            counting_trades = len(df.loc[df["AccumulatedPriceChange"] != 0])
    
            if result != 0:
                # Calculate the average return and Calmar Ratio
                average_return = result / counting_trades
                calmar_ratio = average_return / max_drawdown
    
                return calmar_ratio
            else:
                return 0
        else:
            return 0

    def _sharpe_ratio(self, df):
        """
        Calculate the Sharpe Ratio for a given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing trading signals and accumulated price changes.

        Returns:
            float: The Sharpe Ratio.

        This function calculates the Sharpe Ratio for a given DataFrame. It considers the average return
        divided by the standard deviation of the return as a measure of strategy performance.

        """
        # Assuming you have a column 'AccumulatedPriceChange' in your DataFrame
        returns = df['AccumulatedPriceChange']

        # Calculate the average return and standard deviation of the return
        average_return = returns.mean()
        std_dev_return = returns.std()

        # Calculate the Sharpe Ratio
        sharpe_ratio = average_return / std_dev_return if std_dev_return != 0 else 0

        return sharpe_ratio

    def _sortino_ratio(self, df):
        """
        Calculate the Sortino Ratio for a given DataFrame.
    
        Args:
            df (pd.DataFrame): The input DataFrame containing trading signals and accumulated price changes.
    
        Returns:
            float: The Sortino Ratio.
    
        This function calculates the Sortino Ratio for a given DataFrame. It considers the average return
        divided by the standard deviation of negative returns as a measure of strategy performance.
    
        Note: The Sortino Ratio focuses on the risk associated with negative returns, providing insights into the
        strategy's performance in the presence of downside risk.
    
        """
        accumulated_price_change = df.loc[df["AccumulatedPriceChange"] != 0] 
        sum_price_change = accumulated_price_change["AccumulatedPriceChange"].sum() 
        
        if sum_price_change != 0:
            average_return = sum_price_change / len(accumulated_price_change)
        else :
            return 0
    
        # Filter negative returns
        negative_returns = df.loc[df["AccumulatedPriceChange"] < 0, "AccumulatedPriceChange"]
    
        # Calculate the standard deviation of negative returns
        std_dev_negative_returns = negative_returns.std()
    
        if std_dev_negative_returns != 0 :
            # Calculate the Sortino Ratio
            sortino_ratio = average_return / std_dev_negative_returns
            return sortino_ratio
        else:
            return 0