import pandas as pd
import os

class TradingBot:
    """
    A simple trading bot that makes decisions based on predicted stock prices.

    This bot implements a threshold-based trading strategy based on the project
    proposal for the 'AI Share Trading Agent for JSE'. It reads a CSV file
    with actual and predicted prices, calculates the predicted return, and
    generates buy, sell, or hold signals. It also evaluates the strategy's
    performance by tracking profit and loss.
    """

    def __init__(self, initial_cash=10000.00, buy_threshold=0.015, sell_threshold=-0.01):
        """
        Initializes the TradingBot.

        Args:
            initial_cash (float): The starting cash balance for the portfolio.
            buy_threshold (float): The minimum predicted return to trigger a 'BUY' signal.
                                   (e.g., 0.015 corresponds to a 1.5% predicted increase).
            sell_threshold (float): The maximum predicted return to trigger a 'SELL' signal.
                                    (e.g., -0.01 corresponds to a 1% predicted decrease).
        """
        self.cash = initial_cash
        self.holdings = {}  # Stores the quantity of each stock owned
        self.initial_portfolio_value = initial_cash
        self.portfolio_value = initial_cash
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        print("--- Trading Bot Initialized ---")
        print(f"Initial Cash: ${self.cash:,.2f}")
        print(f"Buy Signal Threshold: > {self.buy_threshold:.2%}")
        print(f"Sell Signal Threshold: < {self.sell_threshold:.2%}")
        print("---------------------------------")


    def load_predictions(self, filepath="predictions.csv"):
        """
        Loads prediction data from a CSV file.

        The CSV file should have the following columns:
        - StockID: The ticker symbol for the stock (e.g., 'FSR.JO').
        - Actual_Price: The current or last known price of the stock.
        - Predicted_Price: The model's predicted price for the next period.

        Args:
            filepath (str): The path to the predictions CSV file.

        Returns:
            pandas.DataFrame: A DataFrame containing the prediction data,
                              or None if the file is not found.
        """
        print(f"\nLoading predictions from '{filepath}'...")
        if not os.path.exists(filepath):
            print(f"Error: The file '{filepath}' was not found.")
            print("Please create a 'predictions.csv' file in the same directory.")
            # Create a sample file for the user
            sample_data = {
                'StockID': ['FSR.JO', 'MTN.JO', 'SLM.JO', 'MRP.JO', 'NPN.JO'],
                'Actual_Price': [6483.029, 21702.17, 7786.933, 15981.29, 15000.00],
                'Predicted_Price': [6605.654, 21454.68, 8067.061, 15730.72, 15300.00]
            }
            pd.DataFrame(sample_data).to_csv(filepath, index=False)
            print(f"A sample '{filepath}' has been created for you.")

        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
            return None

    def generate_signals(self, predictions_df):
        """
        Generates trading signals based on the loaded predictions.

        This method implements the core trading logic defined in the project proposal.

        Args:
            predictions_df (pandas.DataFrame): DataFrame with prediction data.

        Returns:
            pandas.DataFrame: The input DataFrame with two new columns:
                              'Predicted_Return' and 'Signal'.
        """
        if predictions_df is None:
            return None

        print("\nGenerating trading signals...")
        # Calculate the predicted return
        # Formula: (Predicted_Price / Actual_Price) - 1
        predictions_df['Predicted_Return'] = (predictions_df['Predicted_Price'] / predictions_df['Actual_Price']) - 1

        # Generate signals based on thresholds
        signals = []
        for index, row in predictions_df.iterrows():
            if row['Predicted_Return'] >= self.buy_threshold:
                signals.append('BUY')
            elif row['Predicted_Return'] <= self.sell_threshold:
                signals.append('SELL')
            else:
                signals.append('HOLD')

        predictions_df['Signal'] = signals
        print("Signals generated successfully.")
        return predictions_df

    def execute_trades(self, signals_df, trade_amount_per_stock=10000.0):
        """
        Executes trades based on the generated signals and updates the portfolio.

        Args:
            signals_df (pandas.DataFrame): DataFrame with trading signals.
            trade_amount_per_stock (float): The amount of cash to use for each buy trade.
        """
        if signals_df is None:
            return

        print("\n--- Executing Trades ---")
        for index, row in signals_df.iterrows():
            stock_id = row['StockID']
            signal = row['Signal']
            current_price = row['Actual_Price']

            if signal == 'BUY':
                if self.cash >= trade_amount_per_stock:
                    quantity_to_buy = trade_amount_per_stock / current_price
                    self.cash -= trade_amount_per_stock
                    self.holdings[stock_id] = self.holdings.get(stock_id, 0) + quantity_to_buy
                    print(f"BOUGHT:  {quantity_to_buy:.4f} shares of {stock_id} @ ${current_price:,.2f}")
                else:
                    print(f"HOLD (Insufficient Cash): Tried to buy {stock_id}, but not enough cash.")

            elif signal == 'SELL':
                if stock_id in self.holdings and self.holdings[stock_id] > 0:
                    quantity_to_sell = self.holdings[stock_id] # Sell all holdings of this stock
                    self.cash += quantity_to_sell * current_price
                    del self.holdings[stock_id]
                    print(f"SOLD:    {quantity_to_sell:.4f} shares of {stock_id} @ ${current_price:,.2f}")
                else:
                     print(f"HOLD (No Holdings): Tried to sell {stock_id}, but none in portfolio.")
            else: # HOLD
                print(f"HOLD:    {stock_id}")
        print("--------------------------")


    def print_portfolio_status(self, predictions_df):
        """
        Prints the current status of the portfolio, including profit/loss.

        Args:
            predictions_df (pandas.DataFrame): The DataFrame with current prices to value the holdings.
        """
        print("\n--- Portfolio Status & Performance ---")
        holdings_value = 0
        if self.holdings:
            print("Holdings:")
            # Create a dictionary for quick price lookups
            price_map = pd.Series(predictions_df.Actual_Price.values, index=predictions_df.StockID).to_dict()
            for stock, quantity in self.holdings.items():
                current_price = price_map.get(stock, 0)
                value = quantity * current_price
                holdings_value += value
                print(f"  - {stock}: {quantity:.4f} shares, Value: ${value:,.2f}")
        else:
            print("Holdings: No stocks currently held.")

        # --- Profit/Loss Calculation ---
        self.portfolio_value = self.cash + holdings_value
        profit_loss = self.portfolio_value - self.initial_portfolio_value
        profit_loss_percent = (profit_loss / self.initial_portfolio_value) * 100 if self.initial_portfolio_value != 0 else 0

        print(f"\nCash Balance:          ${self.cash:,.2f}")
        print(f"Holdings Value:        ${holdings_value:,.2f}")
        print(f"Total Portfolio Value:   ${self.portfolio_value:,.2f}")
        print("--------------------------------------")
        print(f"Profit/Loss:           ${profit_loss:,.2f} ({profit_loss_percent:.2f}%)")
        print("--------------------------------------\n")


if __name__ == "__main__":
    # 1. Initialize the bot with custom thresholds if desired
    # Example: More aggressive strategy
    # trading_bot = TradingBot(buy_threshold=0.01, sell_threshold=-0.005)
    trading_bot = TradingBot()

    # 2. Load the prediction data from the CSV file
    predictions_data = trading_bot.load_predictions("evaluation_results_POINT_TCN.csv")

    if predictions_data is not None:
        # 3. Generate trading signals
        signals_data = trading_bot.generate_signals(predictions_data)
        print("\n--- Analysis Results ---")
        print(signals_data.to_string())
        print("------------------------")


        # 4. Execute trades based on the signals
        trading_bot.execute_trades(signals_data)

        # 5. Print the final portfolio status and performance
        trading_bot.print_portfolio_status(signals_data)
