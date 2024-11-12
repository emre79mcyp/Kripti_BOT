# import time
# import logging
# import hmac
# import hashlib
# import requests
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# import ta

# # Constants
# API_KEY = 'xphI8ywTK6IIyyLXhvoSLF8MEWbGPbkC0UFtp7wBRrDvp8UkJMtNdl5ifj28L2Lp'
# SECRET_KEY = 'ykGSrhFx79Wb5tFZ8oJaZ15Xqb7Kk65IaxZ4so7K6IkWWGvfKw8Hjoc9J1wJ6PGh'
# BASE_URL = 'https://api.binance.com/api/v3'
# SYMBOL = 'BTCUSDT'
# INTERVAL = '1m'
# MIN_TRADE_AMOUNT = 0.0001
# PRICE_PRECISION = 2
# QUANTITY_PRECISION = 5

# # Trading fees and slippage
# FEE_RATE = 0.001
# SLIPPAGE = 0.001

# # Risk management
# STOP_LOSS_PERCENT = 0.01
# TAKE_PROFIT_PERCENT = 0.02

# # Trade frequency control
# MAX_TRADES_PER_HOUR = 30
# COOLDOWN_PERIOD = 120

# # Model retraining interval
# RETRAIN_INTERVAL = timedelta(hours=1)

# # Logging configuration
# logging.basicConfig(filename='trading_bot.log', level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# def get_signed_params(params):
#     query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
#     signature = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
#     return query_string, signature

# def get_server_time():
#     response = requests.get(f"{BASE_URL}/time")
#     return response.json()['serverTime']

# def get_historical_data(symbol, interval, limit=100):
#     params = {'symbol': symbol, 'interval': interval, 'limit': limit}
#     response = requests.get(f"{BASE_URL}/klines", params=params)
#     data = response.json()
#     df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#     df.set_index('timestamp', inplace=True)
#     for col in ['open', 'high', 'low', 'close', 'volume']:
#         df[col] = df[col].astype(float)
#     return df

# def add_indicators(df):
#     df['rsi'] = ta.momentum.rsi(df['close'], window=14)
#     df['macd'] = ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9)
#     df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=5)
#     df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=10)
#     df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
#     df['volatility'] = df['atr'] / df['close']  # ATR-based volatility
#     df['volume_change'] = df['volume'].pct_change()
#     return df

# def create_model(df):
#     feature_names = ['rsi', 'macd', 'ema_fast', 'ema_slow', 'volatility', 'volume_change']
#     X = df[feature_names]
#     y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    
#     # Drop the last row of X and y since we can't calculate the target for it
#     X = X[:-1]
#     y = y[:-1]

#     # Create a pipeline that first imputes missing values, then fits the model
#     model = Pipeline([
#         ('imputer', SimpleImputer(strategy='mean')),
#         ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0))
#     ])

#     model.fit(X, y)
#     return model, feature_names

# def generate_signal(model, row, feature_names):
#     # Prepare the data for prediction
#     model_input = row[feature_names].to_frame().T 
    
#     # Convert the input to a DataFrame to ensure valid feature names
#     model_input_df = pd.DataFrame(model_input, columns=feature_names)
    
#     # Impute missing values and make predictions
#     model_input_imputed = model.named_steps['imputer'].transform(model_input_df)
#     return model.predict(model_input)[0]

# def place_order(symbol, side, quantity):
#     for _ in range(3):  # Retry up to 3 times
#         try:
#             params = {
#                 'symbol': symbol,
#                 'side': side,
#                 'type': 'MARKET',
#                 'quantity': f"{quantity:.{QUANTITY_PRECISION}f}",
#                 'timestamp': get_server_time()
#             }
#             query_string, signature = get_signed_params(params)
#             url = f"{BASE_URL}/order?{query_string}&signature={signature}"
#             response = requests.post(url, headers={'X-MBX-APIKEY': API_KEY})
#             return response.json()
#         except requests.exceptions.RequestException as e:
#             logging.error(f"Error placing order: {e}")
#             time.sleep(2)  # Wait before retrying
#     return None

# def calculate_slippage_and_fee(quantity, price):
#     slippage_amount = price * SLIPPAGE
#     fee_amount = (quantity * price) * FEE_RATE
#     return slippage_amount, fee_amount

# def dynamic_position_sizing(balance, volatility):
#     risk_per_trade = 0.01  # Risk 1% of balance per trade
#     risk_dollars = balance * risk_per_trade
#     position_size = risk_dollars / (volatility * balance)  # Smaller position size for higher volatility
#     return max(MIN_TRADE_AMOUNT, min(position_size, balance / 10))  # Limit position size

# def sentiment_analysis():
#     # Placeholder for sentiment analysis (e.g., from social media, news)
#     # For now, we assume neutral sentiment with no impact
#     return 0

# def order_book_imbalance():
#     # Placeholder for order book imbalance analysis
#     # For now, we assume neutral order book with no impact
#     return 0

# def adjust_signal_with_advanced_analysis(signal):
#     sentiment = sentiment_analysis()
#     order_imbalance = order_book_imbalance()
#     # Example: Adjust signal based on sentiment and order imbalance
#     if sentiment > 0.5 or order_imbalance > 0.5:
#         signal = 1  # Bias towards buy
#     elif sentiment < -0.5 or order_imbalance < -0.5:
#         signal = 0  # Bias towards sell
#     return signal

# def backtest_strategy(df, model, feature_names):
#     initial_balance = 1000
#     balance = initial_balance
#     position = 0
#     entry_price = 0
#     total_profit = 0
#     trades_count = 0
#     winning_trades = 0

#     for index, row in df.iterrows():
#         signal = generate_signal(model, row, feature_names)
        
#         current_price = row['close']
#         volatility = row['volatility']

#         if position == 0 and signal == 1:  # Buy signal
#             position = balance / current_price
#             entry_price = current_price
#             balance = 0  # All balance is now in the position
#             trades_count += 1
        
#         elif position > 0:  # Evaluate whether to sell
#             profit = (current_price - entry_price) * position
#             if signal == 0 or profit / entry_price >= TAKE_PROFIT_PERCENT or profit / entry_price <= -STOP_LOSS_PERCENT:
#                 balance = position * current_price
#                 total_profit += profit
#                 position = 0
#                 if profit > 0:
#                     winning_trades += 1
        
#     final_balance = balance + (position * df.iloc[-1]['close'] if position > 0 else 0)
#     total_return = (final_balance - initial_balance) / initial_balance * 100
#     win_rate = (winning_trades / trades_count * 100) if trades_count > 0 else 0

#     return {
#         'final_balance': final_balance,
#         'total_profit': total_profit,
#         'total_return': total_return,
#         'trades_count': trades_count,
#         'winning_trades': winning_trades,
#         'win_rate': win_rate
#     }

# def trading_loop(symbol, interval, model, feature_names):
#     balance = 1000  # Starting balance
#     position = 0
#     last_trade_time = time.time() - COOLDOWN_PERIOD

#     try:
#         while True:
#             try:
#                 df = get_historical_data(symbol, interval)
#                 df = add_indicators(df)
#                 current_features = df.iloc[-1]
#                 signal = generate_signal(model, current_features, feature_names)
#                 current_price = current_features['close']

#                 if position == 0 and signal == 1:  # Buy signal
#                     position = balance / current_price
#                     entry_price = current_price
#                     balance = 0  # All balance is now in the position
#                     last_trade_time = time.time()
#                     logging.info(f"Bought {position} {symbol} at {current_price}")

#                 elif position > 0:  # Evaluate whether to sell
#                     profit = (current_price - entry_price) * position
#                     if signal == 0 or profit / entry_price >= TAKE_PROFIT_PERCENT or profit / entry_price <= -STOP_LOSS_PERCENT:
#                         balance = position * current_price
#                         logging.info(f"Sold {position} {symbol} at {current_price}")
#                         logging.info(f"Profit: {profit:.2f} USDT")
#                         position = 0
#                         last_trade_time = time.time()

#                 # Monitor performance
#                 logging.info(f"Current balance: {balance:.2f} USDT")
                
#                 time.sleep(60)  # Sleep for the interval period (1 minute)

#             except Exception as e:
#                 logging.error(f"Error in trading loop: {e}")
#                 time.sleep(60)  # Sleep and try again

#     except KeyboardInterrupt:
#         logging.info("Trading bot stopped by user.")
#         # Perform any necessary cleanup here

# def main():
#     logging.info("Fetching historical data for backtesting...")
#     df = get_historical_data(SYMBOL, INTERVAL, limit=1000)  # Increased limit for backtesting
#     df = add_indicators(df)

#     logging.info("Creating model...")
#     model, feature_names = create_model(df)

#     logging.info("Running backtest...")
#     backtest_results = backtest_strategy(df, model, feature_names)

#     logging.info("Backtest Results:")
#     for key, value in backtest_results.items():
#         logging.info(f"{key}: {value:.2f}")

#     avg_profit_per_trade = backtest_results['total_profit'] / backtest_results['trades_count'] if backtest_results['trades_count'] > 0 else 0
#     logging.info(f"Average Profit per Trade: {avg_profit_per_trade:.4f} USDT")

#     logging.info("Starting live trading...")
#     trading_loop(SYMBOL, INTERVAL, model, feature_names)

# if __name__ == "__main__":
#     main()
































import time
import requests
import hmac
import hashlib
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle
import os

API_KEY = 'xphI8ywTK6IIyyLXhvoSLF8MEWbGPbkC0UFtp7wBRrDvp8UkJMtNdl5ifj28L2Lp'
SECRET_KEY = 'ykGSrhFx79Wb5tFZ8oJaZ15Xqb7Kk65IaxZ4so7K6IkWWGvfKw8Hjoc9J1wJ6PGh'
BASE_URL = 'https://api.binance.com/api/v3'

class ImprovedAIPoweredTradingBot:
    def __init__(self, symbol, time_frame, risk_percentage, stop_loss_pct, take_profit_pct, live_trading=False):
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY
        self.base_url = BASE_URL
        self.symbol = symbol
        self.time_frame = time_frame
        self.risk_percentage = risk_percentage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.live_trading = live_trading
        self.logger = self.setup_logger()
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'total_loss': 0,
        }
        self.max_trade_size = 1000  # Set maximum trade size
        self.max_total_loss = 5000  # Set maximum total loss
        self.recent_predictions = []
        self.recent_outcomes = []
        self.performance_threshold = 0.55  # Threshold for model evaluation
        self.load_state()

    def setup_logger(self):
        logger = logging.getLogger('TradingBot')
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler('trading_bot.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def sign_request(self, params):
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        signature = hmac.new(self.secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        return signature

    def get_server_time(self):
        response = requests.get(f"{self.base_url}/time")
        server_time = response.json()['serverTime']
        return server_time

    def make_api_request(self, endpoint, params=None, headers=None, method='GET'):
        self.logger.debug(f"Making API request: {method} {endpoint}")
        self.logger.debug(f"Params: {params}")
        self.logger.debug(f"Headers: {headers}")
        try:
            if method == 'GET':
                response = requests.get(endpoint, params=params, headers=headers)
            elif method == 'POST':
                response = requests.post(endpoint, params=params, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise

    def validate_api_keys(self):
        try:
            endpoint = f"{self.base_url}/account"
            timestamp = self.get_server_time()
            params = {'timestamp': timestamp}
            params['signature'] = self.sign_request(params)
            response = self.make_api_request(endpoint, params=params, headers={'X-MBX-APIKEY': self.api_key})
            self.logger.info("API keys are valid.")
        except Exception as e:
            self.logger.error(f"API key validation failed: {str(e)}")
            raise

    def check_account_balance(self):
        try:
            endpoint = f"{self.base_url}/account"
            timestamp = self.get_server_time()
            params = {'timestamp': timestamp}
            params['signature'] = self.sign_request(params)
            response = self.make_api_request(endpoint, params=params, headers={'X-MBX-APIKEY': self.api_key})
            balances = response['balances']
            usdt_balance = next(balance['free'] for balance in balances if balance['asset'] == 'USDT')
            return float(usdt_balance)
        except Exception as e:
            self.logger.error(f"Failed to fetch account balance: {str(e)}")
            return None

    def fetch_ohlcv(self):
        endpoint = f"{self.base_url}/klines"
        params = {
            'symbol': self.symbol,
            'interval': self.time_frame,
            'limit': 1000
        }
        headers = {'X-MBX-APIKEY': self.api_key}

        try:
            data = self.make_api_request(endpoint, params=params, headers=headers)
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                                             'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                                             'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch OHLCV data: {str(e)}")
            return None

    def calculate_indicators(self, df):
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        df['RSI'] = self.calculate_rsi(df['close'])
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = self.calculate_macd(df['close'])
        df['ATR'] = self.calculate_atr(df)
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bollinger_bands(df['close'])
        return df

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, series, fast_period=12, slow_period=26, signal_period=9):
        fast_ema = series.ewm(span=fast_period, min_periods=fast_period).mean()
        slow_ema = series.ewm(span=slow_period, min_periods=slow_period).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period, min_periods=signal_period).mean()
        macd_hist = macd - signal
        return macd, signal, macd_hist

    def calculate_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = high_low.combine(high_close, max).combine(low_close, max)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_bollinger_bands(self, series, window=20, num_std=2):
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band

    def generate_ai_signal(self, df):
        latest_data = df.iloc[-1][['SMA_50', 'SMA_200', 'RSI', 'MACD_hist', 'ATR', 'BB_upper', 'BB_lower']].values
        features = np.array(latest_data).reshape(1, -1)
        signal = self.model.predict(features)[0]
        return signal

    def analyze_multiple_timeframes(self):
        timeframes = ['15m', '1h', '4h']
        signals = []
        for tf in timeframes:
            self.time_frame = tf
            df = self.fetch_ohlcv()
            if df is not None:
                df = self.calculate_indicators(df)
                signals.append(self.generate_ai_signal(df))
        return sum(signals)  # Aggregate signals

    def place_order(self, side, quantity):
        try:
            endpoint = f"{self.base_url}/order"
            timestamp = self.get_server_time()
            params = {
                'symbol': self.symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': quantity,
                'timestamp': timestamp
            }
            params['signature'] = self.sign_request(params)
            order = self.make_api_request(endpoint, params=params, headers={'X-MBX-APIKEY': self.api_key}, method='POST')
            self.logger.info(f"Order placed: {order}")
            return order
        except Exception as e:
            self.logger.error(f"Failed to place order: {str(e)}")
            return None

    def fetch_current_price(self, symbol):
        endpoint = f"{self.base_url}/ticker/price"
        params = {'symbol': symbol}
        data = self.make_api_request(endpoint, params=params)
        return float(data['price'])

    def execute_trade(self, signal, capital, current_price, atr):
        position_size = self.calculate_position_size(capital, current_price, atr)
        stop_loss = current_price - (atr * self.stop_loss_pct) if signal == 1 else current_price + (atr * self.stop_loss_pct)
        take_profit = current_price + (atr * self.take_profit_pct) if signal == 1 else current_price - (atr * self.take_profit_pct)

        if self.check_risk_limits(position_size):
            if self.live_trading:
                side = 'BUY' if signal == 1 else 'SELL'
                order = self.place_order(side, position_size)
                if order:
                    self.monitor_open_trades()
                    return 0  # In live trading, the profit/loss is realized later
            else:
                trade_result = self.simulate_trade(signal, current_price, stop_loss, take_profit)
                return trade_result * position_size
        return 0

    def simulate_trade(self, signal, current_price, stop_loss, take_profit):
        if signal == 1:  # Buy
            trade_result = min(take_profit - current_price, current_price - stop_loss)
        else:  # Sell
            trade_result = min(current_price - take_profit, stop_loss - current_price)
        return trade_result

    def monitor_open_trades(self):
        open_orders = self.fetch_open_orders()
        for order in open_orders:
            current_price = self.fetch_current_price(order['symbol'])
            if (order['side'] == 'BUY' and current_price <= order['stop_loss']) or \
               (order['side'] == 'SELL' and current_price >= order['stop_loss']) or \
               (order['side'] == 'BUY' and current_price >= order['take_profit']) or \
               (order['side'] == 'SELL' and current_price <= order['take_profit']):
                self.close_trade(order['orderId'])
                self.logger.info(f"Closed trade {order['orderId']} at price {current_price}")

    def fetch_open_orders(self):
        endpoint = f"{self.base_url}/openOrders"
        timestamp = self.get_server_time()
        params = {'timestamp': timestamp}
        params['signature'] = self.sign_request(params)
        return self.make_api_request(endpoint, params=params, headers={'X-MBX-APIKEY': self.api_key})

    def close_trade(self, order_id):
        endpoint = f"{self.base_url}/order"
        timestamp = self.get_server_time()
        params = {
            'symbol': self.symbol,
            'orderId': order_id,
            'timestamp': timestamp
        }
        params['signature'] = self.sign_request(params)
        return self.make_api_request(endpoint, params=params, headers={'X-MBX-APIKEY': self.api_key}, method='DELETE')

    def calculate_position_size(self, capital, current_price, atr):
        risk_amount = capital * self.risk_percentage
        position_size = risk_amount / (atr * self.stop_loss_pct)
        return min(position_size, self.max_trade_size)

    def check_risk_limits(self, trade_size):
        if trade_size > self.max_trade_size or \
           self.performance_metrics['total_loss'] + trade_size > self.max_total_loss:
            self.logger.warning("Trade size exceeds risk limits. Trade not placed.")
            return False
        return True

    def update_performance_metrics(self, trade_result):
        self.performance_metrics['total_trades'] += 1
        if trade_result > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
            self.performance_metrics['total_loss'] += abs(trade_result)
        self.performance_metrics['total_profit'] += trade_result

    def update_recent_predictions(self, prediction, outcome):
        self.recent_predictions.append(prediction)
        self.recent_outcomes.append(outcome)
        if len(self.recent_predictions) > 100:  # Keep only the last 100 predictions
            self.recent_predictions.pop(0)
            self.recent_outcomes.pop(0)

    def calculate_recent_performance(self):
        if not self.recent_outcomes:
            return 0
        recent_returns = self.recent_outcomes[-20:]  # Consider last 20 trades
        return np.mean(recent_returns)

    def log_performance_metrics(self):
        win_rate = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades'] if self.performance_metrics['total_trades'] > 0 else 0
        avg_profit = self.performance_metrics['total_profit'] / self.performance_metrics['total_trades'] if self.performance_metrics['total_trades'] > 0 else 0
        
        self.logger.info(f"Win Rate: {win_rate:.2%}")
        self.logger.info(f"Average Profit per Trade: {avg_profit:.2f}")
        self.logger.info(f"Sharpe Ratio: {self.calculate_sharpe_ratio():.2f}")
        self.logger.info(f"Max Drawdown: {self.calculate_max_drawdown():.2%}")

    def calculate_sharpe_ratio(self):
        returns = pd.Series(self.recent_outcomes)
        if len(returns) < 2 or returns.std() == 0:
            return 0
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        return sharpe_ratio

    def calculate_max_drawdown(self):
        returns = pd.Series(self.recent_outcomes)
        if len(returns) == 0:
            return 0
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        return drawdown.min()

    def save_state(self):
        state = {
            'performance_metrics': self.performance_metrics,
            'recent_predictions': self.recent_predictions,
            'recent_outcomes': self.recent_outcomes,
            'model': self.model
        }
        with open('bot_state.pkl', 'wb') as f:
            pickle.dump(state, f)

    def load_state(self):
        if os.path.exists('bot_state.pkl'):
            with open('bot_state.pkl', 'rb') as f:
                state = pickle.load(f)
            self.performance_metrics = state['performance_metrics']
            self.recent_predictions = state['recent_predictions']
            self.recent_outcomes = state['recent_outcomes']
            self.model = state['model']
        else:
            self.logger.info("No previous state found. Starting fresh.")

    def run(self, capital):
        self.validate_api_keys()
        if self.live_trading:
            capital = self.check_account_balance()
            if capital is None:
                self.logger.error("Unable to fetch account balance. Exiting.")
                return

        while True:
            try:
                df = self.fetch_ohlcv()
                if df is not None:
                    df = self.calculate_indicators(df)
                    
                    if self.performance_metrics['total_trades'] % 100 == 0:
                        self.evaluate_model(df)
                        self.adjust_hyperparameters()

                    signal = self.analyze_multiple_timeframes()
                    if signal != 0:
                        latest_data = df.iloc[-1]
                        trade_result = self.execute_trade(signal, capital, latest_data['close'], latest_data['ATR'])
                        self.update_performance_metrics(trade_result)
                        self.update_recent_predictions(signal, 1 if trade_result > 0 else -1)
                    
                    self.log_performance_metrics()
                    self.calculate_advanced_metrics()
                    self.save_state()

                time.sleep(3600)  # Wait for 1 hour before the next iteration
            except Exception as e:
                self.logger.error(f"An error occurred: {str(e)}")
                time.sleep(60)  # Wait for 1 minute before retrying

    def evaluate_model(self, df):
        if len(self.recent_outcomes) < 100:
            self.logger.info("Not enough data to evaluate model. Continuing with current model.")
            return

        accuracy = accuracy_score(self.recent_outcomes, self.recent_predictions)
        if accuracy < self.performance_threshold:
            self.logger.info(f"Model accuracy ({accuracy:.2%}) below threshold. Retraining model.")
            self.retrain_model(df)

    def retrain_model(self, df):
        features = df[['SMA_50', 'SMA_200', 'RSI', 'MACD_hist', 'ATR', 'BB_upper', 'BB_lower']].values
        target = np.where(df['close'].shift(-1) > df['close'], 1, -1)
        target = target[:-1]  # Remove last row as we don't have a target for it
        features = features[:-1]  # Align features with target

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        self.model = self.optimize_hyperparameters(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.logger.info(f"Model retrained. New accuracy: {accuracy:.2%}")

    def optimize_hyperparameters(self, X, y):
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5]
        }
        grid_search = GridSearchCV(estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                                   param_grid=param_grid,
                                   cv=3,
                                   n_jobs=-1,
                                   verbose=2)
        grid_search.fit(X, y)
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def adjust_hyperparameters(self):
        recent_performance = self.calculate_recent_performance()
        if recent_performance < self.performance_threshold:
            self.logger.info("Recent performance below threshold. Adjusting strategy...")
            self.risk_percentage *= 0.9  # Reduce risk
            self.stop_loss_pct *= 1.1  # Widen stop loss
            self.take_profit_pct *= 1.1  # Increase take profit target
        else:
            self.risk_percentage = min(self.risk_percentage * 1.1, 0.02)  # Cap at 2%
            self.stop_loss_pct = max(self.stop_loss_pct * 0.9, 0.005)  # Floor at 0.5%
            self.take_profit_pct = max(self.take_profit_pct * 0.9, 0.01)  # Floor at 1%

    def detect_market_regime(self, df):
        df['50MA'] = df['close'].rolling(window=50).mean()
        df['200MA'] = df['close'].rolling(window=200).mean()
        
        if df['50MA'].iloc[-1] > df['200MA'].iloc[-1] and df['50MA'].iloc[-2] <= df['200MA'].iloc[-2]:
            return 'Bullish Crossover'
        elif df['50MA'].iloc[-1] < df['200MA'].iloc[-1] and df['50MA'].iloc[-2] >= df['200MA'].iloc[-2]:
            return 'Bearish Crossover'
        elif df['50MA'].iloc[-1] > df['200MA'].iloc[-1]:
            return 'Bullish Trend'
        else:
            return 'Bearish Trend'

    def calculate_advanced_metrics(self):
        returns = pd.Series(self.recent_outcomes)
        if len(returns) == 0:
            self.logger.info("No trades yet. Unable to calculate advanced metrics.")
            return

        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility  # Assuming 2% risk-free rate
        sortino_ratio = (annualized_return - 0.02) / (returns[returns < 0].std() * np.sqrt(252))
        max_drawdown = self.calculate_max_drawdown()
        
        self.logger.info(f"Total Return: {total_return:.2%}")
        self.logger.info(f"Annualized Return: {annualized_return:.2%}")
        self.logger.info(f"Annualized Volatility: {volatility:.2%}")
        self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        self.logger.info(f"Sortino Ratio: {sortino_ratio:.2f}")
        self.logger.info(f"Max Drawdown: {max_drawdown:.2%}")

    def backtest(self, start_date, end_date, initial_capital):
        historical_data = self.fetch_historical_data(start_date, end_date)
        balance = initial_capital
        trades = []
        for i in range(len(historical_data)):
            df = historical_data.iloc[:i+1]
            df = self.calculate_indicators(df)
            if len(df) > 200:
                signal = self.generate_ai_signal(df)
                if signal != 0:
                    trade_result = self.execute_trade(signal, balance, df.iloc[-1]['close'], df.iloc[-1]['ATR'])
                    balance += trade_result
                    trades.append({
                        'date': df.index[-1],
                        'signal': signal,
                        'price': df.iloc[-1]['close'],
                        'result': trade_result,
                        'balance': balance
                    })
        return pd.DataFrame(trades)

    def plot_backtest_results(self, backtest_results):
        plt.figure(figsize=(12, 6))
        plt.plot(backtest_results['date'], backtest_results['balance'])
        plt.title('Backtest Results')
        plt.xlabel('Date')
        plt.ylabel('Balance')
        plt.show()

    def fetch_historical_data(self, start_date, end_date):
        # Implement this method to fetch historical data from your data source
        pass

# Usage
improved_bot = ImprovedAIPoweredTradingBot(
    symbol='BTCUSDT',
    time_frame='1h',
    risk_percentage=0.01,
    stop_loss_pct=0.01,
    take_profit_pct=0.02,
    live_trading=False  # Set to True for live trading
)

improved_bot.run(capital=10000)
