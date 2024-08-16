import time
import logging
import hmac
import hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import warnings
from requests.exceptions import RequestException
import csv
import psutil
import ta

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants uses BOT7 Apis
API_KEY = 'xphI8ywTK6IIyyLXhvoSLF8MEWbGPbkC0UFtp7wBRrDvp8UkJMtNdl5ifj28L2Lp'
SECRET_KEY = 'ykGSrhFx79Wb5tFZ8oJaZ15Xqb7Kk65IaxZ4so7K6IkWWGvfKw8Hjoc9J1wJ6PGh'
BASE_URL = 'https://testnet.binance.vision/api/v3'
BTC_SYMBOL = 'BTCUSDT'
INTERVAL = '5m'  # Adjusted timeframe
MIN_TRADE_AMOUNT = 0.0001  # 0.0001 BTC
MAX_POSITION_SIZE = 0.001
PRICE_PRECISION = 2
QUANTITY_PRECISION = 5
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
PRICE_BUFFER_PERCENT = 0.002  # 0.2% buffer
RETRAINING_INTERVAL = 1  # hours
DAILY_LOSS_LIMIT = -50  # Maximum allowable daily loss in USDT

# Global variables
start_time = datetime.now()
total_profit_loss = 0
daily_profit_loss = 0
total_trades = 0
winning_trades = 0
max_drawdown = 0
total_slippage = 0
total_latency = 0
stop_loss_hits = 0
errors = 0
hourly_profits = []
daily_profits = []
weekly_profits = []
monthly_profits = []
yearly_profits = []

# Setup logging
logging.basicConfig(filename='trading_bot_btc.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def update_trade_log():
    global start_time, total_profit_loss, total_trades, winning_trades, max_drawdown, total_slippage, total_latency, stop_loss_hits, errors

    runtime = (datetime.now() - start_time).total_seconds() / 3600  # Runtime in hours
    usdt_balance, btc_balance = get_available_balance()
    current_price = get_current_price(BTC_SYMBOL)
    total_value = usdt_balance + (btc_balance * current_price)
    profit_loss_percent = (total_profit_loss / (total_value - total_profit_loss)) * 100 if total_value != total_profit_loss else 0
    winning_trades_percent = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_profit_per_trade = total_profit_loss / total_trades if total_trades > 0 else 0
    trade_frequency = total_trades / runtime if runtime > 0 else 0
    avg_execution_time = total_latency / total_trades if total_trades > 0 else 0
    avg_slippage = total_slippage / total_trades if total_trades > 0 else 0
    
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # in MB

    with open('trade_log.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Runtime (hours)", "Total Profit/Loss (USDT)", "Total Value (USDT)", "Profit/Loss (%)",
            "Number of Trades", "Winning Trades (%)", "Average Profit/Loss per Trade (USDT)",
            "Max Drawdown (%)", "Trade Frequency (trades/hour)", "Average Execution Time (ms)",
            "Slippage (USDT)", "Latency (ms)", "BTC Price (Current)", "BTC Price Change (24h)",
            "Market Volatility (%)", "Stop-Loss Hits", "Average Position Size (% of portfolio)",
            "Uptime (hours)", "Errors", "Resource Usage (CPU/Memory)"
        ])
        writer.writerow([
            f"{runtime:.2f}", f"{total_profit_loss:.2f}", f"{total_value:.2f}", f"{profit_loss_percent:.2f}%",
            total_trades, f"{winning_trades_percent:.2f}%", f"{avg_profit_per_trade:.2f}",
            f"{max_drawdown:.2f}%", f"{trade_frequency:.2f}", f"{avg_execution_time:.2f}",
            f"{avg_slippage:.2f}", f"{total_latency / total_trades if total_trades > 0 else 0:.2f}",
            f"{current_price:.2f}", "N/A", "N/A",  # Implement BTC price change and market volatility calculation
            stop_loss_hits, "N/A",  # Implement average position size calculation
            f"{runtime:.2f}", errors, f"{cpu_usage:.1f}%/{memory_usage:.1f}MB"
        ])

def log_current_state(current_price, signal, current_data, position):
    signal_str = 'BUY' if signal == 1 else 'SELL'
    logging.info(
        f"Price: {current_price}, Signal: {signal_str}, RSI: {current_data['rsi']:.2f}, "
        f"MACD: {current_data['macd']:.2f}, StochK: {current_data['stoch_k']:.2f}, "
        f"Position: {position}, Volatility: {current_data['volatility']:.2f}%"
    )

def get_signed_params(params):
    query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
    signature = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    return query_string, signature    

def get_server_time():
    try:
        response = requests.get(f"{BASE_URL}/time")
        response.raise_for_status()
        return response.json()['serverTime']
    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting server time: {e}")
        return None

def get_timestamp():
    server_time = get_server_time()
    return server_time if server_time else int(time.time() * 1000)

def calculate_volatility(df, window=14):
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    return volatility.iloc[-1] * 100  # Convert to percentage

def get_available_balance():
    try:
        params = {'timestamp': get_timestamp()}
        query_string, signature = get_signed_params(params)
        url = f"{BASE_URL}/account?{query_string}&signature={signature}"
        
        logging.info(f"Request URL: {url}")
        response = requests.get(url, headers={'X-MBX-APIKEY': API_KEY})
        response.raise_for_status()

        data = response.json()
        logging.info(f"API Response: {data}")
        
        usdt_balance = next((float(balance['free']) for balance in data['balances'] if balance['asset'] == 'USDT'), 0)
        btc_balance = next((float(balance['free']) for balance in data['balances'] if balance['asset'] == 'BTC'), 0)
        
        logging.info(f"Retrieved balances - USDT: {usdt_balance}, BTC: {btc_balance}")
        return usdt_balance, btc_balance
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error getting balance: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response content: {e.response.content}")
        return 0, 0
    except Exception as e:
        logging.error(f"Unexpected error getting balance: {e}")
        return 0, 0

def get_lot_size_filter(symbol):
    try:
        response = requests.get(f"{BASE_URL}/exchangeInfo")
        response.raise_for_status()
        data = response.json()
        symbol_info = next((s for s in data['symbols'] if s['symbol'] == symbol), None)
        if symbol_info:
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if lot_size_filter:
                logging.info(f"Lot size filter for {symbol}: {lot_size_filter}")
                return {
                    'minQty': float(lot_size_filter['minQty']),
                    'maxQty': float(lot_size_filter['maxQty']),
                    'stepSize': float(lot_size_filter['stepSize'])
                }
    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting lot size filter: {e}")
    return None

def adjust_quantity_to_lot_size(quantity, lot_size_filter):
    min_qty = float(lot_size_filter['minQty'])
    max_qty = float(lot_size_filter['maxQty'])
    step_size = float(lot_size_filter['stepSize'])

    # Ensure quantity is within allowed range
    quantity = max(min_qty, min(max_qty, quantity))

    # Calculate the number of decimal places for rounding
    step_size_str = f"{step_size:.10f}".rstrip('0').rstrip('.')
    decimal_places = len(step_size_str.split('.')[-1])

    # Adjust quantity to be a multiple of step_size
    steps = round(quantity / step_size)
    adjusted_quantity = round(steps * step_size, decimal_places)

    # Ensure the adjusted quantity is at least the minimum
    if adjusted_quantity < min_qty:
        adjusted_quantity = min_qty

    # Ensure the adjusted quantity doesn't exceed the maximum
    if adjusted_quantity > max_qty:
        adjusted_quantity = max_qty

    logging.info(f"Original quantity: {quantity}, Adjusted quantity: {adjusted_quantity}")

    return adjusted_quantity

def get_current_price(symbol):
    try:
        response = requests.get(f"{BASE_URL}/ticker/price", params={'symbol': symbol})
        response.raise_for_status()
        return float(response.json()['price'])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching current price: {e}")
        return None

def place_order(symbol, side, quantity, order_type='LIMIT', price=None):
    global total_slippage, total_latency, errors

    for attempt in range(MAX_RETRIES):
        try:
            start_time = time.time()
            current_price = get_current_price(symbol)
            if current_price is None:
                raise Exception("Failed to get current price")

            if order_type == 'LIMIT':
                if side == 'BUY':
                    adjusted_price = round(current_price * (1 + PRICE_BUFFER_PERCENT), PRICE_PRECISION)
                elif side == 'SELL':
                    adjusted_price = round(current_price * (1 - PRICE_BUFFER_PERCENT), PRICE_PRECISION)
            else:
                adjusted_price = price

            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': f"{quantity:.8f}",
                'timestamp': get_timestamp(),
            }
            if order_type == 'LIMIT':
                params['price'] = f"{adjusted_price:.2f}"
                params['timeInForce'] = 'GTC'

            query_string, signature = get_signed_params(params)
            url = f"{BASE_URL}/order?{query_string}&signature={signature}"
            
            logging.info(f"Placing order: {url}")
            logging.info(f"Order params: {params}")

            response = requests.post(url, headers={'X-MBX-APIKEY': API_KEY})
            response.raise_for_status()
            order_result = response.json()

            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            total_latency += latency

            if order_result['status'] == 'FILLED':
                executed_price = float(order_result['fills'][0]['price'])
                slippage = abs(executed_price - current_price)
                total_slippage += slippage
                logging.info(f"Order filled: {side} {order_result['executedQty']} {symbol} at {executed_price}")
                logging.info(f"Slippage: {slippage:.8f}, Latency: {latency:.2f}ms")
                return order_result
            else:
                logging.warning(f"Order not filled immediately. Status: {order_result['status']}")
                time.sleep(RETRY_DELAY)
        except RequestException as e:
            logging.error(f"Request error placing order (attempt {attempt + 1}): {e}")
            errors += 1
        except Exception as e:
            logging.error(f"Unexpected error placing order (attempt {attempt + 1}): {e}")
            errors += 1
        time.sleep(RETRY_DELAY)
    return None

def get_historical_data(symbol, interval, limit=5000):
    try:
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        response = requests.get(f"{BASE_URL}/klines", params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching historical data: {e}")
        return None

def add_indicators(df):
    try:
        df = df.copy()
        
        # Basic price and volume indicators
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_change'] = df['volume'].pct_change()

        # Moving averages
        df['sma_short'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_long'] = ta.trend.sma_indicator(df['close'], window=30)
        df['ema'] = ta.trend.ema_indicator(df['close'], window=14)

        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)

        # Volatility indicators
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

        # Trend indicators
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

        # Volume indicators
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=20)

        # Price channels
        df['upper_channel'] = df['high'].rolling(window=20).max()
        df['lower_channel'] = df['low'].rolling(window=20).min()

        # Adjust momentum calculations and add shorter-term momentum indicators
        def safe_momentum(series, periods):
            momentum = series.pct_change(periods=periods)
            return momentum.fillna(0)  # Fill NaN with 0 for insufficient data

        df['momentum_15m'] = safe_momentum(df['close'], periods=15)
        df['momentum_30m'] = safe_momentum(df['close'], periods=30)
        df['momentum_1h'] = safe_momentum(df['close'], periods=60)
        df['momentum_4h'] = safe_momentum(df['close'], periods=240)
        df['momentum_1d'] = safe_momentum(df['close'], periods=1440)

        # Custom indicators
        df['volatility_ratio'] = df['atr'] / df['close']
        df['rsi_divergence'] = df['rsi'] - df['rsi'].shift(1)
        df['macd_accel'] = df['macd'] - df['macd'].shift(1)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['vwap'] = (df['volume'] * (df['high'] + df['low']) / 2).cumsum() / df['volume'].cumsum()
        df['price_vwap_ratio'] = df['close'] / df['vwap']
        df['stoch_rsi'] = ta.momentum.stochrsi(df['close'], window=14, smooth1=3, smooth2=3)
        df['ema_slope'] = (df['ema'] - df['ema'].shift(5)) / 5

        # Fractal dimension
        def fractal_dimension(series, window=20):
            dates = np.array([i for i in range(window)])
            series = np.array(series[-window:])
            slopes = np.polyfit(dates, series, 1)[0]
            return np.log(window) / (np.log(window) + np.log(1 / np.abs(slopes)))
        
        df['fractal_dim'] = df['close'].rolling(window=20).apply(fractal_dimension)

        # Z-score of close price
        df['close_zscore'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()

        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Volatility
        df['volatility'] = calculate_volatility(df)

        # Handle NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Check for any remaining NaN values
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            logging.warning(f"Columns with NaN values after filling: {nan_columns}")
            print(f"Columns with NaN values after filling: {nan_columns}")
            
            # Fill remaining NaNs with column means
            for col in nan_columns:
                mean_value = df[col].mean()
                df[col].fillna(mean_value, inplace=True)
                logging.info(f"Filled remaining NaNs in {col} with mean value: {mean_value}")
                print(f"Filled remaining NaNs in {col} with mean value: {mean_value}")

        logging.info(f"Successfully added indicators. DataFrame shape: {df.shape}")
        return df

    except Exception as e:
        logging.error(f"Error in add_indicators: {str(e)}")
        return None

def create_model(df):
    try:
        feature_names = [
            'rsi', 'macd', 'stoch_k', 'sma_short', 'sma_long',
            'momentum_15m', 'momentum_30m', 'momentum_1h', 'momentum_4h', 'momentum_1d',
            'volatility_ratio', 'rsi_divergence', 'macd_accel',
            'bb_position', 'price_vwap_ratio', 'stoch_rsi',
            'cmf', 'obv', 'ema_slope', 'fractal_dim', 'close_zscore',
            'hour', 'day_of_week', 'is_weekend', 'volatility'
        ]
        
        X = df[feature_names]
        y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        
        # Handle NaN values
        X = X.fillna(X.mean())
        
        # Remove the last row as it will have NaN in the target variable
        X = X[:-1]
        y = y[:-1]
        
        if X.shape[0] == 0:
            raise ValueError("No valid data points after handling NaNs")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100],  # Reduced number of estimators for faster training
            'learning_rate': [0.05, 0.1],  # Reduced options for learning rate
            'max_depth': [3, 4]  # Reduced options for max depth
        }
        model = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')  # Reduced cross-validation folds
        logging.info("Starting GridSearchCV...")
        model.fit(X, y)
        best_model = model.best_estimator_
        
        scores = cross_val_score(best_model, X, y, cv=3, scoring='accuracy')  # Reduced cross-validation folds
        logging.info(f"Model cross-validation scores: {scores}")
        logging.info(f"Best model parameters: {model.best_params_}")
        
        best_model.fit(X, y)
        return best_model, feature_names
    except Exception as e:
        logging.error(f"Error creating model: {e}")
        print(f"Error creating model: {e}")
        return None, None

def generate_signal(model, current_features, feature_names):
    try:
        model_input = current_features[feature_names].to_frame().T
        model_prediction = model.predict(model_input)
        return model_prediction[0]
    except Exception as e:
        logging.error(f"Error generating signal: {e}")
        return None

def update_profit_logs(profit):
    global hourly_profits, daily_profits, weekly_profits, monthly_profits, yearly_profits
    
    current_time = datetime.now()
    
    hourly_profits.append((current_time, profit))
    daily_profits.append((current_time, profit))
    weekly_profits.append((current_time, profit))
    monthly_profits.append((current_time, profit))
    yearly_profits.append((current_time, profit))
    
    # Remove old entries
    hourly_profits = [p for p in hourly_profits if current_time - p[0] <= timedelta(hours=1)]
    daily_profits = [p for p in daily_profits if current_time - p[0] <= timedelta(days=1)]
    weekly_profits = [p for p in weekly_profits if current_time - p[0] <= timedelta(weeks=1)]
    monthly_profits = [p for p in monthly_profits if current_time - p[0] <= timedelta(days=30)]
    yearly_profits = [p for p in yearly_profits if current_time - p[0] <= timedelta(days=365)]

def calculate_periodic_profits():
    hourly_profit = sum(p[1] for p in hourly_profits)
    daily_profit = sum(p[1] for p in daily_profits)
    weekly_profit = sum(p[1] for p in weekly_profits)
    monthly_profit = sum(p[1] for p in monthly_profits)
    yearly_profit = sum(p[1] for p in yearly_profits)
    
    return hourly_profit, daily_profit, weekly_profit, monthly_profit, yearly_profit

def calculate_drawdown(current_value, peak_value):
    return ((peak_value - current_value) / peak_value) * 100 if peak_value > 0 else 0

def backtest_strategy(df, model, feature_names):
    initial_balance = 1000
    balance = initial_balance
    position = 0
    entry_price = 0
    total_profit = 0
    trades_count = 0
    winning_trades = 0
    peak_value = initial_balance
    max_drawdown = 0
    FEE_RATE = 0.001  # 0.1% fee

    for index, row in df.iterrows():
        signal = generate_signal(model, row, feature_names)
        current_price = row['close']
        volatility = row['volatility']

        volatility_factor = min(max(volatility / 20, 0.5), 2)
        dynamic_stop_loss = -0.01 * volatility_factor
        dynamic_take_profit = 0.03 * volatility_factor
        dynamic_max_position_size = MAX_POSITION_SIZE / volatility_factor

        # Calculate max position size as 1% of current balance, adjusted for volatility
        max_position_size = min(balance * 0.01 / current_price, dynamic_max_position_size)

        current_value = balance + (position * current_price)
        if current_value > peak_value:
            peak_value = current_value
        current_drawdown = ((peak_value - current_value) / peak_value) * 100
        max_drawdown = max(max_drawdown, current_drawdown)

        if position == 0:  # Not in a position, look for buy signals
            if signal == 1:
                max_buy_amount = min(balance, current_price * max_position_size)
                if max_buy_amount >= MIN_TRADE_AMOUNT * current_price:
                    quantity = max_buy_amount / current_price
                    fee = quantity * current_price * FEE_RATE
                    position = quantity
                    entry_price = current_price
                    balance -= (quantity * current_price) + fee
                    trades_count += 1

        elif position > 0:  # In a position, look for sell signals
            profit_percentage = (current_price - entry_price) / entry_price
            exit_signal = (
                signal == 0 or
                profit_percentage >= dynamic_take_profit or
                profit_percentage <= dynamic_stop_loss
            )

            if exit_signal:
                sell_value = position * current_price
                fee = sell_value * FEE_RATE
                balance += sell_value - fee
                profit = (current_price - entry_price) * position - fee
                total_profit += profit
                winning_trades += 1 if profit > 0 else 0
                position = 0
                entry_price = 0

    final_balance = balance + (position * df.iloc[-1]['close'] if position > 0 else 0)
    total_return = (final_balance - initial_balance) / initial_balance * 100
    win_rate = (winning_trades / trades_count * 100) if trades_count > 0 else 0

    return {
        'final_balance': final_balance,
        'total_profit': total_profit,
        'total_return': total_return,
        'trades_count': trades_count,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown
    }

def trading_loop(symbol, interval, model, feature_names):
    global total_profit_loss, total_trades, winning_trades, max_drawdown, stop_loss_hits, errors, daily_profit_loss, start_time

    MIN_REQUIRED_BALANCE = 10  # Minimum USDT balance required to trade
    peak_value = 0
    current_position = None
    entry_price = 0
    accumulated_position = 0
    last_model_update = datetime.now()
    
    lot_size_filter = get_lot_size_filter(symbol)
    if lot_size_filter is None:
        logging.error("Failed to get lot size filter. Exiting trading loop.")
        return

    while True:
        try:
            print("\n--- New trading iteration ---")
            
            # Reset daily profit/loss at the start of a new day
            if datetime.now().date() != start_time.date():
                daily_profit_loss = 0
                start_time = datetime.now()

            # Check if it's time to retrain the model
            if (datetime.now() - last_model_update).total_seconds() / 3600 >= RETRAINING_INTERVAL:
                logging.info("Retraining model...")
                df_retrain = get_historical_data(symbol, interval, limit=5000)
                df_retrain = add_indicators(df_retrain)
                model, feature_names = create_model(df_retrain)
                last_model_update = datetime.now()
                logging.info("Model retrained successfully.")

            # 1. Check balances
            available_usdt, available_btc = get_available_balance()
            print(f"Available USDT balance: {available_usdt:.8f}")
            print(f"Available BTC balance: {available_btc:.8f}")

            if available_usdt < MIN_REQUIRED_BALANCE and available_btc < float(lot_size_filter['minQty']):
                print(f"Insufficient funds. Please deposit at least {MIN_REQUIRED_BALANCE} USDT or {lot_size_filter['minQty']} BTC to start trading.")
                time.sleep(300)  # Wait for 5 minutes before checking again
                continue

            # 2. Get current price
            current_price = get_current_price(symbol)
            if current_price is None:
                print("Failed to get current price. Skipping this iteration.")
                time.sleep(60)
                continue

            print(f"Current {symbol} price: {current_price:.2f}")

            # 3. Calculate current portfolio value and update drawdown
            current_value = available_usdt + (available_btc * current_price)
            if current_value > peak_value:
                peak_value = current_value
            current_drawdown = calculate_drawdown(current_value, peak_value)
            max_drawdown = max(max_drawdown, current_drawdown)

            # 4. Get market data and generate signal
            df = get_historical_data(symbol, interval, limit=100)
            if df is None or df.empty:
                print("Failed to get historical data. Skipping this iteration.")
                time.sleep(60)
                continue

            df = add_indicators(df)
            current_features = df.iloc[-1][feature_names]
            
            # Handle potential NaN values
            nan_columns = current_features[current_features.isnull()].index.tolist()
            if nan_columns:
                for col in nan_columns:
                    mean_value = df[col].mean()
                    current_features[col] = mean_value
                    logging.info(f"Filled NaN in {col} with mean value: {mean_value}")

            current_features = current_features.astype(float)
            signal = generate_signal(model, current_features, feature_names)
            
            # 5. Calculate volatility and adjust parameters
            volatility = df.iloc[-1]['volatility']
            print(f"Current volatility: {volatility:.2f}%")

            volatility_factor = min(max(volatility / 20, 0.5), 2)
            dynamic_stop_loss = -0.01 * volatility_factor
            dynamic_take_profit = 0.03 * volatility_factor
            dynamic_max_position_size = MAX_POSITION_SIZE / volatility_factor

            print(f"Current signal: {'BUY' if signal == 1 else 'SELL'}")
            print(f"Dynamic stop-loss: {dynamic_stop_loss*100:.2f}%")
            print(f"Dynamic take-profit: {dynamic_take_profit*100:.2f}%")
            print(f"Dynamic max position size: {dynamic_max_position_size:.8f} BTC")

            log_current_state(current_price, signal, df.iloc[-1], current_position)

            # 6. Execute trading logic
            if current_position is None and signal == 1:  # BUY signal
                max_buy_amount = min(available_usdt, dynamic_max_position_size * current_value)
                quantity_to_buy = adjust_quantity_to_lot_size(max_buy_amount / current_price, lot_size_filter)

                if quantity_to_buy >= float(lot_size_filter['minQty']):
                    order = place_order(symbol, 'BUY', quantity_to_buy)
                    if order and order['status'] == 'FILLED':
                        print(f"Placed BUY order for {quantity_to_buy:.8f} {symbol} at {current_price:.2f} USDT")
                        total_trades += 1
                        current_position = 'LONG'
                        accumulated_position += quantity_to_buy
                        entry_price = float(order['fills'][0]['price'])
                    else:
                        print("Failed to place BUY order.")
                else:
                    print(f"Buy amount too small. Minimum trade amount: {lot_size_filter['minQty']}")

            elif current_position == 'LONG':
                profit_percentage = (current_price - entry_price) / entry_price
                should_sell = (
                    signal == 0 or  # SELL signal
                    profit_percentage >= dynamic_take_profit or
                    profit_percentage <= dynamic_stop_loss
                )

                if should_sell and accumulated_position >= float(lot_size_filter['minQty']):
                    quantity_to_sell = adjust_quantity_to_lot_size(accumulated_position, lot_size_filter)
                    order = place_order(symbol, 'SELL', quantity_to_sell)
                    if order and order['status'] == 'FILLED':
                        print(f"Placed SELL order for {quantity_to_sell:.8f} {symbol} at {current_price:.2f} USDT")
                        total_trades += 1
                        current_position = None
                        exit_price = float(order['fills'][0]['price'])
                        profit = (exit_price - entry_price) * quantity_to_sell
                        total_profit_loss += profit
                        daily_profit_loss += profit
                        update_profit_logs(profit)
                        if profit > 0:
                            winning_trades += 1
                        if profit_percentage <= dynamic_stop_loss:
                            stop_loss_hits += 1
                        print(f"Trade closed. Profit: {profit:.2f} USDT ({profit_percentage*100:.2f}%)")
                        accumulated_position = 0
                    else:
                        print("Failed to place SELL order.")
                elif should_sell:
                    print(f"Accumulated position {accumulated_position:.8f} is below minimum sell quantity {lot_size_filter['minQty']}. Holding position.")

            # Check daily loss limit
            if daily_profit_loss <= DAILY_LOSS_LIMIT:
                logging.warning(f"Daily loss limit reached: {daily_profit_loss:.2f} USDT. Stopping trading for today.")
                print(f"Daily loss limit reached: {daily_profit_loss:.2f} USDT. Stopping trading for today.")
                break

            # 7. Update logs and calculate profits
            update_trade_log()
            hourly_profit, daily_profit, weekly_profit, monthly_profit, yearly_profit = calculate_periodic_profits()
            print(f"Hourly Profit: {hourly_profit:.2f}, Daily Profit: {daily_profit:.2f}, "
                  f"Weekly Profit: {weekly_profit:.2f}, Monthly Profit: {monthly_profit:.2f}, "
                  f"Yearly Profit: {yearly_profit:.2f}")

        except Exception as e:
            logging.error(f"Error in trading loop: {e}")
            print(f"Error in trading loop: {e}")
            errors += 1

        print("Waiting for next iteration...")
        time.sleep(60)  # Wait for 60 seconds before next iteration

def main():
    logging.info("Starting trading bot...")
    print("Starting trading bot...")

    try:
        # Fetch historical data for backtesting
        logging.info("Fetching historical data for backtesting...")
        print("Fetching historical data for backtesting...")
        df_btc = get_historical_data(BTC_SYMBOL, INTERVAL, limit=5000)
        
        if df_btc is None or df_btc.empty:
            logging.error("Failed to fetch historical data. Exiting.")
            print("Failed to fetch historical data. Exiting.")
            return

        logging.info(f"Fetched historical data shape: {df_btc.shape}")
        print(f"Fetched historical data shape: {df_btc.shape}")

        # Add indicators
        logging.info("Adding indicators...")
        print("Adding indicators...")
        df_btc = add_indicators(df_btc)
        
        if df_btc is None:
            logging.error("Failed to add indicators. Exiting.")
            print("Failed to add indicators. Exiting.")
            return

        logging.info(f"DataFrame shape after adding indicators: {df_btc.shape}")
        print(f"DataFrame shape after adding indicators: {df_btc.shape}")

        # Train the model
        logging.info("Training the model...")
        print("Training the model...")
        model_btc, feature_names_btc = create_model(df_btc)
        
        if model_btc is None or feature_names_btc is None:
            logging.error("Failed to create model. Exiting.")
            print("Failed to create model. Exiting.")
            return

        logging.info("Model created successfully.")
        print("Model created successfully.")

        # Run backtest
        logging.info("Running backtest...")
        print("Running backtest...")
        backtest_results = backtest_strategy(df_btc, model_btc, feature_names_btc)

        logging.info("Backtest Results:")
        print("Backtest Results:")
        for key, value in backtest_results.items():
            logging.info(f"{key}: {value:.2f}")
            print(f"{key}: {value:.2f}")

        # Calculate additional metrics
        avg_profit_per_trade = backtest_results['total_profit'] / backtest_results['trades_count'] if backtest_results['trades_count'] > 0 else 0
        logging.info(f"Average Profit per Trade: {avg_profit_per_trade:.4f} USDT")
        print(f"Average Profit per Trade: {avg_profit_per_trade:.4f} USDT")

        # Start live trading
        logging.info("Starting live trading...")
        print("Starting live trading...")
        trading_loop(BTC_SYMBOL, INTERVAL, model_btc, feature_names_btc)

    except KeyboardInterrupt:
        logging.info("\nTrading bot stopped by user.")
        print("\nTrading bot stopped by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
    finally:
        logging.info("Trading session ended.")
        print("Trading session ended.")

if __name__ == "__main__":
    main()
