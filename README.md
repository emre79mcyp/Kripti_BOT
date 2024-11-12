# Kripti_BOT
An advanced AI-powered cryptocurrency trading bot called "KriptiBot". The key functionalities of this bot are:

Market Data Fetching: The bot fetches historical and real-time market data for the specified cryptocurrency trading pair (in this case, BTCUSDT) and time frame (hourly).
Technical Indicator Calculation: The bot calculates various technical indicators such as SMA (Simple Moving Average), RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), ATR (Average True Range), and Bollinger Bands to provide a comprehensive market analysis.
Machine Learning Model: The bot uses an XGBoost Classifier model to generate trading signals (buy/sell) based on the calculated technical indicators. The model is initially trained on historical data and then periodically retrained and optimized using techniques like Grid Search.
Trading Execution: The bot executes trades (buy or sell) based on the generated trading signals, taking into account risk management parameters like stop-loss and take-profit levels, as well as overall risk limits.
Performance Tracking and Optimization: The bot keeps track of various performance metrics like win rate, average profit per trade, Sharpe ratio, and max drawdown. It also dynamically adjusts its trading parameters (risk percentage, stop-loss, take-profit) based on the recent performance to optimize the strategy.
State Management: The bot saves its current state (including performance metrics, recent predictions, and the trained model) to a file and loads it on the next run, allowing it to continue its operation without losing context.
Logging and Monitoring: The bot logs various events, errors, and performance metrics to a log file, providing visibility into its operation and decision-making.

In summary, the "KriptiBot" is an advanced cryptocurrency trading bot that leverages machine learning and comprehensive technical analysis to navigate the crypto markets, with a focus on adaptive risk management and performance optimization.





AI-powered trading bot using machine learning (ML) with a focus on the Binance cryptocurrency exchange. Here’s a breakdown of its key components and functionality:

Setup and Initialization
Libraries and Imports: It imports libraries for web requests, data processing, ML (XGBoost), logging, and more.
API Configuration: It sets up API keys and endpoints to interact with Binance.
Trading Parameters: Several attributes are defined in the class, such as stop-loss, take-profit percentages, and model parameters.
Logger Setup: It configures logging for tracking bot activity and errors.
Core Functions
API Interaction:

sign_request(): Creates a signed API request using HMAC SHA-256.
make_api_request(): Executes the API calls to Binance.
validate_api_keys(): Checks API key validity.
fetch_ohlcv(): Retrieves historical price data (OHLCV format) from Binance.
Technical Indicators:

Calculates indicators like SMA, RSI, MACD, ATR, and Bollinger Bands, using these to support trade decisions.
AI Signal Generation:

generate_ai_signal(): Uses an XGBoost model to predict buy/sell signals based on the latest market data.
analyze_multiple_timeframes(): Aggregates signals across multiple timeframes for improved decision-making.
Trade Execution and Risk Management:

execute_trade(): Manages trade execution based on AI signals, risk limits, and market conditions.
simulate_trade(): Allows backtesting by simulating trade outcomes.
calculate_position_size(): Calculates the size of each position based on risk management rules.
Performance Monitoring:

Tracks metrics like win rate, average profit, and calculates advanced metrics like Sharpe Ratio and max drawdown.
Dynamic Adjustments: Adjusts risk parameters if recent performance falls below a threshold.
Machine Learning and Model Management
Model Training and Optimization:

retrain_model(): Retrains the model on new data if performance drops.
optimize_hyperparameters(): Uses grid search to optimize XGBoost hyperparameters for better accuracy.
State Persistence:

Saves and loads the bot’s state, model, and performance data using pickle.
Main Loop
run(): The bot’s main loop, which continuously:
Validates API keys.
Fetches data and indicators.
Analyzes signals across timeframes.
Executes or simulates trades based on AI predictions.
Logs metrics, saves state, and optimizes the model periodically.
Summary: This bot combines trading signals, technical analysis, and AI-based decision-making to autonomously execute trades while managing risk dynamically.
