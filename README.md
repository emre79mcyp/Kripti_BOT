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
