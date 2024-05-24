import numpy as np
import pandas as pd
import random
import time
import sys
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.composer.index.atlas import Atlas
from src.composer.utils.logger import Logger
from src.composer.utils.misc import print_df

# Constants.
TIME_INTERVAL = "S"  # Interval between data points.
CHUNK_SIZE = 5 * 24 * 60 * 60  # 5 days in seconds.

TOKENS = {
    "start": 1577836800,  # UTC timestamp for 2020-01-01 00:00:00
    "end": 1735689600,  # UTC timestamp for 2025-01-01 00:00:00
    # "end": 1580515200,  # UTC timestamp for 2020-02-01 00:00:00
    # "end": 1583020800,  # UTC timestamp for 2020-04-01 00:00:00
    "stable": "USDC",
    "tokens": [
        {
            "name": "TEST",
            "initial_price": 10.00,
            "volatility": 0.001,
            "popularity": 1.0,
            "total_tokens": 1000000,
            "factors": []
        }
    ]
}

logger = Logger(should_print=True)

def utc_to_formatted(utc_time_seconds: float) -> str:
    date = datetime.datetime.fromtimestamp(utc_time_seconds, tz=datetime.timezone.utc)
    return date.strftime("%Y-%m-%d %H:%M:%S")

def generate_factors_wave(start_time: int, end_time: int, original_price: float, valuation: dict) -> pd.DataFrame:
    """
    Generate a valuation wave and combine it with the base wave.

    Args:
        start_time (int): Start timestamp in UTC seconds.
        end_time (int): End timestamp in UTC seconds.
        original_price (float): Floating point value for original price in pre-configured stable, used as reference for normalization.
        valuation (dict): Dictionary containing 'slope', 'amplitude', and 'frequency' for the valuation wave.

    Returns:
        pd.DataFrame: DataFrame with daily timestamps and normalized sine wave values between -1.0 and 1.0.
    """
    # Generate hourly timestamps.
    periods = pd.date_range(
        start=pd.to_datetime(start_time, unit='s'),
        end=pd.to_datetime(end_time, unit='s'),
        freq='H'
    )
    t = np.arange(len(periods))

    # Generate the bear/bull cycle as a sine wave between 0.5 and 2.0, scaled by original price.
    logger.debug(__file__, generate_factors_wave.__name__, f"Generating sine wave for bear/bull cycle from {utc_to_formatted(start_time)} to {utc_to_formatted(end_time)}.")
    frequency = 1 / (14 * 30.44 * 24)  # 1 cycle per 14 months, adjusted for hourly frequency.
    # Generate a random, gentle function to add some variation here. Not noise, but random.
    variation = np.cumsum(np.random.normal(0, 0.7, len(t)))  # Cumulative sum to create gentle variation
    bear_bull_wave = (np.sin(2 * np.pi * frequency * t) + 1.5) * original_price + variation  # Between 0.5 and 2.0, scaled by original_price.
    logger.debug(__file__, generate_factors_wave.__name__, "Generated normalized sine wave for bear/bull cycle.")

    # Generate the valuation wave as a linear function starting at original price and increasing by slope.
    # valuation_wave = original_price + valuation['slope'] * t
    # Generate the valuation wave as an ascending sine wave with added noise
    linear_component = original_price + valuation['slope'] * t
    frequency = 1 / 72 # One cycle every 72 hours.
    amplitude = 0.1
    # sine_component = np.sin(2 * np.pi * (frequency / 24 * 60) * t) * 100 * original_price  # Adjusted frequency and increased amplitude
    sine_component = np.sin(2 * np.pi * frequency * t) * amplitude * original_price
    # noise = np.random.normal(0, 0.2, len(t))  # Adjust noise level as needed
    variation = np.cumsum(np.random.normal(0, 0.3, len(t)))  # Cumulative sum to create gentle variation
    valuation_wave = linear_component + sine_component + variation

    # Create DataFrame for both waves.
    bear_bull_wave_df = pd.DataFrame({'timestamp': periods, 'bear_bull_wave': bear_bull_wave})
    valuation_wave_df = pd.DataFrame({'timestamp': periods, 'valuation_wave': valuation_wave})

    # Combine the waves by averaging them.
    combined_wave = pd.DataFrame({
        'timestamp': periods,
        'combined_wave': (bear_bull_wave_df['bear_bull_wave'] + valuation_wave_df['valuation_wave']) / 2
    })

    # Plot the waves.
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    # Plot combined wave.
    fig.add_trace(
        go.Scatter(x=combined_wave['timestamp'], y=combined_wave['combined_wave'], mode='lines', name='Combined Wave'),
        row=1, col=1
    )
    # Plot bear/bull wave.
    fig.add_trace(
        go.Scatter(x=bear_bull_wave_df['timestamp'], y=bear_bull_wave_df['bear_bull_wave'], mode='lines', name='Bear/Bull Wave'),
        row=2, col=1
    )
    # Plot valuation wave.
    fig.add_trace(
        go.Scatter(x=valuation_wave_df['timestamp'], y=valuation_wave_df['valuation_wave'], mode='lines', name='Valuation Wave'),
        row=3, col=1
    )

    fig.update_layout(
        title="Generated Valuation Wave and Combined Waves",
        xaxis_title="Time",
        template="plotly_dark",
        height=900  # Adjust height based on the number of plots.
    )
    fig.show()

    return combined_wave

def calculate_trend(combined_value: float, original_price: float, current_price: float) -> float:
    """
    Calculate the trend based on the combined wave value, original price, and current price.
    The trend is adjusted to flatten out as the current price approaches 0.10.

    Args:
        combined_value (float): The combined wave value.
        original_price (float): The original starting price at the beginning of the total period.
        current_price (float): The current price.

    Returns:
        float: The calculated trend.
    """
    # Calculate base trend based on combined value and original price
    trend = combined_value ** 2 * 1.35e-07 * original_price

    # Adjust trend as the current price approaches 0.10
    if current_price < 0.2:  # Start flattening out when price is below 0.2
        factor = (current_price - 0.10) / 0.10  # Normalize between 0 and 1 as price approaches 0.10
        trend *= factor  # Reduce the trend as the price approaches 0.10
    return trend

def generate_ou_process_values(length: int, theta: float, mu: float, sigma: float, dt: float, initial_value: float) -> np.ndarray:
    """
    Generate values for an Ornstein-Uhlenbeck (OU) process.

    Args:
        length (int): Length of the time series.
        theta (float): Speed of reversion to the mean.
        mu (float): Long-term mean level.
        sigma (float): Volatility parameter.
        dt (float): Time increment.
        initial_value (float): Initial value of the process.

    Returns:
        np.ndarray: Array of generated OU process values.
    """
    ou_values = np.zeros(length)
    ou_values[0] = initial_value

    for t in range(1, length):
        ou_values[t] = ou_values[t-1] + theta * (mu - ou_values[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        # Clip the OU values to avoid extreme values
        ou_values[t] = np.clip(ou_values[t], -np.inf, np.inf)  # You can adjust the clipping range if necessary
        # Log each step for debugging
        # logger.debug(__file__, "generate_ou_process_values", f"OU Value at t={t}: {ou_values[t]}")

    return ou_values

    # Log input values for debugging
    # logger.debug(__file__, "generate_ou_process_values", f"last_price: {last_price}, theta: {theta}, mu: {mu}, sigma: {sigma}, dt: {dt}")

    # # Calculate the OU process value
    # mean_reversion_term = theta * (mu - last_price)
    # stochastic_term = sigma * np.sqrt(dt) * np.random.normal()

    # # Log terms for debugging
    # # logger.debug(__file__, "generate_ou_process_values", f"mean_reversion_term: {mean_reversion_term}, stochastic_term: {stochastic_term}")

    # # Calculate the new price
    # new_price = last_price + mean_reversion_term * dt + stochastic_term

    # # Log the new price
    # logger.debug(__file__, "generate_ou_process_values", f"mean_reversion_term: {mean_reversion_term}, stochastic_term: {stochastic_term}, new_price: {new_price}")

    # return new_price

def derive_ou_inputs(price_data: pd.DataFrame) -> dict:
    """
    Derive OU process parameters (theta, mu, sigma) from the recent price data.

    Args:
        price_data (pd.DataFrame): DataFrame containing recent price data with columns ['timestamp', 'price'].

    Returns:
        dict: Dictionary containing derived OU process parameters.
    """
    prices = price_data['price'].values
    dt = 1 / (60 * 60)  # Time increment in hours

    # Estimate mu as the mean of the prices
    mu = np.mean(prices)

    # Estimate theta using a simple method (negative slope of log of price differences)
    log_prices = np.log(prices)
    theta = -np.polyfit(np.arange(len(log_prices)), log_prices, 1)[0] / dt

    # Estimate sigma as the standard deviation of the price differences
    price_diffs = np.diff(prices)
    sigma = np.std(price_diffs) / np.sqrt(dt)

    return {"theta": theta, "mu": mu, "sigma": sigma}

def generate_price_data(
        original_price: float,
        start_price: float,
        start_time: int,
        end_time: int,
        ou_vars: dict,
        volatility: float,
        combined_wave_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Generate price data using a random walk influenced by a bear/bull cycle sine wave.

    Args:
        original_price (float): Original starting price at the beginning of the total period.
        start_price (float): Starting price.
        start_time (int): Start timestamp in UTC seconds.
        end_time (int): End timestamp in UTC seconds.
        ou_vars (dict): "theta", "mu", "sigma" for OU calculation based on historical data.
        volatility (float): Base volatility.
        combined_wave_df (pd.DataFrame): DataFrame containing the aggregated external factors.

    Returns:
        pd.DataFrame: Generated price data.
    """
    logger.info(__file__, generate_price_data.__name__, f"Generating price data from {utc_to_formatted(start_time)} to {utc_to_formatted(end_time)}.")
    method_start_time = time.time()

    periods = pd.date_range(
        start=pd.to_datetime(start_time, unit='s'),
        end=pd.to_datetime(end_time, unit='s'),
        freq=TIME_INTERVAL
    )[:-1]  # Remove the last period to match the length of prices

    prices = [start_price]

    # Loading spinner and bar items.
    spinner = ['|', '/', '―', '\\']
    spinner_index = 0
    total_seconds = (end_time - start_time)
    hourly_increment = (3600 / total_seconds) * 100  # Increment percentage for each hour
    progress = 0

    # Precompute combined wave values for efficiency.
    combined_wave_dict = combined_wave_df.set_index('timestamp')['combined_wave'].to_dict()

    # Extract OU process parameters from ou_vars.
    # theta = ou_vars.get('theta', 0.1)
    # mu = ou_vars.get('mu', original_price)
    # sigma = ou_vars.get('sigma', 0.01)

    # Generate OU process values.
    # dt = 1 / (60 * 60)  # Time increment in hours.
    # ou_start_time = time.time()
    # ou_process_values = generate_ou_process_values(len(periods), theta, mu, sigma, dt, start_price)
    # logger.info(__file__, generate_price_data.__name__, f"Generated OU process values. Took: {int(time.time() - ou_start_time)}s.")

    # Generate random walk.
    for i, current_time in enumerate(periods[1:], start=1):
        last_price = prices[-1]
        current_hour = current_time.floor('H')  # Use hourly granularity for combined_wave.
        combined_value = combined_wave_dict.get(current_hour, 0)

        mean_price = combined_value  # Use the combined influence wave value as the mean price for mean reversion.
        mean_reversion_coeff = 0.001
        change = random.gauss(1e-07, volatility) + mean_reversion_coeff * (mean_price - last_price) #+ ou_process_values[i]
        
        # Clip the change to avoid overflow
        change = np.clip(change, -1e2, 1e2)
        
        price = last_price * (1 + change)
        
        # Check for overflow before calculation
        if np.isinf(change) or np.isnan(change):
            logger.error(__file__, generate_price_data.__name__, f"Overflow or NaN detected in change calculation: change={change}")
            raise ValueError(f"Overflow or NaN detected in change calculation: change={change}")
        
        prices.append(price)

        if i % 3600 == 0:  # Update every hour.
            spinner_index = (spinner_index + 1) % len(spinner)
            progress += hourly_increment
            bar = '=' * int(progress / 2) + ' ' * (50 - int(progress / 2))
            sys.stdout.write(f"\rGenerating price data: [{bar}] {progress:.2f}% {spinner[spinner_index]}")# OU Value: {ou_process_values[i]}")
            sys.stdout.flush()
            time.sleep(0.1)
    sys.stdout.write(f"\rGenerating price data: [==================================================] 100%         \n")

    prices = np.array(prices[:len(periods)])

    # Convert to DataFrame.
    price_data = pd.DataFrame({'timestamp': periods.astype(int) // 10**9, 'price': prices})

    logger.info(__file__, generate_price_data.__name__, f"Generated price data. Took: {int(time.time() - method_start_time)}s.")
    return price_data


def convert_to_ohlcv(price_data: pd.DataFrame, token_config: dict) -> pd.DataFrame:
    """
    Convert price data to OHLCV data.

    Args:
        price_data (pd.DataFrame): DataFrame containing price data with columns ['timestamp', 'price'].
        token_config (dict): Configuration dictionary for the token.

    Returns:
        pd.DataFrame: DataFrame containing OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    """
    logger.info(__file__, convert_to_ohlcv.__name__, "Converting price data to OHLCV data.")
    
    # Calculate market cap
    total_tokens = token_config["total_tokens"]
    market_caps = price_data['price'] * total_tokens

    ohlcv_data = {
        'timestamp': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }

    prices = price_data['price'].values
    timestamps = price_data['timestamp'].values

    # Initialize the first data point
    ohlcv_data['timestamp'].append(timestamps[0])
    ohlcv_data['open'].append(prices[0])
    ohlcv_data['close'].append(prices[1])
    ohlcv_data['high'].append(max(prices[0], prices[1]))
    ohlcv_data['low'].append(min(prices[0], prices[1]))
    ohlcv_data['volume'].append(market_caps[0] * 0.001)  # Use a small volume for the initial point

    for i in range(1, len(prices) - 1):
        open_price = ohlcv_data['close'][-1]
        close_price = prices[i + 1]

        high_price = open_price + random.uniform(0, abs(close_price - open_price) * 1.2)
        low_price = open_price - random.uniform(0, abs(open_price - close_price) * 1.2)
        high_price = max(high_price, close_price)
        low_price = min(low_price, close_price)

        volume = market_caps[i] * 0.001 + abs(open_price - close_price) * total_tokens * 0.01

        ohlcv_data['timestamp'].append(timestamps[i])
        ohlcv_data['open'].append(open_price)
        ohlcv_data['close'].append(close_price)
        ohlcv_data['high'].append(high_price)
        ohlcv_data['low'].append(low_price)
        ohlcv_data['volume'].append(volume)

    # Handle the last data point
    ohlcv_data['timestamp'].append(timestamps[-1])
    ohlcv_data['open'].append(ohlcv_data['close'][-1])
    ohlcv_data['close'].append(prices[-1])
    ohlcv_data['high'].append(max(prices[-1], ohlcv_data['open'][-1]))
    ohlcv_data['low'].append(min(prices[-1], ohlcv_data['open'][-1]))
    ohlcv_data['volume'].append(market_caps[-1] * 0.001)

    return pd.DataFrame(ohlcv_data)

def plot_ohlcv_data(ohlcv_data: pd.DataFrame) -> None:
    """
    Plot OHLCV data.

    Args:
        ohlcv_data (pd.DataFrame): DataFrame containing OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    """
    logger.debug(__file__, "plot_ohlcv_data", "Generating plot for OHLCV data.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, specs=[[{"type": "candlestick"}], [{"type": "bar"}]])

    fig.add_trace(
        go.Candlestick(
            x=ohlcv_data['timestamp'],
            open=ohlcv_data['open'],
            high=ohlcv_data['high'],
            low=ohlcv_data['low'],
            close=ohlcv_data['close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=ohlcv_data['timestamp'], y=ohlcv_data['volume'], name='Volume'),
        row=2, col=1
    )

    fig.update_layout(
        title="OHLCV Data",
        xaxis_title="Time",
        yaxis_title="Price/Volume",
        template="plotly_dark",
        height=900  # Adjust height based on the number of plots
    )

    fig.show()

def plot_price_and_sine_wave_chunk(price_data: pd.DataFrame, sine_wave_df: pd.DataFrame, start_time: int, end_time: int) -> None:
    """
    Plot price data and sine wave chunk.

    Args:
        price_data (pd.DataFrame): Generated price data.
        sine_wave_df (pd.DataFrame): DataFrame containing the sine wave for bear/bull cycles.
        start_time (int): Start timestamp in UTC seconds.
        end_time (int): End timestamp in UTC seconds.
    """
    logger.debug(__file__, "plot_price_and_sine_wave_chunk", f"Generating plot for price data and sine wave chunk for period {start_time} to {end_time}.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # Plot price data
    fig.add_trace(
        go.Scatter(x=price_data['timestamp'], y=price_data['price'], mode='lines', name='Price'),
        row=1, col=1
    )

    # Plot sine wave for the chunk on the same x-axis as price data
    sine_wave_section = sine_wave_df[(sine_wave_df['timestamp'] >= pd.to_datetime(start_time, unit='s')) &
                                     (sine_wave_df['timestamp'] <= pd.to_datetime(end_time, unit='s'))]

    fig.add_trace(
        go.Scatter(x=sine_wave_section['timestamp'], y=sine_wave_section['sine_wave'], mode='lines', name='Sine Wave Chunk'),
        row=2, col=1
    )

    fig.update_layout(
        title="Price Data and Sine Wave Chunk",
        xaxis_title="Time (UTC seconds)",
        yaxis_title="Value",
        template="plotly_dark",
        height=600  # Adjust height based on the number of plots
    )

    fig.show()

def plot_sine_wave_full_period(sine_wave_df: pd.DataFrame) -> None:
    """
    Plot the sine wave for the full period.

    Args:
        sine_wave_df (pd.DataFrame): DataFrame containing the sine wave for bear/bull cycles.
    """
    logger.debug(__file__, plot_sine_wave_full_period.__name__, "Generating plot for sine wave full period.")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=sine_wave_df['timestamp'], y=sine_wave_df['sine_wave'], mode='lines', name='Sine Wave Full Period')
    )

    fig.update_layout(
        title="Sine Wave Full Period",
        xaxis_title="Time",
        yaxis_title="Sine Wave Value",
        template="plotly_dark"
    )

    fig.show()

def main() -> None:
    atlas = Atlas(logger)

    start_time = TOKENS["start"]
    end_time = TOKENS["end"]

    # Check if the catalog is set, if not, set it.
    catalog = None
    chunk_size = None
    stable = None
    try:
        catalog = atlas.get_catalog()
        start_time, end_time, chunk_size, stable, _ = catalog
        logger.debug(__file__, main.__name__, "Catalog data retrieved and validated.", Atlas.__name__)
    except ValueError:
        logger.debug(__file__, main.__name__, "Catalog data not found. Generating...", Atlas.__name__)
        atlas.set_catalog(start_time, end_time, CHUNK_SIZE, TOKENS["stable"], np.array([]))
        logger.debug(__file__, main.__name__, "Catalog data generated and saved.", Atlas.__name__)

    # Set chunk size and stable if left at None value.
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if stable is None:
        stable = TOKENS["stable"]

    logger.debug(__file__, main.__name__, f"Catalog:\n\tStart: {utc_to_formatted(start_time)}\n\tEnd: {utc_to_formatted(end_time)}\n\tChunk Size: {chunk_size}\n\tStable: {stable}", Atlas.__name__)

    # Checks to ensure config is remaining consistent with generated data in DB.
    if chunk_size != CHUNK_SIZE:
        logger.error(__file__, main.__name__, f"Catalog chunk size {chunk_size} != configured chunk size {CHUNK_SIZE}. Exiting.", Atlas.__name__)
        exit()
    if stable != TOKENS["stable"]:
        configured_stable = TOKENS["stable"]
        logger.warn(__file__, main.__name__, f"Catalog stable {stable} != configured stable {configured_stable}.", Atlas.__name__)
    if start_time != TOKENS["start"]:
        configured_start = TOKENS["start"]
        logger.error(__file__, main.__name__, f"Catalog start time {utc_to_formatted(start_time)} != configured start time {utc_to_formatted(configured_start)}. Exiting.", Atlas.__name__)
        exit()
    if end_time != TOKENS["end"]:
        configured_end = TOKENS["end"]
        logger.error(__file__, main.__name__, f"Catalog end time {utc_to_formatted(end_time)} != configured end time {utc_to_formatted(configured_end)}. Exiting.", Atlas.__name__)
        exit()

    # Initialize token table
    token_config = TOKENS["tokens"][0]
    # Set the token in Atlas before accessing token-specific methods
    atlas.token = token_config["name"]
    atlas.create_token_table(token_config["name"])
    atlas.set_token_initial(token_config["initial_price"], token_config["volatility"], token_config["popularity"])
    logger.debug(__file__, main.__name__, "Token table initialized as needed and initial configuration set.")

    # Generate or retrieve combined wave and save it to the database
    try:
        combined_wave_df = atlas.get_token_factors()
        if combined_wave_df.empty:
            raise ValueError("No combined wave data found.")
        logger.debug(__file__, main.__name__, "Combined wave data retrieved from database.", Atlas.__name__)
    except ValueError:
        combined_wave_df = generate_factors_wave(start_time, end_time, token_config["initial_price"], {
            'slope': 0.001,
            'amplitude': 1.0,
            'frequency': 0.1
        })
        atlas.set_token_factors(combined_wave_df)
        logger.debug(__file__, main.__name__, "Generated and saved combined wave data.", Atlas.__name__)

    # Retrieve the price data if it exists
    try:
        price_data = atlas.get_token_price()
        if not price_data.empty:
            last_timestamp = price_data['timestamp'].iloc[-1]
            start_price = price_data['price'].iloc[-1]
            current_time = last_timestamp + 1  # Continue from the last timestamp
            logger.info(__file__, main.__name__, f"Continuing price data generation from timestamp {utc_to_formatted(last_timestamp)}.")
        else:
            start_price = token_config["initial_price"]
            current_time = start_time
            logger.info(__file__, main.__name__, "Starting price data generation from the beginning.")
    except ValueError:
        start_price = token_config["initial_price"]
        current_time = start_time
        logger.info(__file__, main.__name__, "Starting price data generation from the beginning (no previous data).")

    ou_vars = {"theta": 0.1, "mu": token_config["initial_price"], "sigma": 0.01}  # Default OU parameters

    while current_time < end_time:
        next_time = current_time + CHUNK_SIZE
        if next_time > end_time:
            next_time = end_time

        chunk_price_data = generate_price_data(token_config["initial_price"], start_price, current_time, next_time, ou_vars, token_config["volatility"], combined_wave_df)
        atlas.append_token_price(chunk_price_data)

        # Update OU parameters based on recent chunk
        ou_vars = derive_ou_inputs(chunk_price_data)

        start_price = chunk_price_data['price'].iloc[-1]
        current_time = next_time

        logger.info(__file__, main.__name__, f"Price data chunk saved for period {utc_to_formatted(current_time - CHUNK_SIZE)} to {utc_to_formatted(current_time)}.")

    # Retrieve the full price data and convert to daily datapoints
    full_price_data = atlas.get_token_price()
    full_price_data['timestamp'] = pd.to_datetime(full_price_data['timestamp'], unit='s')
    hourly_price_data = full_price_data.resample("H", on='timestamp').mean().reset_index()

    # Plot the hourly price data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly_price_data['timestamp'], y=hourly_price_data['price'], mode='lines', name='Hourly Price'))
    fig.update_layout(
        title="Hourly Price Data",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark"
    )
    fig.show()
    logger.info(__file__, main.__name__, "Completed plot generation for hourly price data.")

    # Convert price data to OHLCV data
    ohlcv_data = convert_to_ohlcv(hourly_price_data, token_config)
    atlas.append_token_ohlcv(ohlcv_data)

    # Plot the OHLCV data
    plot_ohlcv_data(ohlcv_data)

    logger.info(__file__, main.__name__, "Completed plot generation for OHLCV data.")

if __name__ == "__main__":
    main()
