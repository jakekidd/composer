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
            "factors": [
                # {
                #     "name": "valuation",
                #     "weight": 0.5,
                #     "sine": {
                #         "frequency": 2.0,
                #         "amplitude": 5.0,
                #         "slope": 10.0,
                #         "noise": 2.0,
                #         "period_days": 30  # Period in days
                #     }
                # }
            ]
        }
    ]
}

logger = Logger(should_print=True)

def utc_to_formatted(utc_time_seconds: float) -> str:
    date = datetime.datetime.fromtimestamp(utc_time_seconds, tz=datetime.timezone.utc)
    return date.strftime("%Y-%m-%d %H:%M:%S")

def generate_cycle_wave(start_time: int, end_time: int, frequency: float = 1 / (14 * 30.44)) -> pd.DataFrame:
    """
    Generate a sine wave representing bear/bull cycles.

    Args:
        start_time (int): Start timestamp in UTC seconds.
        end_time (int): End timestamp in UTC seconds.
        frequency (float): Frequency of the sine wave. Default is 1 cycle per 14 months.

    Returns:
        pd.DataFrame: DataFrame with daily timestamps and normalized sine wave values between -1.0 and 1.0.
    """
    logger.debug(__file__, generate_cycle_wave.__name__, f"Generating sine wave for bear/bull cycle from {utc_to_formatted(start_time)} to {utc_to_formatted(end_time)}.")
    # Generate daily timestamps
    periods = pd.date_range(
        start=pd.to_datetime(start_time, unit='s'),
        end=pd.to_datetime(end_time, unit='s'),
        freq='D'
    )

    # Generate the sine wave
    t = np.arange(len(periods))
    sine_wave = np.sin(2 * np.pi * frequency * t)

    # Normalize the sine wave between -1.0 and 1.0
    sine_wave = (sine_wave - sine_wave.min()) / (sine_wave.max() - sine_wave.min()) * 2 - 1

    # Create DataFrame
    sine_wave_df = pd.DataFrame({'timestamp': periods, 'sine_wave': sine_wave})

    logger.debug(__file__, generate_cycle_wave.__name__, "Generated normalized sine wave for bear/bull cycle.")
    return sine_wave_df

def generate_factors_wave(start_time: int, end_time: int, base_wave: pd.DataFrame, valuation: dict) -> pd.DataFrame:
    """
    Generate a valuation wave and combine it with the base wave.

    Args:
        start_time (int): Start timestamp in UTC seconds.
        end_time (int): End timestamp in UTC seconds.
        base_wave (pd.DataFrame): DataFrame containing the base sine wave with daily timestamps.
        valuation (dict): Dictionary containing 'slope', 'amplitude', and 'frequency' for the valuation wave.

    Returns:
        pd.DataFrame: DataFrame with daily timestamps and normalized sine wave values between -1.0 and 1.0.
    """
    # Generate hourly timestamps
    periods = pd.date_range(
        start=pd.to_datetime(start_time, unit='s'),
        end=pd.to_datetime(end_time, unit='s'),
        freq='H'
    )

    # Generate the base sine wave for the full period
    base_period = 1 # 1 day periodicity.
    t = np.arange(len(periods))
    frequency = valuation['frequency'] * (2 * np.pi / (base_period * 24))  # Assume 30 days as a base period
    valuation_wave = valuation['amplitude'] * np.sin(frequency * t) + valuation['slope'] * 15.0 * t / len(t)

    # Add noise to the valuation wave
    # noise = np.random.normal(0, 0.01, len(valuation_wave))  # Adjust noise level as needed # *
    # valuation_wave += noise
    noise_level = 2.0  # Adjust noise level as needed
    noise = np.random.normal(0, noise_level, len(valuation_wave))
    valuation_wave += noise

    # Create DataFrame for valuation wave
    valuation_wave_df = pd.DataFrame({'timestamp': periods, 'valuation_wave': valuation_wave})

    # Interpolate base wave to hourly frequency
    base_wave_hourly = base_wave.set_index('timestamp').resample('H').interpolate(method='linear').reset_index()

    # Sum the valuation wave with the base wave
    combined_wave = base_wave_hourly.copy()
    combined_wave['combined_wave'] = base_wave_hourly['sine_wave'] * valuation_wave_df['valuation_wave'].values

    # Normalize the summed wave between -1 and 1
    min_value = np.min(combined_wave['combined_wave'].values)  # Find the minimum value in the combined_wave
    max_value = np.max(combined_wave['combined_wave'].values)
    # combined_wave['combined_wave'] += abs(min_value)  # Raise the combined_wave by the minimum value
    combined_wave['combined_wave'] = 2 * (combined_wave['combined_wave'] - min_value) / (max_value - min_value) - 1

    # Plot the waves
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # Plot summed wave
    fig.add_trace(
        go.Scatter(x=combined_wave['timestamp'], y=combined_wave['combined_wave'], mode='lines', name='Summed Wave'),
        row=1, col=1
    )

    # Plot base wave
    fig.add_trace(
        go.Scatter(x=base_wave_hourly['timestamp'], y=base_wave_hourly['sine_wave'], mode='lines', name='Base Wave'),
        row=2, col=1
    )

    # Plot valuation wave
    fig.add_trace(
        go.Scatter(x=valuation_wave_df['timestamp'], y=valuation_wave_df['valuation_wave'], mode='lines', name='Valuation Wave'),
        row=3, col=1
    )

    fig.update_layout(
        title="Generated Valuation Wave and Combined Waves",
        xaxis_title="Time",
        template="plotly_dark",
        height=900  # Adjust height based on the number of plots
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

def generate_price_data(original_price: float, start_price: float, start_time: int, end_time: int, volatility: float, combined_wave_df: pd.DataFrame):
    """
    Generate price data using a random walk influenced by a bear/bull cycle sine wave.

    Args:
        original_price (float): Original starting price at the beginning of the total period.
        start_price (float): Starting price.
        start_time (int): Start timestamp in UTC seconds.
        end_time (int): End timestamp in UTC seconds.
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

    # Precompute combined wave values for efficiency
    combined_wave_dict = combined_wave_df.set_index('timestamp')['combined_wave'].to_dict()

    # Generate random walk
    for i, current_time in enumerate(periods[1:], start=1):
        # current_day = current_time.floor('D')
        # sine_value = sine_wave_df.loc[sine_wave_df['timestamp'] == current_day, 'sine_wave'].values[0]

        # adjusted_volatility = volatility * (1 + sine_value)  # Adjust volatility based on sine wave
        # Reverse the effect: less std. deviation in bull (sine_value > 0), more in bear (sine_value < 0)
        # adjusted_volatility = volatility * (1 - sine_value)

        # trend = 0.0001 * sine_value  # Base trend adjustment based on sine wave # *
        # adjusted_volatility = volatility * (1 - abs(sine_value))  # Adjust volatility based on sine wave # *
        # change = random.gauss(trend, adjusted_volatility)  # Use trend as the mean # *
        # current_hour = current_time.floor('H')  # Use hourly granularity for combined_wave
        # combined_value = combined_wave_df.loc[combined_wave_df['timestamp'] == current_hour, 'combined_wave'].values[0]

        # current_hour = current_time.floor('H')  # Use hourly granularity for combined_wave
        # combined_value = combined_wave_dict.get(current_hour, 0)

        # trend = 0.0001 * combined_value  # Base trend adjustment based on combined wave
        # adjusted_volatility = volatility #* (1 - abs(combined_value))  # Adjust volatility based on combined wave
        # change = random.gauss(0, volatility)
        # price = prices[-1] * (1 + change)
        # price += combined_value * (original_price / 10.0) # TODO: Use the original value scale.
        # prices.append(price)

        last_price = prices[-1]
        current_hour = current_time.floor('H')  # Use hourly granularity for combined_wave
        combined_value = combined_wave_dict.get(current_hour, 0)

        trend = calculate_trend(combined_value, original_price, last_price)
        # adjusted_volatility = volatility * (1 + (last_price / (last_price + 1e-5)))
        change = random.gauss(trend, 6.52e-04)# 0.005)
        price = last_price * (1 + change)
        # price += combined_value * (original_price / 10.0) # TODO: Use the original value scale.
        prices.append(price)
        # if len(prices) % (24 * 3600) == 0:  # Log progress every day
        #     logger.debug(__file__, generate_price_data.__name__, f"Generated price data for {len(prices) // (24 * 3600)} days.")
        if i % 3600 == 0:  # Update every hour
            spinner_index = (spinner_index + 1) % len(spinner)
            progress += hourly_increment
            bar = '=' * int(progress / 2) + ' ' * (50 - int(progress / 2))
            sys.stdout.write(f"\rGenerating price data: [{bar}] {progress:.2f}% {spinner[spinner_index]}")
            sys.stdout.flush()
            time.sleep(0.1)
    sys.stdout.write(f"\rGenerating price data: [==================================================] 100%         \n")

    prices = np.array(prices[:len(periods)])

    # Convert to DataFrame
    price_data = pd.DataFrame({'timestamp': periods.astype(int) // 10**9, 'price': prices})

    logger.info(__file__, generate_price_data.__name__, f"Generated price data. Took: {int(time.time() - method_start_time)}s.")
    return price_data

def apply_combined_wave_to_price_data(price_data: pd.DataFrame, combined_wave_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the combined wave influence to the finalized price data.

    Args:
        price_data (pd.DataFrame): DataFrame containing the finalized price data with 1-second granularity.
        combined_wave_df (pd.DataFrame): DataFrame containing the combined wave with 1-hour granularity.

    Returns:
        pd.DataFrame: Updated price data with combined wave influence applied.
    """
    price_data = price_data.copy()  # To avoid modifying the original DataFrame
    combined_wave_dict = combined_wave_df.set_index('timestamp')['combined_wave'].to_dict()

    # Apply combined wave influence
    for i, row in price_data.iterrows():
        current_hour = row['timestamp'] // 3600 * 3600  # Align to the nearest hour
        combined_value = combined_wave_dict.get(current_hour, 0)
        price_data.at[i, 'price'] += combined_value * (row['price'] * 9.9)

    return price_data

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
    cycle_wave_df = None
    chunk_size = None
    stable = None
    try:
        catalog = atlas.get_catalog()
        start_time, end_time, chunk_size, stable, cycle = catalog
        cycle_wave_df = pd.DataFrame({'timestamp': pd.date_range(start=pd.to_datetime(start_time, unit='s'), end=pd.to_datetime(end_time, unit='s'), freq='D'), 'sine_wave': cycle})
        cycle = None # Erase cycle to clear mem, we only need the dataframe.

        # Check if sine_wave_df is valid
        if cycle_wave_df.empty or cycle_wave_df['timestamp'].iloc[0] != pd.to_datetime(start_time, unit='s') or cycle_wave_df['timestamp'].iloc[-1] != pd.to_datetime(end_time, unit='s'):
            logger.warn(__file__, main.__name__, "Invalid sine_wave_df. Regenerating...", Atlas.__name__)
            cycle_wave_df = generate_cycle_wave(start_time, end_time)
            atlas.set_catalog(start_time, end_time, CHUNK_SIZE, TOKENS["stable"], cycle_wave_df['sine_wave'].values)
            logger.debug(__file__, main.__name__, "Catalog data regenerated and saved.", Atlas.__name__)
        else:
            logger.debug(__file__, main.__name__, "Catalog data retrieved and validated.", Atlas.__name__)
    except ValueError:
        logger.debug(__file__, main.__name__, "Catalog data not found. Generating...", Atlas.__name__)
        cycle_wave_df = generate_cycle_wave(start_time, end_time)
        atlas.set_catalog(start_time, end_time, CHUNK_SIZE, TOKENS["stable"], cycle_wave_df['sine_wave'].values)
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

    # Display the second graph for the full period sine wave
    # plot_sine_wave_full_period(cycle_wave_df)

    # Initialize token table
    token_config = TOKENS["tokens"][0]
    # Set the token in Atlas before accessing token-specific methods
    atlas.token = token_config["name"]  # Add this line
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
        combined_wave_df = generate_factors_wave(start_time, end_time, cycle_wave_df, {
            'slope': 0.5,
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

    while current_time < end_time:
        next_time = current_time + CHUNK_SIZE
        if next_time > end_time:
            next_time = end_time

        chunk_price_data = generate_price_data(token_config["initial_price"], start_price, current_time, next_time, token_config["volatility"], combined_wave_df)
        atlas.append_token_price(chunk_price_data)

        start_price = chunk_price_data['price'].iloc[-1]
        current_time = next_time

        # Commenting out the chunk display for now
        # plot_price_and_sine_wave_chunk(chunk_price_data, sine_wave_df, current_time - CHUNK_SIZE, current_time)

        logger.info(__file__, main.__name__, f"Price data chunk saved for period {utc_to_formatted(current_time - CHUNK_SIZE)} to {utc_to_formatted(current_time)}.")


    # Retrieve the full price data and convert to daily datapoints
    full_price_data = atlas.get_token_price()
    # logger.info(__file__, main.__name__, f"Applying combined factor wave to price data...")
    # apply_start_time = time.time()
    # updated_price_data = apply_combined_wave_to_price_data(full_price_data, combined_wave_df)
    # logger.info(__file__, main.__name__, f"Applied combined factor wave to price data. Took: {time.time() - apply_start_time}s")

    full_price_data['timestamp'] = pd.to_datetime(full_price_data['timestamp'], unit='s')
    # updated_price_data['timestamp'] = pd.to_datetime(updated_price_data['timestamp'], unit='s')

    # daily_price_data = full_price_data.resample('D', on='timestamp').mean().reset_index()
    # hourly_price_data = updated_price_data.resample("H", on='timestamp').mean().reset_index()
    hourly_price_data = full_price_data.resample("H", on='timestamp').mean().reset_index()

    # Display the second graph for the full period sine wave
    # plot_sine_wave_full_period(sine_wave_df)

    # Plot the daily price data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly_price_data['timestamp'], y=hourly_price_data['price'], mode='lines', name='Daily Price'))
    fig.update_layout(
        title="Daily Price Data",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark"
    )
    fig.show()
    logger.info(__file__, main.__name__, "Completed plot generation for daily price data.")

if __name__ == "__main__":
    main()
