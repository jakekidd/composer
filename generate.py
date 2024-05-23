import numpy as np
import pandas as pd
import random
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constants.
TIME_INTERVAL = "S"  # Interval between data points.
CHUNK_SIZE = 5 * 24 * 60 * 60  # 5 days in seconds.

TOKENS = {
    "start": 1577836800,  # UTC timestamp for 2020-01-01 00:00:00
    "end": 1735689600,  # UTC timestamp for 2025-01-01 00:00:00
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

def generate_sine_wave(start_time: int, end_time: int, frequency: float = 1 / (14 * 30.44)) -> pd.DataFrame:
    """
    Generate a sine wave representing bear/bull cycles.

    Args:
        start_time (int): Start timestamp in UTC seconds.
        end_time (int): End timestamp in UTC seconds.
        frequency (float): Frequency of the sine wave. Default is 1 cycle per 14 months.

    Returns:
        pd.DataFrame: DataFrame with daily timestamps and normalized sine wave values between -1.0 and 1.0.
    """
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
    return sine_wave_df

def generate_price_data(start_price: float, start_time: int, end_time: int, volatility: float, sine_wave_df: pd.DataFrame):
    """
    Generate price data using a random walk influenced by a bear/bull cycle sine wave.

    Args:
        start_price (float): Starting price.
        start_time (int): Start timestamp in UTC seconds.
        end_time (int): End timestamp in UTC seconds.
        volatility (float): Base volatility.
        sine_wave_df (pd.DataFrame): DataFrame containing the sine wave for bear/bull cycles.

    Returns:
        pd.DataFrame: Generated price data.
    """
    periods = pd.date_range(
        start=pd.to_datetime(start_time, unit='s'),
        end=pd.to_datetime(end_time, unit='s'),
        freq=TIME_INTERVAL
    )[:-1]  # Remove the last period to match the length of prices

    prices = [start_price]

    # Generate random walk
    for current_time in periods[1:]:
        current_day = current_time.floor('D')
        sine_value = sine_wave_df.loc[sine_wave_df['timestamp'] == current_day, 'sine_wave'].values[0]
        adjusted_volatility = volatility * (1 + sine_value)  # Adjust volatility based on sine wave

        change = random.gauss(0, adjusted_volatility)
        price = prices[-1] * (1 + change)
        prices.append(price)

    prices = np.array(prices[:len(periods)])

    # Convert to DataFrame
    price_data = pd.DataFrame({'timestamp': periods.astype(int) // 10**9, 'price': prices})

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
    start_time = TOKENS["start"]
    end_time = TOKENS["end"]

    # Generate sine wave for bear/bull cycles
    sine_wave_df = generate_sine_wave(start_time, end_time)

    # Generate price data in chunks
    current_time = start_time
    start_price = TOKENS["tokens"][0]["initial_price"]

    while current_time < end_time:
        next_time = current_time + CHUNK_SIZE
        if next_time > end_time:
            next_time = end_time

        token_config = TOKENS["tokens"][0]

        price_data = generate_price_data(start_price, current_time, next_time, TOKENS["tokens"][0]["volatility"], sine_wave_df)

        # TODO: Save price_data to database.

        start_price = price_data['price'].iloc[-1]
        current_time = next_time

        # Display the first graph for price data and sine wave chunk
        plot_price_and_sine_wave_chunk(price_data, sine_wave_df, current_time - CHUNK_SIZE, current_time)

        # Display the second graph for the full period sine wave
        plot_sine_wave_full_period(sine_wave_df)

        # For demonstration purposes, we will exit after one chunk.
        exit()

if __name__ == "__main__":
    main()
