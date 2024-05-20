import numpy as np
import pandas as pd
import random
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.composer.utils.factor import Factor

# Constants
TIME_INTERVAL = "S"  # Interval between data points
CHUNK_SIZE = 5 * 24 * 60 * 60  # 5 days in seconds

TOKENS = {
    "start": 1577836800,  # UTC timestamp for 2020-01-01 00:00:00
    "end": 1735689600,  # UTC timestamp for 2025-01-01 00:00:00
    "stable": "USDC",
    "tokens": [
        {
            "name": "TEST",
            "initial_price": 10.00,
            "volatility": 0.05,
            "popularity": 1.0,
            "factors": [
                {
                    "name": "valuation",
                    "weight": 0.5,
                    "sine": {
                        "frequency": 0.1,
                        "amplitude": 0.2,
                        "slope": 0.5,
                        "noise": 0.08
                    }
                }
            ]
        }
    ]
}

def generate_price_data(start_price: float, start_time: int, end_time: int, factors_config: list) -> (pd.DataFrame, list):
    """
    Generate price data using a random walk influenced by factors.

    Args:
        start_price (float): Starting price.
        start_time (int): Start timestamp in UTC seconds.
        end_time (int): End timestamp in UTC seconds.
        factors_config (list): List of factor configurations.

    Returns:
        pd.DataFrame: Generated price data.
        list: List of Factor objects.
    """
    periods = pd.date_range(start=pd.to_datetime(start_time, unit='s'), end=pd.to_datetime(end_time, unit='s'), freq=TIME_INTERVAL)
    prices = [start_price]

    # Initialize factors
    factors = []
    for config in factors_config:
        factor = Factor(config['name'], start_time, end_time, config['weight'])
        if 'sine' in config:
            sine_config = config['sine']
            factor.generate_sine(frequency=sine_config['frequency'], amplitude=sine_config['amplitude'], 
                                 slope=sine_config['slope'], noise_level=sine_config['noise'], start_price=start_price)
        factors.append(factor)

    # Generate random walk
    for i in range(1, len(periods)):
        change = random.normalvariate(0, 0.01)  # Small random change
        price = prices[-1] * (1 + change)
        prices.append(price)

    prices = np.array(prices)

    # Apply factors to the price data
    for factor in factors:
        factor_wave = factor.data[:len(prices) - 1]  # Adjusting length
        factor_wave = np.append(factor_wave, factor_wave[-1])  # Adding the last element to match lengths
        prices = prices * (1 + (factor_wave - 1) * factor.weight)

    # Convert to DataFrame
    price_data = pd.DataFrame({'timestamp': periods.astype(int) // 10**9, 'price': prices})

    return price_data, factors

def display_combined_plots(price_data: pd.DataFrame, factors: list) -> None:
    """
    Display combined plots of price data and factors.

    Args:
        price_data (pd.DataFrame): Generated price data.
        factors (list): List of Factor objects.
    """
    num_plots = len(factors) + 2  # Including aggregated factors plot and price data plot

    fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # Plot price data
    fig.add_trace(
        go.Scatter(x=price_data['timestamp'], y=price_data['price'], mode='lines', name='Price'),
        row=1, col=1
    )

    # Plot aggregated factors
    aggregated_factors = np.sum([factor.data[:len(price_data)] * factor.weight for factor in factors], axis=0)
    fig.add_trace(
        go.Scatter(x=price_data['timestamp'], y=aggregated_factors, mode='lines', name='Aggregated Factors'),
        row=2, col=1
    )

    # Plot individual factors
    for i, factor in enumerate(factors, start=3):
        fig.add_trace(
            go.Scatter(x=price_data['timestamp'], y=factor.data[:len(price_data)], mode='lines', name=factor.name),
            row=i, col=1
        )

    fig.update_layout(
        title="Generated Price Data and Factors",
        xaxis_title="Time (UTC seconds)",
        yaxis_title="Value",
        template="plotly_dark",
        height=300 * num_plots  # Adjust height based on the number of plots
    )

    fig.show()

def main() -> None:
    start_time = TOKENS["start"]
    end_time = TOKENS["end"]
    current_time = start_time
    start_price = TOKENS["tokens"][0]["initial_price"]

    while current_time < end_time:
        next_time = current_time + CHUNK_SIZE
        if next_time > end_time:
            next_time = end_time

        token_config = TOKENS["tokens"][0]
        factors_config = token_config["factors"]

        price_data, factors = generate_price_data(start_price, current_time, next_time, factors_config)

        # TODO: Save price_data to database

        start_price = price_data['price'].iloc[-1]
        current_time = next_time

        # Display combined plots
        display_combined_plots(price_data, factors)

if __name__ == "__main__":
    main()
