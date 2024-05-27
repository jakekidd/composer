import numpy as np
import pandas as pd
from src.composer.utils.logger import Logger

class Generator:
    def __init__(self, logger: Logger, config: dict):
        self.logger = logger
        self.config = config

    def ohlcv(self, start_date, periods, start_price, col_name, seed, token_index=0):
        self.logger.debug(__file__, 'ohlcv', 'Starting OHLCV data generation')

        np.random.seed(seed)

        # Extract token-specific configuration
        token = next((token for token in self.config["tokens"] if token["name"] == col_name), None)
        if not token:
            raise ValueError(f"Token {col_name} not found in configuration")

        volatility = token["volatility"]
        slope = token["slope"]

        # Define maximum positive and negative drifts
        max_positive_drift = 0.001  # Maximum rate of increase
        min_negative_drift = -0.001  # Maximum rate of decrease

        # Calculate the drift based on the slope
        drift = slope * (max_positive_drift if slope > 0 else min_negative_drift)

        # Create a volatility profile with noise
        base_volatility_profile = np.abs(np.sin(np.linspace(0, 2 * np.pi, periods)))  # Example: sine wave
        noise = np.random.normal(loc=0, scale=0.1, size=periods)  # Adding noise to the volatility profile
        volatility_profile = base_volatility_profile + noise
        volatility_profile = np.clip(volatility_profile, 0, None)  # Ensure no negative volatility

        max_scale = 0.09
        scales = volatility * max_scale * (1 + volatility_profile)

        # Generate the steps with a drift
        steps = np.random.normal(loc=drift, scale=scales, size=periods)
        steps[0] = 0  # Ensure the first step is 0 to start from the initial price
        prices = start_price + np.cumsum(steps)
        
        # Apply logarithmic scaling to prevent negative prices
        for i in range(1, len(prices)):
            if prices[i] < 0.5 * start_price:
                prices[i] = prices[i-1] * (1 + 0.1 * np.log1p(prices[i] - prices[i-1]))
            prices[i] = max(prices[i], 0.01)

        prices = [round(i, 4) for i in prices]

        dates = pd.date_range(start=start_date, periods=periods, freq='H')
        fx_df = pd.DataFrame({
            'timestamp': dates,
            'price': prices
        })
        fx_df.set_index('timestamp', inplace=True)

        # Resample to generate OHLC data
        ohlcv_df = fx_df['price'].resample('H').ohlc()

        # Generate volume
        ohlcv_df = self._volume(ohlcv_df, token_index)

        ohlcv_df.reset_index(inplace=True)
        ohlcv_df['timestamp'] = ohlcv_df['timestamp'].astype(np.int64) // 10**9

        self.logger.debug(__file__, 'ohlcv', 'Finished OHLCV data generation')
        return ohlcv_df

    def _volume(self, ohlcv_df, token_index):
        base_volume = self.config["tokens"][token_index]["total"] * self.config["tokens"][token_index]["volatility"] * 0.1
        volume_random_walk = np.random.normal(loc=0, scale=0.8, size=len(ohlcv_df))
        volume = []

        for i in range(len(ohlcv_df)):
            if i == 0:
                previous_volume = base_volume
            else:
                price_change = abs(ohlcv_df['close'][i] - ohlcv_df['open'][i]) / ohlcv_df['close'][i]
                mean_reversion_factor = 0.4
                random_walk_factor = volume_random_walk[i]
                current_volume = mean_reversion_factor * base_volume * (1 + price_change) + random_walk_factor * previous_volume
                volume.append(current_volume)
                previous_volume = current_volume

        # Ensure the volume array matches the length of the DataFrame index
        volume = np.array(volume)
        if len(volume) < len(ohlcv_df):
            volume = np.append(volume, volume[-1])

        # Second pass to correct negative or zero volumes
        for i in range(1, len(volume)):
            if volume[i] <= 0:
                volume[i] = volume[i-1]

        ohlcv_df['volume'] = volume
        return ohlcv_df
