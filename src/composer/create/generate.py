import numpy as np
import pandas as pd
from datetime import datetime
import sys
import time

class Generator:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

    def ohlcv(self, start_date, periods, start_price, col_name, seed):
        self.logger.debug(__file__, 'ohlcv', 'Starting OHLCV data generation')

        np.random.seed(seed)
        steps = np.random.normal(loc=0, scale=0.0018, size=periods)
        steps[0] = 0
        prices = start_price + np.cumsum(steps)
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
        ohlcv_df = self._volume(ohlcv_df)

        ohlcv_df.reset_index(inplace=True)
        ohlcv_df['timestamp'] = ohlcv_df['timestamp'].astype(np.int64) // 10**9

        self.logger.debug(__file__, 'ohlcv', 'Finished OHLCV data generation')
        return ohlcv_df

    def _volume(self, ohlcv_df):
        base_volume = self.config["tokens"][0]["total"] * self.config["tokens"][0]["volatility"] * 0.1
        volume_random_walk = np.random.normal(loc=0, scale=0.9, size=len(ohlcv_df))
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
