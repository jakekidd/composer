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

        # Calculate volume
        total_tokens = self.config["tokens"][0]["total"]
        market_caps = ohlcv_df['close'] * total_tokens
        volumes = []

        # Loading spinner and bar items.
        spinner = ['|', '/', '―', '\\']
        spinner_index = 0
        total_seconds = periods * 3600
        hourly_increment = (3600 / total_seconds) * 100  # Increment percentage for each hour
        progress = 0

        for i in range(len(ohlcv_df)):
            if i == 0:
                previous_volume = market_caps.iloc[0] * self.config["tokens"][0]["volatility"]
            else:
                previous_volume = volumes[-1]
            volume = market_caps.iloc[i] * self.config["tokens"][0]["volatility"] + abs(ohlcv_df['open'][i] - ohlcv_df['close'][i]) * 0.05 + previous_volume * 0.001
            volumes.append(volume)

            if i % 3600 == 0:  # Update every hour
                spinner_index = (spinner_index + 1) % len(spinner)
                progress += hourly_increment
                bar = '█' * int(progress / 2) + ' ' * (50 - int(progress / 2))
                sys.stdout.write(f"\rGenerating OHLCV data: [{bar}] {progress:.2f}% {spinner[spinner_index]}")
                sys.stdout.flush()
                time.sleep(0.1)
        sys.stdout.write(f"\rGenerating OHLCV data: [{'█' * 50}] 100%         \n")

        ohlcv_df['volume'] = volumes
        ohlcv_df.reset_index(inplace=True)
        ohlcv_df['timestamp'] = ohlcv_df['timestamp'].astype(np.int64) // 10**9

        self.logger.debug(__file__, 'ohlcv', 'Finished OHLCV data generation')
        return ohlcv_df
