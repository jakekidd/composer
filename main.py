import sys
import time
import random
import pandas as pd
from src.composer.utils.logger import Logger
from src.composer.utils.config import load_config
from src.composer.create.generate import Generator
from src.composer.index.atlas import Atlas
from src.composer.create.plot import plot_partial_ohlcv, plot_full_ohlcv

# Constants
CHUNK_SIZE_DAYS = 365

def main():
    logger = Logger()
    try:
        config = load_config()
        atlas = Atlas(logger)
        atlas.get_or_create_catalog(config)
        generator = Generator(logger, config)

        for token in config["tokens"]:
            print(f"Processing token: {token['name']}")
            atlas.create_token_table(token)
            start_timestamp = config["start"]
            end_timestamp = config["end"]

            # Set chunk size to the total duration.
            total_duration_seconds = end_timestamp - start_timestamp
            chunk_size_seconds = total_duration_seconds
            current_timestamp = start_timestamp

            seed = atlas.get_token_seed(token["name"])
            if seed is None:
                seed = config.get("seed", random.randint(0, 1000000))
                atlas.store_token_seed(token["name"], seed)
            print(f"Using seed: {seed}")

            while current_timestamp < end_timestamp:
                start_time = time.time()
                chunk_end_timestamp = min(current_timestamp + chunk_size_seconds, end_timestamp)
                periods = (chunk_end_timestamp - current_timestamp) // 3600  # Number of hours in the chunk

                latest_ohlcv = atlas.get_latest_ohlcv(token["name"])
                print(f"Latest OHLCV: {latest_ohlcv}")

                if latest_ohlcv.empty or latest_ohlcv is None:
                    start_price = token["initial"]
                    print(f"Setting start_price to initial: {start_price}")
                else:
                    start_price = latest_ohlcv['close'].iloc[0]
                    print(f"Setting start_price from latest_ohlcv: {start_price}")

                if start_price is None:
                    raise ValueError(f"Start price for token {token['name']} could not be determined.")

                ohlcv_df = generator.ohlcv(
                    start_date=pd.to_datetime(current_timestamp, unit='s'),
                    periods=periods,
                    start_price=start_price,
                    col_name=token["name"],
                    seed=seed
                )

                atlas.append_ohlcv(token["name"], ohlcv_df)
                logger.info(__file__, 'main', f"Processed chunk from {current_timestamp} to {chunk_end_timestamp}. Time taken: {time.time() - start_time} seconds.")

                # Comment out the partial OHLCV plotting code
                # plot_file = plot_partial_ohlcv(ohlcv_df, filename=f"{token['name']}_ohlcv.html")
                # logger.info(__file__, 'main', f"Plot saved to {plot_file}")

                current_timestamp = chunk_end_timestamp

            logger.info(__file__, 'main', f'Generated and stored OHLCV data for {token["name"]}')

            # Plot the full OHLCV data
            full_ohlcv_df = atlas.get_all_ohlcv(token["name"])  # Assuming this method exists to get all data
            plot_full_ohlcv(full_ohlcv_df, filename=f"{token['name']}_full_ohlcv.html")

    except Exception as e:
        logger.error(__file__, 'main', str(e))
        raise e
        # sys.exit(1)

if __name__ == "__main__":
    main()
