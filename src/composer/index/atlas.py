import os
import sqlite3
import json
import pandas as pd
import numpy as np
from typing import List, Tuple, Any
from ..utils.logger import Logger

class Atlas:
    def __init__(self, logger: Logger, db_path: str = "data/market.db") -> None:
        """
        Initialize the Atlas object with the specified logger and database path.

        Args:
            logger (Logger): The logger object for logging messages.
            db_path (str): The path where the database file will be stored.
        """
        self.logger = logger
        self.token = None
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialize_db()

    def _initialize_db(self) -> None:
        """
        Initialize the database and create necessary tables if they do not exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Catalog (
                    id INTEGER PRIMARY KEY,
                    start INTEGER,
                    end INTEGER,
                    chunk INTEGER,
                    stable TEXT,
                    cycle BLOB
                )
            """)
            conn.commit()
        self.logger.debug(__file__, "_initialize_db", "Initialized database and Catalog table.", Atlas.__name__)

    def set_catalog(self, start: int, end: int, chunk: int, stable: str, cycle: np.ndarray) -> None:
        """
        Set the catalog information in the database.

        Args:
            start (int): Start timestamp.
            end (int): End timestamp.
            chunk (int): Chunk size in seconds.
            stable (str): Stable asset name.
            cycle (np.ndarray): Bear/bull cycle sine wave.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM Catalog")
            cursor.execute("""
                INSERT INTO Catalog (start, end, chunk, stable, cycle) VALUES (?, ?, ?, ?, ?)
            """, (start, end, chunk, stable, cycle.tobytes()))
            conn.commit()
        self.logger.debug(__file__, "set_catalog", "Saved catalog information.", Atlas.__name__)

    def get_catalog(self) -> Tuple[int, int, int, str, np.ndarray]:
        """
        Retrieve the catalog information from the database.

        Returns:
            Tuple[int, int, int, str, np.ndarray]: The catalog information.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT start, end, chunk, stable, cycle FROM Catalog")
            catalog_data = cursor.fetchone()
            if catalog_data:
                start, end, chunk, stable, cycle_data = catalog_data
                cycle = np.frombuffer(cycle_data, dtype=np.float64)
                return start, end, chunk, stable, cycle
            else:
                raise ValueError("Catalog information not found.")

    def create_token_table(self) -> None:
        """
        Create a table for a token in the database if it does not exist.

        Args:
            token (str): The token name.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.token} (
                    id INTEGER PRIMARY KEY,
                    initial REAL,
                    volatility REAL,
                    popularity REAL,
                    price_timestamp REAL,
                    price REAL,
                    ohlcv_timestamp REAL,
                    ohlcv BLOB,
                    factors BLOB
                )
            """)
            conn.commit()
        self.logger.debug(__file__, "create_token_table", f"Created table for token {self.token}.", Atlas.__name__)

    def set_token_initial(self, initial: float, volatility: float, popularity: float) -> None:
        """
        Set the initial configuration for a token in the database.

        Args:
            initial (float): Initial price.
            volatility (float): Volatility value.
            popularity (float): Popularity value.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {self.token} (initial, volatility, popularity) VALUES (?, ?, ?)
            """, (initial, volatility, popularity))
            conn.commit()
        self.logger.debug(__file__, "set_token_initial", f"Saved initial configuration for token {self.token}.", Atlas.__name__)

    def append_token_price(self, price_data: pd.DataFrame) -> None:
        """
        Append new price data for a token in the database.

        Args:
            price_data (pd.DataFrame): DataFrame containing price data with columns ['price_timestamp', 'price'].
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(f"""
                INSERT INTO {self.token} (price_timestamp, price) VALUES (?, ?)
            """, [(row['price_timestamp'], row['price']) for _, row in price_data.iterrows()])
            conn.commit()
        self.logger.debug(__file__, "append_token_price", f"Appended new price data for token {self.token}.", Atlas.__name__)

    def append_token_ohlcv(self, ohlcv_data: pd.DataFrame) -> None:
        """
        Append new OHLCV data for a token in the database.

        Args:
            ohlcv_data (pd.DataFrame): DataFrame containing OHLCV data with columns ['ohlcv_timestamp', 'open', 'high', 'low', 'close', 'volume'].
        """
        # Validate and convert timestamps
        for i, value in ohlcv_data['ohlcv_timestamp'].items():
            if isinstance(value, int):
                ohlcv_data.at[i, 'ohlcv_timestamp'] = float(value)
            elif not isinstance(value, float):
                raise ValueError(f"Invalid timestamp value at index {i}: {value}")

        ohlcv_data['ohlcv_timestamp'] = ohlcv_data['ohlcv_timestamp'].astype(float)

        # Prepare data for insertion
        records = [
            (row['ohlcv_timestamp'], row[['open', 'high', 'low', 'close', 'volume']].to_json().encode('utf-8'))
            for _, row in ohlcv_data.iterrows()
        ]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(f"""
                INSERT INTO {self.token} (ohlcv_timestamp, ohlcv) VALUES (?, ?)
            """, records)
            conn.commit()

        self.logger.debug(__file__, "append_token_ohlcv", f"Appended new OHLCV data for token {self.token}.", Atlas.__name__)

    def set_token_factors(self, factors_data: pd.DataFrame) -> None:
        """
        Set the factors data for a token in the database.

        Args:
            factors_data (pd.DataFrame): DataFrame containing factors data.
        """
        factors_blob = factors_data.to_json().encode('utf-8')
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE {self.token} SET factors = ? WHERE id = 1
            """, (factors_blob,))
            conn.commit()
        self.logger.debug(__file__, "set_token_factors", f"Set factors data for token {self.token}.", Atlas.__name__)

    def set_token_price(self, price_data: pd.DataFrame) -> None:
        """
        Set the token price data in the database, overwriting any existing data.

        Args:
            price_data (pd.DataFrame): DataFrame containing price data with columns ['timestamp', 'price'].
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.token} WHERE price IS NOT NULL")
            cursor.executemany(f"""
                INSERT INTO {self.token} (price_timestamp, price) VALUES (?, ?)
            """, [(row['price_timestamp'], row['price']) for _, row in price_data.iterrows()])
            conn.commit()
        self.logger.debug(__file__, "set_token_price", f"Set new price data for token {self.token}.", Atlas.__name__)

    def get_latest_price(self):
        """
        Retrieve the latest data point for the token.
        """
        query = f"SELECT price_timestamp, price FROM {self.token} ORDER BY price_timestamp DESC LIMIT 1"
        self.logger.debug(__file__, "get_latest_price", f"Executing query: {query}")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                latest_data = pd.DataFrame([result], columns=['price_timestamp', 'price'])
                return latest_data
            else:
                return pd.DataFrame(columns=['price_timestamp', 'price'])

    def get_latest_ohlcv(self):
        """
        Retrieve the latest OHLCV data point for the token.
        """
        query = f"SELECT ohlcv_timestamp, ohlcv FROM {self.token} ORDER BY ohlcv_timestamp DESC LIMIT 1"
        self.logger.debug(__file__, "get_latest_ohlcv", f"Executing query: {query}")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                latest_data = pd.DataFrame([result], columns=['ohlcv_timestamp', 'ohlcv'])
                return latest_data
            else:
                return pd.DataFrame(columns=['ohlcv_timestamp', 'ohlcv'])

    def get_token_factors(self) -> pd.DataFrame:
        """
        Retrieve the factors data for a token from the database.

        Returns:
            pd.DataFrame: DataFrame containing the factors data.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT factors FROM {self.token} WHERE factors IS NOT NULL")
            data = cursor.fetchone()
        if data:
            factors_data = pd.read_json(data[0].decode('utf-8'))
            self.logger.debug(__file__, "get_token_factors", f"Retrieved factors data for token {self.token}.", Atlas.__name__)
            return factors_data
        else:
            self.logger.debug(__file__, "get_token_factors", f"No factors data found for token {self.token}.", Atlas.__name__)
            return pd.DataFrame()

    def get_token_price(self) -> pd.DataFrame:
        """
        Retrieve the token price data from the database.

        Returns:
            pd.DataFrame: DataFrame containing the token price data with columns ['timestamp', 'price'].
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT price_timestamp, price FROM {self.token} WHERE price_timestamp IS NOT NULL")
            data = cursor.fetchall()
        df = pd.DataFrame(data, columns=['price_timestamp', 'price'])
        self.logger.debug(__file__, "get_token_price", f"Retrieved token price data for {self.token}.", Atlas.__name__)
        return df

    def get_token_ohlcv(self) -> pd.DataFrame:
        """
        Retrieve the token OHLCV data from the database.

        Returns:
            pd.DataFrame: DataFrame containing the token OHLCV data with columns ['ohlcv_timestamp', 'open', 'high', 'low', 'close', 'volume'].
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT ohlcv FROM {self.token} WHERE ohlcv IS NOT NULL")
            data = cursor.fetchall()
        if data:
            ohlcv_data = pd.read_json(data[0].decode('utf-8'))
            self.logger.debug(__file__, "get_token_ohlcv", f"Retrieved token OHLCV data for {self.token}.", Atlas.__name__)
            return ohlcv_data
        else:
            self.logger.debug(__file__, "get_token_ohlcv", f"No OHLCV data found for token {self.token}.", Atlas.__name__)
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # To be used a few times in generate.py to get data where it needs to be.
    def migrate_timestamps(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Check if the migration is needed
            cursor.execute(f"PRAGMA table_info({self.token})")
            columns = [info[1] for info in cursor.fetchall()]
            if "price_timestamp" not in columns:
                cursor.execute(f"ALTER TABLE {self.token} RENAME COLUMN timestamp TO price_timestamp")
                cursor.execute(f"ALTER TABLE {self.token} ADD COLUMN ohlcv_timestamp REAL")
                conn.commit()
            self.logger.debug(__file__, "__migrate_timestamps", f"Migrated timestamp columns for token {self.token}.")


# Ensure the data directory exists
os.makedirs("data", exist_ok=True)
