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

    def create_token_table(self, token: str) -> None:
        """
        Create a table for a token in the database if it does not exist.

        Args:
            token (str): The token name.
        """
        self.token = token
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {token} (
                    id INTEGER PRIMARY KEY,
                    initial REAL,
                    volatility REAL,
                    popularity REAL,
                    timestamp REAL,
                    price REAL,
                    ohlcv BLOB
                )
            """)
            conn.commit()
        self.logger.debug(__file__, "create_token_table", f"Created table for token {token}.", Atlas.__name__)

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
            price_data (pd.DataFrame): DataFrame containing price data with columns ['timestamp', 'price'].
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(f"""
                INSERT INTO {self.token} (timestamp, price) VALUES (?, ?)
            """, [(row['timestamp'], row['price']) for _, row in price_data.iterrows()])
            conn.commit()
        self.logger.debug(__file__, "append_token_price", f"Appended new price data for token {self.token}.", Atlas.__name__)

    def append_token_ohlcv(self, ohlcv_data: pd.DataFrame) -> None:
        """
        Append new OHLCV data for a token in the database.

        Args:
            ohlcv_data (pd.DataFrame): DataFrame containing OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        """
        ohlcv_blob = ohlcv_data.to_json().encode('utf-8')
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {self.token} (timestamp, ohlcv) VALUES (?, ?)
            """, (ohlcv_data['timestamp'].iloc[0], ohlcv_blob))
            conn.commit()
        self.logger.debug(__file__, "append_token_ohlcv", f"Appended new OHLCV data for token {self.token}.", Atlas.__name__)

    def get_token_price(self) -> pd.DataFrame:
        """
        Retrieve the token price data from the database.

        Returns:
            pd.DataFrame: DataFrame containing the token price data with columns ['timestamp', 'price'].
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT timestamp, price FROM {self.token} WHERE timestamp IS NOT NULL")
            data = cursor.fetchall()
        df = pd.DataFrame(data, columns=['timestamp', 'price'])
        self.logger.debug(__file__, "get_token_price", f"Retrieved token price data for {self.token}.", Atlas.__name__)
        return df

    def get_token_ohlcv(self) -> pd.DataFrame:
        """
        Retrieve the token OHLCV data from the database.

        Returns:
            pd.DataFrame: DataFrame containing the token OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT ohlcv FROM {self.token} WHERE ohlcv IS NOT NULL")
            data = cursor.fetchall()
        if data:
            ohlcv_data = pd.read_json(data[0][0].decode('utf-8'))
            self.logger.debug(__file__, "get_token_ohlcv", f"Retrieved token OHLCV data for {self.token}.", Atlas.__name__)
            return ohlcv_data
        else:
            self.logger.debug(__file__, "get_token_ohlcv", f"No OHLCV data found for token {self.token}.", Atlas.__name__)
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)
