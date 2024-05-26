import os
import sqlite3
import numpy as np
import pandas as pd
from ..utils.logger import Logger

class Atlas:
    def __init__(self, logger: Logger, db_path: str = "data/market.db") -> None:
        self.logger = logger
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialize_db()

    def _initialize_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Catalog (
                    id INTEGER PRIMARY KEY,
                    start INTEGER,
                    end INTEGER,
                    chunk INTEGER,
                    stable TEXT,
                    tickers TEXT
                )
            """)
            conn.commit()
        self.logger.debug(__file__, '_initialize_db', 'Initialized database and tables')

    def get_or_create_catalog(self, config) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Catalog WHERE id = 0")
            row = cursor.fetchone()
            if row:
                start, end, chunk, stable, tickers = row[1], row[2], row[3], row[4], row[5]
                if start != config["start"] or end != config["end"] or stable != config["stable"]:
                    raise ValueError(f"Config values for start, end, or stable do not match the existing catalog.\n"
                                     f"Existing catalog: start={start}, end={end}, stable={stable}")
                config_tickers = ','.join([token["name"] for token in config["tokens"]])
                if tickers != config_tickers:
                    cursor.execute("""
                        UPDATE Catalog
                        SET tickers = ?
                        WHERE id = 0
                    """, (config_tickers,))
                    conn.commit()
            else:
                tickers = ','.join([token["name"] for token in config["tokens"]])
                cursor.execute("""
                    INSERT INTO Catalog (id, start, end, chunk, stable, tickers)
                    VALUES (0, ?, ?, ?, ?, ?)
                """, (config["start"], config["end"], 30 * 24 * 3600, config["stable"], tickers))
                conn.commit()
        self.logger.debug(__file__, 'get_or_create_catalog', 'Retrieved or created catalog')

    def create_token_table(self, token: dict) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # First, create the table if it does not exist
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {token['name']} (
                    id INTEGER PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    cycle BLOB,
                    timestamp INTEGER,
                    initial REAL,
                    volatility REAL,
                    popularity REAL,
                    total INTEGER,
                    slope REAL,
                    seed INTEGER,
                    frozen BOOLEAN DEFAULT FALSE
                )
            """)
            # Now, check the schema of the existing table
            cursor.execute(f"SELECT initial, volatility, popularity, total, slope, seed, frozen FROM {token['name']} LIMIT 1")
            row = cursor.fetchone()
            if row:
                initial, volatility, popularity, total, slope, seed, frozen = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
                print(f"Initial: {initial}, Volatility: {volatility}, Popularity: {popularity}, Total: {total}, Slope: {slope}, Seed: {seed}, Frozen: {frozen}")
                if frozen:
                    raise ValueError(f"Token table for {token['name']} is frozen and cannot be modified.")
                if (initial != token["initial"] or volatility != token["volatility"] or
                    popularity != token["popularity"] or total != token["total"] or
                    slope != token["slope"]):
                    raise ValueError(f"Token configuration values do not match the existing table for ticker {token['name']}.\n"
                                     f"Existing table: initial={initial}, volatility={volatility}, popularity={popularity}, total={total}, slope={slope}")
            else:
                # TODO:
                # if frozen:
                #     raise ValueError(f"Token table for {token['name']} is frozen and cannot be modified.")
                cursor.execute(f"""
                    INSERT INTO {token['name']} (initial, volatility, popularity, total, slope, seed)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (token["initial"], token["volatility"], token["popularity"], token["total"], token["slope"], token["seed"]))
                conn.commit()
        self.logger.debug(__file__, 'create_token_table', f'Created or verified table for token {token["name"]}')

    def get_latest_ohlcv(self, token: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT open, high, low, close, volume, timestamp FROM {token} ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            if row and all(x is not None for x in row):
                return pd.DataFrame([row], columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
            else:
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])

    def get_token_seed(self, token: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT seed FROM {token} LIMIT 1")
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                return None

    def store_token_seed(self, token: str, seed: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE {token}
                SET seed = ?
                WHERE id = 1
            """, (seed,))
            conn.commit()
        self.logger.debug(__file__, 'store_token_seed', f'Stored seed for token {token}')

    def append_ohlcv(self, token: str, ohlcv_df: pd.DataFrame) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for _, row in ohlcv_df.iterrows():
                cursor.execute(f"""
                    INSERT INTO {token} (open, high, low, close, volume, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (row['open'], row['high'], row['low'], row['close'], row['volume'], row['timestamp']))
            conn.commit()
        self.logger.debug(__file__, 'append_ohlcv', f'Appended OHLCV data for token {token}')

    def get_all_ohlcv(self, token: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT open, high, low, close, volume, timestamp FROM {token} ORDER BY timestamp ASC")
            rows = cursor.fetchall()
            if rows:
                return pd.DataFrame(rows, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
            else:
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
