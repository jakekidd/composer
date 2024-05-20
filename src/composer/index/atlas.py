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
        Initialize the database for the current token. Creates the database file and the tables if they do not exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Catalog (
                    id INTEGER PRIMARY KEY,
                    climate_cycle BLOB
                )
            """)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.token} (
                    id INTEGER PRIMARY KEY,
                    config TEXT,
                    timestamp REAL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL
                )
            """)
            conn.commit()
        self.logger.debug(__file__, "_initialize_db", f"Initialized database for {self.token}", Atlas.__name__)

    # SAVE METHODS

    def save_random_events(self, events: pd.DataFrame) -> None:
        """
        Save the generated random events to the Catalog table.

        Args:
            events (pd.DataFrame): DataFrame containing random events with columns ['timestamp', 'magnitude'].
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS random_events (timestamp REAL, magnitude REAL)")
            cursor.executemany("INSERT INTO random_events (timestamp, magnitude) VALUES (?, ?)",
                            [(row['timestamp'].timestamp(), row['magnitude']) for _, row in events.iterrows()])
            conn.commit()
        self.logger.debug(__file__, "save_random_events", "Saved random events to Catalog table.", Atlas.__name__)

    def save_climate_cycle(self, climate_cycle: np.ndarray) -> None:
        """
        Save the generated climate cycle to the Catalog table.

        Args:
            climate_cycle (np.ndarray): Generated climate cycle data.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM Catalog")
            cursor.execute("""
                INSERT INTO Catalog (climate_cycle) VALUES (?)
            """, (climate_cycle.tobytes(),))
            conn.commit()
        self.logger.debug(__file__, "save_climate_cycle", "Saved climate cycle to Catalog table.", Atlas.__name__)

    def save_target_valuation(self, target_valuation: np.ndarray) -> None:
        """
        Save the generated target valuation to the Catalog table.

        Args:
            target_valuation (np.ndarray): Generated target valuation data.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                ALTER TABLE Catalog ADD COLUMN target_valuation BLOB
            """)
            cursor.execute("""
                UPDATE Catalog SET target_valuation = ?
            """, (target_valuation.tobytes(),))
            conn.commit()
        self.logger.debug(__file__, "save_target_valuation", "Saved target valuation to Catalog table.", Atlas.__name__)

    def save_token_data(self, config: dict, data: List[Tuple[float, float, float, float, float, float]], target_valuation_wave: np.ndarray = None) -> None:
        """
        Save the token configuration, candle data, and optionally the target valuation wave to the database for the specified token.

        Args:
            config (dict): Token configuration dictionary.
            data (List[Tuple[float, float, float, float, float, float]]): The candle data to save.
            target_valuation_wave (np.ndarray, optional): The target valuation wave data. Defaults to None.
        """
        self._initialize_db()
        config_str = json.dumps(config)  # Convert config to JSON string
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if target_valuation_wave is not None:
                cursor.execute(f"""
                    INSERT OR REPLACE INTO {self.token} (id, config, target_valuation_wave) VALUES (1, ?, ?)
                """, (config_str, target_valuation_wave.tobytes()))
            else:
                cursor.execute(f"""
                    INSERT OR REPLACE INTO {self.token} (id, config) VALUES (1, ?)
                """, (config_str,))
            cursor.executemany(f"""
                INSERT INTO {self.token} (timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            """, data)
            conn.commit()
        self.logger.debug(__file__, "save_token_data", f"Saved config, data, and target valuation wave for {self.token}", Atlas.__name__)
        
    # GET METHODS

    def get_target_valuation(self) -> np.ndarray:
        """
        Retrieve the target valuation from the Catalog table.

        Returns:
            np.ndarray: The target valuation as a NumPy array.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT target_valuation FROM Catalog")
            target_valuation_data = cursor.fetchone()
            if target_valuation_data:
                target_valuation = np.frombuffer(target_valuation_data[0], dtype=np.float64)
                return target_valuation
            else:
                return None

    def get_token_data(self) -> Tuple[dict, List[Tuple[Any, ...]], np.ndarray]:
        """
        Retrieve the token configuration, candle data, and target valuation wave from the database for the specified token.

        Returns:
            Tuple[dict, List[Tuple[Any, ...]], np.ndarray]: The token configuration, candle data, and target valuation wave.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT config, target_valuation_wave FROM {self.token} WHERE id = 1")
            config_data = cursor.fetchone()
            if config_data:
                config = json.loads(config_data[0])
                target_valuation_wave = np.frombuffer(config_data[1], dtype=np.float64) if config_data[1] else None
            else:
                raise ValueError(f"Configuration for {self.token} not found.")
            
            cursor.execute(f"SELECT * FROM {self.token} WHERE config IS NULL")
            data = cursor.fetchall()
        self.logger.debug(__file__, "get_token_data", f"Retrieved config, data, and target valuation wave for token {self.token}", Atlas.__name__)
        return config, data, target_valuation_wave

    def get_config(self) -> dict:
        """
        Retrieve the token configuration from the database.

        Returns:
            dict: The token configuration dictionary.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = cursor.cursor()
            cursor.execute(f"SELECT config FROM {self.token} WHERE id = 1")
            config_data = cursor.fetchone()
            if config_data:
                config = json.loads(config_data[0])
                return config
            else:
                raise ValueError(f"Configuration for {self.token} not found.")

    def get_climate_cycle(self) -> np.ndarray:
        """
        Retrieve the climate cycle from the Catalog table.

        Returns:
            np.ndarray: The climate cycle as a NumPy array.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT climate_cycle FROM Catalog")
            climate_cycle_data = cursor.fetchone()
            if climate_cycle_data:
                climate_cycle = np.frombuffer(climate_cycle_data[0], dtype=np.float64)
                return climate_cycle
            else:
                return None

    def get_random_events(self) -> pd.DataFrame:
        """
        Retrieve random events from the Catalog table.

        Returns:
            pd.DataFrame: DataFrame containing random events with columns ['timestamp', 'magnitude'].
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, magnitude FROM random_events")
            events = cursor.fetchall()

        if events:
            df_events = pd.DataFrame(events, columns=['timestamp', 'magnitude'])
            df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit='s')
            return df_events
        else:
            return pd.DataFrame(columns=['timestamp', 'magnitude'])


    # MODIFYING METHODS

    def assign_token(self, new_token: str) -> None:
        """
        Reassign the current token to a new token. Creates a table for a token in the database
        if needed.

        Args:
            new_token (str): The new token name.
        """
        self.token = new_token
        # self._initialize_db()  # Ensure the new token's table is created
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {new_token} (
                    id INTEGER PRIMARY KEY,
                    config TEXT,
                    timestamp REAL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    target_valuation_wave BLOB
                )
            """)
            conn.commit()
            self.logger.debug(__file__, "create_token_table", f"Ran create-if-not-exists for table for token {new_token}.", Atlas.__name__)
        self.logger.debug(__file__, "reassign_token", f"Assigned token to {new_token}", Atlas.__name__)

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)
