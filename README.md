# Composer

Composer is designed to generate realistic synthetic price data for tokens, to be used as a training data for developing trading strategy bots. This system simulates market behavior by considering various influencing factors and generates price data over specified periods. The primary script, `generate.py`, orchestrates the generation process, incorporating cycles, random walks, and external factors to produce price data which gets converted to OHLCV data. Graphs are illustrated using plotly.

NOTE: Using this data for training a bot is not guaranteed to function well in production, but can illuminate whether a bot concept, for example, is feasible, without paying for 1s OHLCV data.

## Installation

Ensure you are using python 3.9+. To install all required dependencies:
```bash
python3 -m pip -r requirements.txt
```

## Usage

To generate synthetic price data, run the `generate.py` script:

```bash
python3 generate.py
```

For each configured token, will generate the price data first, save to a database under data/market.db, and then convert to OHLCV candle data and save that to the database as well. All logs are saved under logs/app.log. 

## Configuration

The token configuration is defined in the `TOKENS` dictionary within `generate.py`. You can customize the parameters such as `start`, `end`, `stable`, and the attributes of the tokens.

Example configuration:
```python
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
            "factors": []
        }
    ]
}
```
This will be represented by a JSON file in the future.

## Factors

We generate a normalized valuation wave and combine it with the base wave to form a combined wave, which influences the trendline of the generated price data. In the future, will generate other external factors like random events (e.g. good news, bad news, etc.), fiduciary climate, liquidations, short squeezes, paradigm shifts, etc.

## License

This project is licensed under the MIT License.
