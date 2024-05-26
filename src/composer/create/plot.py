import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

def plot_partial_ohlcv(ohlcv_df: pd.DataFrame, filename='ohlcv_partial_plot.html'):
    # Convert timestamp to human-readable dates
    ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='s')

    fig = go.Figure(data=[
        go.Candlestick(
            x=ohlcv_df['timestamp'],
            open=ohlcv_df['open'],
            high=ohlcv_df['high'],
            low=ohlcv_df['low'],
            close=ohlcv_df['close'],
            name='OHLC'
        ),
        go.Bar(
            x=ohlcv_df['timestamp'],
            y=ohlcv_df['volume'],
            name='Volume',
            marker_color='rgba(0, 0, 255, 0.3)',
            yaxis='y2'
        )
    ])
    fig.update_layout(
        title='OHLCV Data',
        yaxis=dict(
            title='Price'
        ),
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        xaxis=dict(
            title='Date',
            type='date'
        ),
        template='plotly_dark'
    )
    fig.write_html(filename)
    fig.show()  # This will open the plot in a new browser tab
    return filename

def plot_full_ohlcv(ohlcv_df, filename='ohlcv_full_plot.html'):
    # Convert timestamp to human-readable dates
    ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='s')
    
    # Resample to daily OHLCV
    daily_ohlcv_df = ohlcv_df.resample('D', on='timestamp').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Create a subplot with two rows
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, subplot_titles=('OHLCV Data', 'Volume'))

    # Add OHLC data to the first row
    fig.add_trace(go.Candlestick(
        x=daily_ohlcv_df.index,
        open=daily_ohlcv_df['open'],
        high=daily_ohlcv_df['high'],
        low=daily_ohlcv_df['low'],
        close=daily_ohlcv_df['close'],
        name='OHLC'
    ), row=1, col=1)

    # Add volume data to the second row
    fig.add_trace(go.Bar(
        x=daily_ohlcv_df.index,
        y=daily_ohlcv_df['volume'],
        name='Volume',
        marker_color='rgba(0, 0, 255, 1.0)'
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        title='Daily OHLCV Data',
        yaxis=dict(
            title='Price'
        ),
        yaxis2=dict(
            title='Volume'
        ),
        xaxis=dict(
            title='Date',
            type='date'
        ),
        template='plotly_dark'
    )
    fig.write_html(filename)
    fig.show()  # This will open the plot in a new browser tab
    return filename