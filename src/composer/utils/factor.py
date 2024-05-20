import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go

class Factor:
    def __init__(self, name: str, start: int, end: int, weight: float) -> None:
        """
        Initialize a Factor object.

        Args:
            name (str): Name of the factor.
            start (int): Start timestamp in UTC seconds format.
            end (int): End timestamp in UTC seconds format.
            weight (float): Weight of the factor.
        """
        self.name = name
        self.start = start
        self.end = end
        self.weight = weight
        self.data = np.array([])

    def generate_sine(self, frequency: float, amplitude: float, slope: float, noise_level: float = 0.05, 
                      prev_half_chunk: np.ndarray = None, start_price: float = None, chunk_size: int = 432000) -> np.ndarray:
        """
        Generate a sine wave for the next chunk of the factor data.

        Args:
            frequency (float): Frequency of the sine wave.
            amplitude (float): Amplitude of the sine wave.
            slope (float): Slope of the sine wave.
            noise_level (float): Noise level to add to the sine wave.
            prev_half_chunk (np.ndarray, optional): Previous half chunk of data.
            start_price (float, optional): Starting price for the first chunk.
            chunk_size (int): Size of the chunk to generate in seconds (default is 5 days).

        Returns:
            np.ndarray: Generated chunk of factor data.
        """
        if prev_half_chunk is not None:
            half_chunk_size = len(prev_half_chunk)
            start_time = self.start + half_chunk_size
        elif start_price is not None:
            half_chunk_size = 0
            start_time = self.start
        else:
            raise ValueError("Either prev_half_chunk or start_price must be provided")

        if start_time + chunk_size > self.end:
            raise ValueError("Cannot generate chunk beyond the end time")

        t = np.linspace(start_time, start_time + chunk_size - 1, chunk_size)
        noise = np.random.normal(0, noise_level, chunk_size)
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * t / chunk_size)
        trend_line = slope * np.linspace(0, chunk_size, chunk_size)
        wave = sine_wave + noise + trend_line
        wave *= np.random.uniform(0.95, 1.05, chunk_size)

        # Normalize the wave to be around 1.0
        wave = 1 + (wave / np.max(np.abs(wave)))

        if prev_half_chunk is not None:
            self.data = np.concatenate((prev_half_chunk, wave[half_chunk_size:]))
        else:
            self.data = wave

        return self.data

    def plot(self) -> go.Figure:
        """
        Create a Plotly graph of the generated data.

        Returns:
            go.Figure: Plotly figure of the generated data.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(self.data)) + self.start, y=self.data, mode='lines', name=self.name))
        fig.update_layout(
            title=f"Generated Factor: {self.name}",
            xaxis_title="Time (UTC seconds)",
            yaxis_title="Value",
            template="plotly_dark"
        )
        return fig
