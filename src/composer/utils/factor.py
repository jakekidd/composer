import numpy as np
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
        self.sine_daily = np.array([])  # Initialize sine_daily property

    def _generate_full_period_sine(self, frequency: float, amplitude: float, slope: float, noise_level: float, period_days: float) -> np.ndarray:
        """
        Generate a full period sine wave for the entire time span.

        Args:
            frequency (float): Frequency of the sine wave.
            amplitude (float): Amplitude of the sine wave.
            slope (float): Slope of the sine wave.
            noise_level (float): Noise level to add to the sine wave.
            period_days (float): Period of the sine wave in days.

        Returns:
            np.ndarray: Generated full period sine wave.
        """
        period_seconds = period_days * 24 * 3600
        total_seconds = self.end - self.start
        adjusted_frequency = 2 * np.pi / period_seconds

        t = np.linspace(0, total_seconds, total_seconds)
        sine_wave = amplitude * np.sin(adjusted_frequency * t)

        # Normalize time vector to [0, 1] range for the trend line
        normalized_t = t / total_seconds
        trend_line = slope * normalized_t

        wave = sine_wave + trend_line

        # Normalize the wave to be between 0.0 and 15.0
        wave_min = np.min(wave)
        wave_max = np.max(wave)
        wave = 15 * (wave - wave_min) / (wave_max - wave_min)

        return wave

    def generate_sine(self, frequency: float, amplitude: float, slope: float, noise_level: float = 0.05, 
                      period_days: float = 30, prev_half_chunk: np.ndarray = None, start_price: float = None, chunk_size: int = 432000) -> np.ndarray:
        """
        Generate a sine wave for the next chunk of the factor data.

        Args:
            frequency (float): Frequency scaling factor for the sine wave.
            amplitude (float): Amplitude of the sine wave.
            slope (float): Slope of the sine wave.
            noise_level (float): Noise level to add to the sine wave.
            period_days (float): Period of the sine wave in days.
            prev_half_chunk (np.ndarray, optional): Previous half chunk of data.
            start_price (float, optional): Starting price for the first chunk.
            chunk_size (int): Size of the chunk to generate in seconds (default is 5 days).

        Returns:
            np.ndarray: Generated chunk of factor data.
        """
        if self.sine_daily.size == 0:
            self.sine_daily = self._generate_full_period_sine(frequency, amplitude, slope, noise_level, period_days)

        if prev_half_chunk is not None:
            half_chunk_size = len(prev_half_chunk)
            start_time = self.start + half_chunk_size
        elif start_price is not None:
            half_chunk_size = 0
            start_time = self.start
            # Start normalization at 1.0 if starting from a given price
            self.data = np.ones(chunk_size)
        else:
            raise ValueError("Either prev_half_chunk or start_price must be provided")

        if start_time + chunk_size > self.end:
            raise ValueError("Cannot generate chunk beyond the end time")

        # Extract the relevant section of the sine_daily for this chunk
        start_index = start_time - self.start
        end_index = start_index + chunk_size
        sine_chunk = self.sine_daily[start_index:end_index]

        noise = np.random.normal(0, noise_level, chunk_size)
        wave = sine_chunk + noise

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
