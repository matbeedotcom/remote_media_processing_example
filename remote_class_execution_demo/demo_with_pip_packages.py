"""
Demo: Remote Execution with Pip Package Dependencies

This example shows how to execute Python code remotely that requires
external pip packages to be installed on the remote server.
"""

import asyncio
import sys
import os

# Add the parent directory to path so we can import remotemedia
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from remotemedia.core.node import RemoteExecutorConfig
from remotemedia.remote.proxy_client import RemoteProxyClient


class NumpyProcessor:
    """Example class that requires numpy."""
    
    def __init__(self, name: str = "NumpyProcessor"):
        self.name = name
    
    def process_array(self, data: list) -> dict:
        """Process array using numpy operations."""
        import numpy as np
        
        arr = np.array(data)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "sum": float(np.sum(arr))
        }
    
    def matrix_operations(self, matrix_a: list, matrix_b: list) -> dict:
        """Perform matrix operations using numpy."""
        import numpy as np
        
        a = np.array(matrix_a)
        b = np.array(matrix_b)
        
        return {
            "dot_product": a.dot(b).tolist(),
            "element_wise_multiply": (a * b).tolist(),
            "transpose_a": a.T.tolist(),
            "determinant_a": float(np.linalg.det(a)) if a.shape[0] == a.shape[1] else None
        }


class PandasProcessor:
    """Example class that requires pandas."""
    
    def __init__(self, name: str = "PandasProcessor"):
        self.name = name
    
    def analyze_data(self, data: list) -> dict:
        """Analyze data using pandas."""
        import pandas as pd
        
        df = pd.DataFrame(data)
        
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "description": df.describe().to_dict(),
            "null_counts": df.isnull().sum().to_dict()
        }
    
    def process_timeseries(self, dates: list, values: list) -> dict:
        """Process time series data."""
        import pandas as pd
        
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'value': values
        })
        df.set_index('date', inplace=True)
        
        # Resample to daily frequency and calculate statistics
        daily = df.resample('D').agg(['mean', 'min', 'max', 'count'])
        
        return {
            "daily_stats": daily.to_dict(),
            "total_days": len(daily),
            "missing_days": (daily['value']['count'] == 0).sum()
        }


class ScipyProcessor:
    """Example class that requires scipy."""
    
    def __init__(self, name: str = "ScipyProcessor"):
        self.name = name
    
    def signal_processing(self, signal: list, sample_rate: float = 1000.0) -> dict:
        """Process signal using scipy."""
        import numpy as np
        from scipy import signal as sp_signal
        from scipy.fft import fft, fftfreq
        
        # Convert to numpy array
        data = np.array(signal)
        
        # Apply a Butterworth filter
        b, a = sp_signal.butter(4, 100, 'low', fs=sample_rate)
        filtered = sp_signal.filtfilt(b, a, data)
        
        # Compute FFT
        N = len(data)
        yf = fft(data)
        xf = fftfreq(N, 1/sample_rate)[:N//2]
        
        # Find peaks
        peaks, properties = sp_signal.find_peaks(filtered, height=0)
        
        return {
            "filtered_signal": filtered.tolist()[:100],  # First 100 samples
            "peak_count": len(peaks),
            "peak_indices": peaks.tolist()[:10],  # First 10 peaks
            "dominant_frequency": float(xf[np.argmax(2.0/N * np.abs(yf[0:N//2]))])
        }


async def demo_numpy_processing():
    """Demonstrate remote execution with numpy dependency."""
    print("\n" + "="*60)
    print("DEMO: Remote Execution with NumPy")
    print("="*60)
    
    # Configure with pip packages
    config = RemoteExecutorConfig(
        host="localhost", 
        port=50052, 
        ssl_enabled=False,
        pip_packages=["numpy"]  # Specify required packages
    )
    
    async with RemoteProxyClient(config) as client:
        processor = NumpyProcessor()
        remote_processor = await client.create_proxy(processor)
        
        # Test array processing
        print("\n1. Array Processing:")
        data = [1.5, 2.3, 3.7, 4.1, 5.9, 6.2, 7.8, 8.4, 9.1, 10.5]
        result = await remote_processor.process_array(data)
        print(f"   Input: {data}")
        for key, value in result.items():
            print(f"   {key}: {value:.4f}")
        
        # Test matrix operations
        print("\n2. Matrix Operations:")
        matrix_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        matrix_b = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        result = await remote_processor.matrix_operations(matrix_a, matrix_b)
        print(f"   Matrix A: {matrix_a}")
        print(f"   Matrix B: {matrix_b}")
        print(f"   Dot Product: {result['dot_product']}")
        print(f"   Determinant of A: {result['determinant_a']}")


async def demo_pandas_processing():
    """Demonstrate remote execution with pandas dependency."""
    print("\n" + "="*60)
    print("DEMO: Remote Execution with Pandas")
    print("="*60)
    
    config = RemoteExecutorConfig(
        host="localhost", 
        port=50052, 
        ssl_enabled=False,
        pip_packages=["pandas", "numpy"]  # Pandas requires numpy
    )
    
    async with RemoteProxyClient(config) as client:
        processor = PandasProcessor()
        remote_processor = await client.create_proxy(processor)
        
        # Test data analysis
        print("\n1. Data Analysis:")
        data = [
            {"name": "Alice", "age": 25, "score": 85.5},
            {"name": "Bob", "age": 30, "score": 92.0},
            {"name": "Charlie", "age": 35, "score": 78.5},
            {"name": "David", "age": 28, "score": 88.0},
            {"name": "Eve", "age": 32, "score": 95.5}
        ]
        result = await remote_processor.analyze_data(data)
        print(f"   Shape: {result['shape']}")
        print(f"   Columns: {result['columns']}")
        print(f"   Age stats: mean={result['description']['age']['mean']:.1f}, std={result['description']['age']['std']:.1f}")
        
        # Test time series
        print("\n2. Time Series Processing:")
        dates = ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-05"]
        values = [10, 15, 20, 25, 30, 35]
        result = await remote_processor.process_timeseries(dates, values)
        print(f"   Total days: {result['total_days']}")
        print(f"   Missing days: {result['missing_days']}")


async def demo_scipy_processing():
    """Demonstrate remote execution with scipy dependency."""
    print("\n" + "="*60)
    print("DEMO: Remote Execution with SciPy")
    print("="*60)
    
    config = RemoteExecutorConfig(
        host="localhost", 
        port=50052, 
        ssl_enabled=False,
        pip_packages=["scipy", "numpy"]  # SciPy requires numpy
    )
    
    async with RemoteProxyClient(config) as client:
        processor = ScipyProcessor()
        remote_processor = await client.create_proxy(processor)
        
        # Generate a test signal
        import numpy as np
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + 0.2 * np.random.randn(len(t))
        
        print("\n1. Signal Processing:")
        print(f"   Signal length: {len(signal)} samples")
        print(f"   Sample rate: 1000 Hz")
        
        result = await remote_processor.signal_processing(signal.tolist(), 1000.0)
        print(f"   Peak count: {result['peak_count']}")
        print(f"   First few peak indices: {result['peak_indices']}")
        print(f"   Dominant frequency: {result['dominant_frequency']:.2f} Hz")


async def demo_multiple_packages():
    """Demonstrate using multiple packages together."""
    print("\n" + "="*60)
    print("DEMO: Multiple Package Dependencies")
    print("="*60)
    
    class MultiPackageProcessor:
        """Uses multiple scientific packages."""
        
        def complex_analysis(self, data: list) -> dict:
            """Perform analysis using multiple packages."""
            import numpy as np
            import pandas as pd
            from scipy import stats
            
            # Convert to numpy array
            arr = np.array(data)
            
            # Create pandas series for additional analysis
            series = pd.Series(data)
            
            # Perform statistical tests
            shapiro_stat, shapiro_p = stats.shapiro(arr)
            
            return {
                "numpy_mean": float(np.mean(arr)),
                "pandas_median": float(series.median()),
                "scipy_skewness": float(stats.skew(arr)),
                "scipy_kurtosis": float(stats.kurtosis(arr)),
                "is_normal": shapiro_p > 0.05,
                "shapiro_p_value": float(shapiro_p)
            }
    
    config = RemoteExecutorConfig(
        host="localhost", 
        port=50052, 
        ssl_enabled=False,
        pip_packages=["numpy", "pandas", "scipy"]
    )
    
    async with RemoteProxyClient(config) as client:
        processor = MultiPackageProcessor()
        remote_processor = await client.create_proxy(processor)
        
        # Test with sample data
        data = [2.5, 3.1, 2.8, 3.5, 4.2, 3.9, 3.7, 4.1, 3.3, 3.6]
        result = await remote_processor.complex_analysis(data)
        
        print("\nComplex Analysis Results:")
        for key, value in result.items():
            print(f"   {key}: {value}")


async def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("REMOTE EXECUTION WITH PIP PACKAGE DEPENDENCIES")
    print("="*80)
    print("\nThis demo shows how to execute Python code remotely that requires")
    print("external pip packages to be installed on the remote server.")
    
    print("\n⚠️  Prerequisites:")
    print("1. Ensure the remote execution server is running:")
    print("   cd ../../remote_service")
    print("   python src/server.py")
    
    # input("\nPress Enter to start the demonstrations...")
    
    demos = [
        ("NumPy Processing", demo_numpy_processing),
        ("Pandas Processing", demo_pandas_processing),
        ("SciPy Processing", demo_scipy_processing),
        ("Multiple Packages", demo_multiple_packages),
    ]
    
    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\n❌ Error in {name} demo: {e}")
            print("   Make sure the remote server is running!")
        
        if demo_func != demos[-1][1]:  # Don't wait after last demo
            pass  # input("\nPress Enter to continue to next demo...")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("- Specify pip packages in RemoteExecutorConfig")
    print("- Packages are installed automatically on the remote server")
    print("- Virtual environments isolate dependencies per session")
    print("- Supports any pip-installable package")
    print("- Multiple packages can be specified as a list")


if __name__ == "__main__":
    asyncio.run(main())