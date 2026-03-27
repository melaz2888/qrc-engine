"""Benchmark tasks for reservoir computing."""

from qrc_engine.tasks.narma import narma10
from qrc_engine.tasks.timeseries import mackey_glass, sine_forecasting

__all__ = ["narma10", "mackey_glass", "sine_forecasting"]
