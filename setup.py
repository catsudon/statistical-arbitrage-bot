"""Shim so `pip install -e .` works on older setuptools (pre-PEP 660).
All real configuration lives in `pyproject.toml`."""
from setuptools import setup

setup()
