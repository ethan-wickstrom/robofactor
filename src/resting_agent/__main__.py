"""
Main entry point for the Resting Agent CLI.

This module serves as the entry point when running the package as a module:
    python -m resting_agent

or after installation:
    resting-agent
"""

from .core.cli import main

if __name__ == "__main__":
    main()
