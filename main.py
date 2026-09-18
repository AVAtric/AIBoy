#!/usr/bin/env python
"""AIboy entry point.

    python main.py                 # the app (wizard opens first)
    python main.py train ...       # train headless
    python main.py play ...        # watch a trained agent in a window
    python main.py --help

See aiboy/cli.py for the commands and README.md for everything else.
"""
from aiboy.cli import main

if __name__ == "__main__":
    main()
