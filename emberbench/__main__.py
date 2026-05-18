"""Entry point for `python -m eval.emberbench`."""
from eval.emberbench.cli import main

if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
