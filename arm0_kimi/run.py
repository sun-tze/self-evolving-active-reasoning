"""Matched-matrix entry point. Same CLI as smoke; the caller chooses the protocol."""
from .smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
