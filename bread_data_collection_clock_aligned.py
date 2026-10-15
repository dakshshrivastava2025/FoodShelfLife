"""CSV-only collector aligned to interval boundaries (5 minutes by default)."""
from bread_data_collection import main

if __name__ == '__main__':
    main(clock_aligned=True)
