import csv
import os
from pathlib import Path

# A program to check why csv file is not read
# pwd
# /home/zepho/work/chain
# (chain)  python chaining/csv_test.py

def read_csv_file():
    # Get absolute path to the script's directory
    script_dir = Path(__file__).resolve().parent

    # Define the CSV file path relative to the script
    csv_path = script_dir / 'data.csv'

    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Looking for CSV at: {csv_path}")

    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            print("\nHeader:", header)
            for i, row in enumerate(csv_reader, 1):
                print(f"Line {i}:", row)
    except FileNotFoundError:
        print(f"\nError: Could not find CSV file at {csv_path}")
        print("\nMake sure data.csv is in the same directory as this script.")

if __name__ == "__main__":
    read_csv_file()