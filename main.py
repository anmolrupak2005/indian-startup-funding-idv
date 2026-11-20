"""
main.py
Main Execution Orchestrator for Indian Startup Funding IDV Pipeline.
Executes:
  1. Data Collection & Loading (src/load.py)
  2. Data Cleaning & Standardization (src/clean.py)
  3. Feature Engineering (src/features.py)
  4. Exploratory Data Analysis & Visualization (src/eda.py)
"""

import sys
import os
import time

# Add src to system path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from load import load_all_data
from clean import clean_dataset
from features import engineer_features
from eda import run_eda

def main():
    start_time = time.time()
    print("\n" + "=" * 70)
    print("   INDIAN STARTUP FUNDING TRENDS: INTERACTIVE VISUAL ANALYSIS")
    print("   IDV 7th Semester B.Tech CSE Capstone Project")
    print("=" * 70 + "\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Step 1: Data Collection & Loading
        raw_df = load_all_data(data_dir=data_dir)

        # Step 2: Data Cleaning
        clean_df = clean_dataset(raw_df, output_dir=output_dir)

        # Step 3: Feature Engineering
        featured_df = engineer_features(clean_df, output_dir=output_dir)

        # Step 4: Exploratory Data Analysis & Chart Generation
        charts_dir, insights = run_eda(featured_df, output_dir=output_dir)

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("   PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Total Processing Time: {elapsed:.2f} seconds\n")
        print("Artifacts Generated:")
        print(f"  1. Cleaned CSV Data:  {os.path.join(output_dir, 'clean_startup_funding.csv')}")
        print(f"  2. Cleaning Report:   {os.path.join(output_dir, 'cleaning_report.txt')}")
        print(f"  3. Visual Charts (9): {charts_dir}/")
        print(f"  4. Insights Summary:  {os.path.join(output_dir, 'insights.md')}")
        print("\nTo launch the interactive dashboard, run:")
        print("   streamlit run app.py")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n[!] Pipeline Failed with Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
