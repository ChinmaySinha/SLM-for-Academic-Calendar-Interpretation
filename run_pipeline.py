import subprocess
import sys

def run_script(script_path):
    """Executes a Python script and checks for errors."""
    try:
        print(f"--- Running {script_path} ---")
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"--- Finished {script_path} ---\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print(e.stderr)
        sys.exit(1)

def main():
    print("Starting the data processing pipeline...\n")

    scripts = [
        "src/ingest_data.py",
        "src/normalize_dates.py",
        "src/categorize_events.py",
        "src/baseline_retrieval.py"
    ]

    for script in scripts:
        run_script(script)

    print("Data processing pipeline completed successfully!")

if __name__ == '__main__':
    main()
