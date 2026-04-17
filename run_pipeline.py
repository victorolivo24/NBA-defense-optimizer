import runpy
import sys
import subprocess
import matplotlib.pyplot as plt

def install_requirements():
    """Install dependencies from requirements.txt."""
    print(f"\n{'='*50}")
    print("Installing requirements.txt...")
    print(f"{'='*50}")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Requirements installed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def run_script(script_name: str):
    """Run a python script and return its completion status."""
    print(f"\n{'='*50}")
    print(f"Executing: {script_name}")
    print(f"{'='*50}")
    
    try:
        # run_path executes the script within the current python process,
        # allowing non-blocking matplotlib figures to persist.
        runpy.run_path(script_name, run_name="__main__")
        print(f"[{script_name}] completed successfully.\n")
    except Exception as e:
        print(f"Error executing [{script_name}]: {e}")
        sys.exit(1)

def main():
    print("Starting NBA Defensive Scheme Optimizer Pipeline...")
    
    install_requirements()
    
    scripts = [
        "ingest.py",
        "build_features.py",
        "train_model.py",
        "src/features/calculate_scheme_deltas.py",
    ]
    
    for script in scripts:
        run_script(script)
        
    print("Pipeline completed successfully!")
    print("\nKeeping plots open. Close all plot windows to exit.")
    plt.show(block=True)

if __name__ == "__main__":
    main()
