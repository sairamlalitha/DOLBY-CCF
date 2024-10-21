import subprocess

def run_gaussian_fit():
    print("Running Gaussian fit...")
    result = subprocess.run(['python', 'gaussian_fit.py'], capture_output=True, text=True)
    if result.returncode == 0:
        print("Gaussian fit completed successfully.")
        print(result.stdout)  # Print output if necessary
    else:
        print("Error running Gaussian fit:")
        print(result.stderr)

def run_gp_fit():
    print("Running Gaussian Process fit...")
    result = subprocess.run(['python', 'gp_fit.py'], capture_output=True, text=True)
    if result.returncode == 0:
        print("Gaussian Process fit completed successfully.")
        print(result.stdout)  # Print output if necessary
    else:
        print("Error running Gaussian Process fit:")
        print(result.stderr)

if __name__ == "__main__":
    # Run the Gaussian fit first
    run_gaussian_fit()

    # Run the Gaussian Process fit afterwards
    run_gp_fit()
