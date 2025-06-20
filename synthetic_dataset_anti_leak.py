'''
synthetic_dataset_anti_leak.py which will be responsible for running the synthetic dataset generation process with the following features:

- OS Detection: The script will automatically detect whether it is running on Windows or Linux and will use the appropriate command.
- Time-controlled execution: The script will execute the command and monitor its execution time.
- Automatic restart: If the execution exceeds a time limit X (configurable, with 10 minutes as the default), the script will stop the process and restart it.
- Stop condition: The restart cycle will stop when a complete execution of the process takes less than the time limit X.

+---------+
|  Start  |
+----+----+
     |
     v
+----+------------------------+
|      Determine OS           |
+----+------------------------+
     |                        |
     |Windows                 |Linux
     v                        v
+-------------------+   +-----------------------+
| Command:          |   | Command:              |
| python -m         |   | blenderproc run ...   |
| blenderproc ...   |   +-----------------------+
+-------------------+           |
          \                     /
           \                   /
            \                 /
             v               v
          +----------------------+
          | Start main loop      |
          +----------+-----------+
                     |
                     v
          +----------------------+
          | Log start time       |
          +----------+-----------+
                     |
                     v
          +----------------------+
          | Launch generation    |
          | process             |
          +----------+-----------+
                     |
                     v
         +-------------------------------+
         |   Monitor process             |
         +-------------------------------+
             |                    |
     Process finished?       Time > X?
         |                       |
         v                       v
+-------------------+      +----------------------+
| Time < X?         |      | Terminate process    |
+----+--------------+      +----------+-----------+
     |Yes                   |
     v                      v
+---------+            +-------------------+
|  End    |<-----------| Back to main loop |
+---------+            +-------------------+
     |
     No
     |
     v
+-------------------+
| Back to main loop |
+-------------------+

'''
import subprocess
import time
import sys
import os
import argparse

def get_command():
    """Determines the correct command based on the operating system."""
    if sys.platform.startswith('linux'):
        return ["blenderproc", "run", "dataset_generator/03_generate_synthetic_dataset.py"]
    elif sys.platform == "win32":
        return ["python", "-m", "blenderproc", "run", "dataset_generator/03_generate_synthetic_dataset.py"]
    else:
        print(f"Unsupported operating system: {sys.platform}")
        sys.exit(1)

def main(timeout_minutes):
    """
    Runs the dataset generation script with a timeout.
    It will restart the process if it exceeds the timeout, and stop when
    a run completes in less than the timeout.
    """
    command = get_command()
    timeout_seconds = timeout_minutes * 60
    
    while True:
        print(f"Starting process: {' '.join(command)}")
        print(f"Timeout set to {timeout_minutes} minutes.")
        
        start_time = time.time()
        process = subprocess.Popen(command)

        while process.poll() is None:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                print(f"Process exceeded timeout of {timeout_minutes} minutes. Terminating...")
                process.terminate()
                try:
                    # Wait a bit for graceful termination
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("Process did not terminate gracefully. Forcing kill...")
                    process.kill()
                print("Process terminated. Restarting...")
                break  # Break inner loop to restart the process
            time.sleep(1)
        
        # If the loop was not broken by timeout, the process finished
        if process.poll() is not None:
            end_time = time.time()
            execution_time = end_time - start_time
            if execution_time <= timeout_seconds:
                print(f"Process finished successfully in {execution_time:.2f} seconds.")
                print("Execution time was within the limit. Exiting.")
                break # Exit the main while loop
            else:
                # This case is handled by the timeout check inside the inner while,
                # but we keep it for clarity.
                print("Process finished, but took too long. Restarting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the synthetic dataset generator with a timeout and restart logic."
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=10,
        help="Timeout in minutes for the process. Default is 10."
    )
    args = parser.parse_args()
    
    main(args.timeout)
