import subprocess
import sys
import os

def run_data_transformation():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')

    command = [sys.executable, '-m', 'src.components.data_transformation']

    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
        print("Data transformation output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error during data transformation:")
        print(e.stderr)

if __name__ == "__main__":
    run_data_transformation()
