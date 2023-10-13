import os
import subprocess

def convert_py_to_ipynb(py_file, ipynb_file):
    subprocess.run(
        ["jupytext", "--to", "ipynb", py_file, "--output", ipynb_file])

if not os.path.exists('colab'):
    os.makedirs('colab')

for root, dirs, files in os.walk('mean_reversion'):
    for file in files:
        if file.endswith('.py'):
            py_file_path = os.path.join(root, file)

            relative_root = os.path.relpath(root, 'mean_reversion')
            new_root = os.path.join('colab', relative_root)
            if not os.path.exists(new_root):
                os.makedirs(new_root)

            ipynb_file_path = os.path.join(new_root,
                                           file.replace('.py', '.ipynb'))

            convert_py_to_ipynb(py_file_path, ipynb_file_path)
