import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
storage_folder = os.path.join(script_dir, "table_storage")

file_name_input = 'high_d_low_sigma'
file_name_output = 'high_d_low_sigma_table.txt'

file_path = os.path.join(storage_folder, file_name_input + ".npy")
collection = np.load(file_path)

file_path = os.path.join(storage_folder, file_name_input + ".txt")
with open(file_path, "r") as f:
    lines = f.readlines()

row_labels = eval(lines[0].strip())
row_labels = [str(x) for x in row_labels]
column_labels = eval(lines[1].strip())


means = np.mean(collection, axis=1)
stds = np.std(collection, axis=1)

# Start LaTeX table
latex = "\\begin{table}\n\\centering\n\\begin{tabular}{l" + "c" * len(column_labels) + "}\n"
latex += "$n$ & " + " & ".join(column_labels) + " \\\\\n"
latex += "\\hline\n"

for i in range(len(row_labels)):
    formatted_row = [f"${means[i][j]:.4g} \pm {stds[i][j]:.4g}$" for j in range(len(column_labels))]
    row_str = " & ".join(formatted_row)
    latex += f"{row_labels[i]} & {row_str} \\\\\n"

latex += "\\end{tabular}\n\\end{table}"

file_path = os.path.join(storage_folder, file_name_output)
with open(file_path, "w") as f:
    f.write(latex)