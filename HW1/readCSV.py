import csv
import numpy as np


subdata = np.zeros((64**3, 49), dtype=np.complex128)
with open('subdata.csv') as f:
    for i, line in enumerate(csv.reader(f)):
        for j, num_str in enumerate(line):
            subdata[i, j] = complex(num_str.replace("i", "j").replace("+-", "-"))

np.save("subdata.npy", subdata)
