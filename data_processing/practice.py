import os
import pandas as pd
import numpy as np

name = "reentrancy"
normal_data = pd.read_csv("../dataset/embedding/smart_contract/normal.csv")
print(normal_data.shape)
vul_data = pd.read_csv("../dataset/embedding/" + name + ".csv")
print(vul_data.shape)
generate_data = pd.read_csv("../dataset/generated_" + name + ".csv", index_col=0).iloc[:1000]
print(generate_data.shape)
data = pd.concat([normal_data, vul_data, generate_data], axis=0)
# data_y = data["label"].values
data_x = data.values

print(data_x)