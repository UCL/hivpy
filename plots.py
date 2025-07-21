from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# files = ["output\simulation_output_20240205-230047\simulation_output_20240205-230047.csv",
#          "output\simulation_output_20240205-231331\simulation_output_20240205-231331.csv",
#          "output\simulation_output_20240206-104041\simulation_output_20240206-104041.csv",
#          "output\simulation_output_20240206-104444\simulation_output_20240206-104444.csv",
#          "output\simulation_output_20240206-104524\simulation_output_20240206-104524.csv",
#          "output\simulation_output_20240206-104650\simulation_output_20240206-104650.csv",
#          "output\simulation_output_20240206-105003\simulation_output_20240206-105003.csv",
#          "output\simulation_output_20240206-134436\simulation_output_20240206-134436.csv",
#          "output\simulation_output_20240206-135429\simulation_output_20240206-135429.csv",
#          "output\simulation_output_20240206-135605\simulation_output_20240206-135605.csv"]
files = ["output\simulation_output_20240220-154213\simulation_output_20240220-154213.csv"]
data = [pd.read_csv(file) for file in files]

men_15_24 = np.array([d["Short term partners (15-64)"] for d in data]).transpose()
medians = np.array([np.median(tstep) for tstep in men_15_24])
plt.plot(medians)
plt.show()

def psb(i, sex):
    if(sex == 0):
        sex_string = "male"
    else:
        sex_string = "female"
    return f"Partner sex balance ({i}-{i+9}, {sex_string})"

def pps(i, sex):
    if(sex == 0):
        sex_string = "male"
    else:
        sex_string = "female"
    return f"Short term partners ({i}-{i+9}, {sex})"

sexes = [0, 1]
ages = [15, 25, 35, 45, 55]

for (a,s) in product(ages, sexes):
    for d in data:
        plt.plot(1989 + 0.25 * d["Time Step"], d[psb(a, s)])
    plt.plot(1989 + 0.25 * data[0]["Time Step"], np.zeros(len(data[0])))
    print(psb(a,s), f"mean = {np.mean(d[psb(a,s)])}", f"std = {np.std(d[psb(a,s)])}", 10**(np.mean(d[psb(a,s)])))
    plt.title(psb(a, s))
    plt.show()

# Create a figure with subplots
fig, axs = plt.subplots(len(ages), len(sexes), figsize=(12, 8))
fig.suptitle("Short Term Partners and Partner Sex Balance")

# Iterate over ages and sexes
# for i, a in enumerate(ages):
#     for j, s in enumerate(sexes):
#         ax = axs[i, j]
#         ax.plot(1989 + 0.25 * data[0]["Time Step"], data[0][psb(a, s)], label=pps(a, s))
#         ax.plot(1989 + 0.25 * data[0]["Time Step"], np.zeros(len(data[0])), linestyle="--", color="gray")
#         ax.set_title(f"{psb(a, s)}")
#         ax.set_xlabel("Year")
#         ax.set_ylabel("Log Partner Balance")
#         ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
#         vals = np.ma.masked_invalid(data[0][psb(a,s)])
#         print(psb(a,s), f"mean = {np.mean(vals)}", f"std = {np.std(vals)}", f"average unbalance = {np.round((10**(np.mean(vals))-1)*100,2)}%")

# Add legend
axs[0, 0].legend()

# Adjust spacing between subplots
plt.tight_layout()

# Show the combined plot
plt.show()

# for (a,s) in product(ages, sexes):
#     for d in data:
#         plt.scatter(1989 + 0.25 * d["Time Step"], d[pps(a, s)])
#     plt.plot(1989 + 0.25 * data[0]["Time Step"], np.zeros(len(data[0])))
#     print(pps(a,s), np.mean(d[pps(a,s)]), np.std(d[pps(a,s)]), np.exp(np.mean(d[pps(a,s)])))
#     plt.title(pps(a, s))
#     plt.show()

# for d in data:
#     plt.scatter(d["Time Step"], d["Partner sex balance (15-24, male)"])
# plt.plot(data[0]["Time Step"], np.zeros(len(data[0])))
# plt.show()
# 
# for d in data:
#     plt.scatter(d["Time Step"], d["Partner sex balance (25-34, male)"])
# plt.plot(data[0]["Time Step"], np.zeros(len(data[0])))
# plt.show()
# 
# for d in data:
#     plt.scatter(d["Time Step"], d["Partner sex balance (35-44, male)"])
# plt.plot(data[0]["Time Step"], np.zeros(len(data[0])))
# plt.show()

