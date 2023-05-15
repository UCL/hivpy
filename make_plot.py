import matplotlib.pyplot as plt
import pandas as pd

outputs = pd.read_csv(r'output\simulation_output_20230515-211358.csv')

plt.plot(outputs["PrEP usage overall"], label="Overal")
plt.plot(outputs["PrEP usage (men)"], label="Men")
plt.plot(outputs["PrEP usage (women)"], label="Women")

plt.legend()
plt.show()
