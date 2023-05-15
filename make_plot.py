import sys
import matplotlib.pyplot as plt
import pandas as pd

if len(sys.argv) < 2:
    raise Exception("File name required.")

file = sys.argv[1]
outputs = pd.read_csv(file)

plt.title("Fractional PrEP Usage in Population")
plt.plot(outputs["PrEP usage overall"], label="Overal")
plt.plot(outputs["PrEP usage (men)"], label="Men")
plt.plot(outputs["PrEP usage (women)"], label="Women")

plt.legend()
plt.show()
