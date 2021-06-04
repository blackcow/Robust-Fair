
import numpy as np
import pandas as pd

name = 'fairtra_.csv'
df = pd.read_csv("results/" + name, header = None)
print(df)

test = df.sort_values(0, ascending=[False])
print(test)

import matplotlib.pyplot as plt

array = np.array(test[1])
array1 = np.array(test[0])
labels = np.array(range(43))

plt.plot(labels, array1, label = 'Clean Acc.', linewidth = 3)
plt.plot(labels, array, label = 'PGD-12 Acc.', color = 'red', linewidth = 2)
plt.tick_params(axis="y", labelsize=20)
plt.tick_params(axis="x", labelsize=10)
plt.legend(fontsize = 20, loc = 2)
plt.xlabel('Class Index', fontsize = 20)
plt.title('Fair Robust Model (GTSRB)', fontsize = 20)
plt.ylim(0,1)

plt.savefig('results/' + name + '.png')
