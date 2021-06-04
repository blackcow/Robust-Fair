
import numpy as np
import pandas as pd

name = 'fair_12_.csv'
df = pd.read_csv("results/" + name, header = None)

#df = df.drop([27])

test = df.sort_values(0, ascending=[False])
test1 = df.sort_values(1, ascending=[False])


#import matplotlib.pyplot as plt

array = np.array(test1[1])
array1 = np.array(test[0])
labels = np.array(range(23))

#plt.plot(labels, array1, label = 'Clean Acc. (FRL)', color = 'green', linewidth = 5, linestyle = '-')
#plt.plot(labels, array, label = 'PGD-16 Acc.(FRL)', color = 'red', linewidth = 3, linestyle = '-')
#print('variance after debias' + str(np.std(array)))

#################################

'''
name = 'adv_12_.csv'
df = pd.read_csv("results/" + name, header = None)

#df = df.drop([27])

test = df.sort_values(0, ascending=[False])
test1 = df.sort_values(1, ascending=[False])

array = np.array(test1[1])
array1 = np.array(test[0])
labels = np.array(range(43))


plt.plot(labels, array1, label = 'Clean Acc.', color = 'blue', linewidth = 3)
plt.plot(labels, array, label = 'PGD-12 Acc.', color = 'orange', linewidth = 3)
print('variance before debias ' + str(np.std(array)))
'''

#############################
input(123)
plt.tick_params(axis="y", labelsize=20)
plt.tick_params(axis="x", labelsize=10)
plt.legend(fontsize = 15, loc = 2)
plt.xlabel('Class Index', fontsize = 20)
plt.title('Natural Model (GTSRB)', fontsize = 20)
plt.ylim(0,1)
plt.grid(axis = 'y')
plt.savefig('results/' + name + '1.png')
