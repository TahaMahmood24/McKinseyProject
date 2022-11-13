import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ss = pd.DataFrame({'Car': ['Honda','Toyota','Suzuki','Mitsubishi','KIA'],'Price': [30000,25000,15000,20000,25000]},columns=['Car','Price'])
#print(ss.index)
ss = ss.sort_values('Price')
ss = ss.assign(New_Index = range(len(ss)))
ss = ss.assign(New_Price=ss['New_Index']*2)
x = 12.00
#print("{:.6f}".format(x))

print(ss.head())
plt.plot(ss.New_Index,ss.New_Price, color='blue')
plt.show()
error=[]
for i in ss:
    error.append(i*2)
    #print("The error for value {} is {}".format(i,i*2))
#print(error)
