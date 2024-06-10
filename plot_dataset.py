import pandas as pd
import matplotlib.pyplot as plt

#Dataframe for Pressure Sensors 
dfp1 = pd.read_csv('PS1.txt' , sep='\t', header=None)
dfp2 = pd.read_csv('PS2.txt' , sep='\t', header=None)
dfp3 = pd.read_csv('PS3.txt' , sep='\t', header=None)
dfp4 = pd.read_csv('PS4.txt' , sep='\t', header=None)
dfp5 = pd.read_csv('PS5.txt' , sep='\t', header=None)
dfp6 = pd.read_csv('PS6.txt' , sep='\t', header=None)

#Dataframe for Temperature Sensors
dft1 =  pd.read_csv('TS1.txt' , sep='\t', header=None)
dft2 =  pd.read_csv('TS2.txt' , sep='\t', header=None)
dft3 =  pd.read_csv('TS3.txt' , sep='\t', header=None)
dft4 =  pd.read_csv('TS4.txt' , sep='\t', header=None)


#Dataframe for Motor Power Sensor
dfe1 =  pd.read_csv('EPS1.txt' , sep='\t', header=None)

#Dataframe for Volume Flow Sensor
dff1 = pd.read_csv('FS1.txt' , sep='\t', header=None)
dff2 = pd.read_csv('FS2.txt' , sep='\t', header=None)

#Dataframe for Vibration Sensor
dfv1 = pd.read_csv('VS1.txt' , sep='\t', header=None)

#Dataframe for Efficiecy Factor
dfs1 = pd.read_csv('SE.txt' , sep='\t', header=None)

#Dataframe for Virtual Cooling Efficiency Sensor
dfce1 = pd.read_csv('CE.txt' , sep='\t', header=None)

#Dataframe for Virtual Cooling Power Sensor
dfcp1 = pd.read_csv('CP.txt' , sep='\t', header=None)

#print(dfp1)
#print(dft1)
#print(dfe1)
#print(dff1)
#print(dfv1)
#print(dfs1)
#print(dfce1)
#print(dfcp1)

#dfp1.iloc[0:10].plot()
#dft1.iloc[0:10].plot()
#dfe1.iloc[0:10].plot()
#dff1.iloc[0:10].plot()
#dfv1.iloc[0:10].plot()
#dfs1.iloc[0:10].plot()
#dfce1.iloc[0:10].plot()
#dfcp1.iloc[0:10].plot()
#plt.show()
