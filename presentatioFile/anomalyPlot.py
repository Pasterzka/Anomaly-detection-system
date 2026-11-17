import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_csv('presentatioFile/dane1.csv')

point_data = [ 20, 17, 21, 18, 19, 21, 22, 18, 20, 52, 20, 17, 21, 18, 19, 21, 22, 18, 20,]
point_anomaly = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
df_point = pd.DataFrame({"Value": point_data, "Anomaly": point_anomaly})

collective_data = [ 20, 17, 21, 18, 19, 21, 22, 18, 50, 52, 51, 54, 21, 18, 19, 21, 22, 18, 20,]
collective_data_anomaly = [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
df_collective= pd.DataFrame({"Value": collective_data, "Anomaly": collective_data_anomaly})

contextual_data = [ 40, 10, 9, 11, -10, 12, 8, 8, 41, 9, 10, 10, -12, 12, 11, 10, 10, 10, 12, 13, -11]
contextual_data_anomaly = [ 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
df_contextual = pd.DataFrame({"Value": contextual_data, "Anomaly": contextual_data_anomaly})

anomaly_point_data = df_point[df_point["Anomaly"] == 1]
anomaly_collective_data = df_collective[df_collective["Anomaly"] == 1]
anomaly_contextual_data = df_contextual[df_contextual["Anomaly"] == 1]

plt.figure(figsize=(10, 6))
plt.plot(point_data, linestyle='-', color='b', label='Value')

plt.scatter( anomaly_point_data.index, anomaly_point_data["Value"], color="red", marker="o", label="Anomaly")

plt.title('Point Anomaly Plot', fontsize=14)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.savefig('presentatioFile/wykres1.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(collective_data, linestyle='-', color='b', label='Value')

plt.scatter( anomaly_collective_data.index, anomaly_collective_data["Value"], color="red", marker="o", label="Anomaly")

plt.title('Collective Anomaly Plot', fontsize=14)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.savefig('presentatioFile/wykres2.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(contextual_data, linestyle='-', color='b', label='Value')

plt.scatter( anomaly_contextual_data.index, anomaly_contextual_data["Value"], color="red", marker="o", label="Anomaly")

plt.title('Contextual Anomaly Plot', fontsize=14)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.savefig('presentatioFile/wykres3.png')
plt.show()