import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('presentatioFile/dane.csv')

plt.figure(figsize=(10, 6))
plt.plot(df.iloc[:, 0], marker='o', linestyle='-', color='b', label='Wartości')

plt.title('Wykres danych', fontsize=14)
plt.xlabel('Indeks', fontsize=12)
plt.ylabel('Wartość', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.savefig('presentatioFile/wykres.png')
plt.show()