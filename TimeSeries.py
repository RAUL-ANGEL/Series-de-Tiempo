import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos
url = 'https://gist.githubusercontent.com/tuanorai/660f1966f4df3a373c8e7e9867f7e299/raw/7d829e72bb002e472575ab40675b48d69e736b36/Electric_production.csv'
df = pd.read_csv(url, parse_dates=['DATE'], index_col='DATE')

# Visualizar la serie temporal
plt.figure(figsize=(10,6))
plt.plot(df, label='Producción Eléctrica')
plt.title('Producción Eléctrica a lo largo del tiempo')
plt.xlabel('Fecha')
plt.ylabel('Millones de Kilowatthoras')
plt.legend()
plt.show()

#-----------------------------------------------------------------

from statsmodels.tsa.stattools import adfuller

result = adfuller(df['IPG2211A2N'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

#-------------------------------------------------------------

df_diff = df.diff().dropna()

plt.figure(figsize=(10,6))
plt.plot(df_diff, label='Producción Eléctrica Diferenciada')
plt.title('Producción Eléctrica Diferenciada')
plt.xlabel('Fecha')
plt.ylabel('Diferencia en Millones de Kilowatthoras')
plt.legend()
plt.show()

#---------------------------------------------------------
from pmdarima import auto_arima

model = auto_arima(df, seasonal=False, trace=True)
model.summary()




