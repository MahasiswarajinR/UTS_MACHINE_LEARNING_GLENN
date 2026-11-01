#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# # Memuat Dataset (Data Understanding)

# In[3]:


file_path = "data-luas-panen-dan-produksi-sayuran-dan-buah-buahan-semusim-di-provinsi-dki-jakarta-tahun-2023-komponen-data.xls"
df = pd.read_html(file_path)[0]

display(df.head())
display(df.info())


# # Data Preparation (Pengecekan Missing Value & Outlier)

# In[4]:


print("--- Pengecekan Missing Value ---")
print(df.isnull().sum())

print("\n--- Deskripsi Statistik ---")
display(df.describe())


# # Penentuan Variabel (X dan y)

# In[5]:


# X = Variabel Independen (Luas Lahan)
# y = Variabel Dependen (Produksi)
X = df[['luas']]
y = df['produksi']


# # Pembagian Data (Train & Test Split)

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Jumlah data latih: {len(X_train)}")
print(f"Jumlah data uji: {len(X_test)}")


# # Pemodelan Regresi Linier

# In[7]:


model = LinearRegression()
model.fit(X_train, y_train)

b = model.coef_[0]
a = model.intercept_
print(f"Persamaan Regresi Linier: y = {a:.2f} + {b:.2f}X")
print(f"Konstanta (a) : {a:.2f}")
print(f"Koefisien (b) : {b:.2f}")


# # Prediksi & Evaluasi Model

# In[8]:


y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("--- Evaluasi Performa Model ---")
print(f"R-squared (R2)           : {r2:.4f}")
print(f"Mean Squared Error (MSE) : {mse:.2f}")
print(f"Root Mean Squared Error  : {rmse:.2f}")


# # Visualisasi Hasil Regresi

# In[9]:


plt.figure(figsize=(10, 6))

plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Data Latih (Aktual)')
plt.scatter(X_test, y_test, color='green', alpha=0.8, label='Data Uji (Aktual)')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Garis Regresi (Prediksi)')

plt.title('Regresi Linier: Luas Lahan vs Produksi (DKI Jakarta 2023)', fontsize=14)
plt.xlabel('Luas Lahan (Hektar)', fontsize=12)
plt.ylabel('Produksi (Kuintal)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[ ]:




