from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


st.title('Stock Trend Prediction')

user_input = st.text_input("Enter stock ticker", 'AAPL')
df = data.get_data_tiingo(
    user_input, api_key='9099441f551d690bf3e68849bc15feab91dff9c4')

# Describing tha data
st.subheader('Data')
st.write(df)

# visualize
st.subheader('Closing price Vs Time chart')
fig = plt.figure(figsize=(12, 6))
cl = df.close
plt.plot(list(cl))
st.pyplot(fig)


st.subheader('Closing price Vs Time chart with 100MA')
ma100 = df.close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
cl = df.close
plt.plot(list(ma100), 'r')
plt.plot(list(cl), 'b')
st.pyplot(fig)


st.subheader('Closing price Vs Time chart with 100MA & 200MA')
ma100 = df.close.rolling(100).mean()
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
cl = df.close
plt.plot(list(ma100), 'r')
plt.plot(list(ma200), 'g')
plt.plot(list(cl), 'b')
st.pyplot(fig)


df = df.reset_index()


# splitting of data
data_train = pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['close'][int(len(df)*0.70):int(len(df))])

sc = MinMaxScaler(feature_range=(0, 1))

data_train_arr = sc.fit_transform(data_train)


# load my model
model = load_model('keras_model.h5')

# testing part

past_100_days = data_train.tail(100)
final_df = past_100_days.append(data_test, ignore_index=True)
input_data = sc.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test)

scaler = sc.scale_

scale_factor = 1/scaler[0]
y_pred = y_pred*scale_factor
y_test = y_test*scale_factor


# final graph

st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(list(y_test), 'b', label='Original price')
plt.plot(list(y_pred), 'r', label='predicted price')
plt.xlabel('Price')
plt.ylabel('Time')
plt.legend()
st.pyplot(fig2)
