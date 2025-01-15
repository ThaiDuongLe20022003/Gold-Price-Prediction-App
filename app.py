# Import Libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Input, Dense, Dropout, Attention, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Reading Dataset
df = pd.read_csv('gold.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by = 'Date', ascending = True, inplace = True)
df.reset_index(drop = True, inplace = True)
df['Price'] = df['Price'].replace({',': ''}, regex = True).astype(float)

# Streamlit title
st.title('Gold Price Prediction Based On Historical Data')
st.write("In this project, we implement two models: Long-Short Term Memory with Attention Mechanism and Gated Reccurent Unit.")
st.write("\nHere's the data from 2001 to now (January 12th, 2025).")

# Visualizing Gold Price in 21st Century
plt.figure(figsize = (15, 6), dpi = 150)
plt.rcParams['axes.facecolor'] = 'lightgray'
plt.rc('axes', edgecolor = 'white')

plt.plot(df.Date, df.Price, color = 'blue', lw = 2)

plt.title('Gold Price in 21st Century', fontsize = 15)
plt.xlabel('Year', fontsize = 12)
plt.ylabel('Price', fontsize = 12)

plt.grid(color = 'white')

plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()

st.pyplot(plt)

# Sidebar for model selection
model_option = st.sidebar.selectbox(
    'Choose a model to predict gold prices',
    ['LSTM+Attention', 'GRU', 'Two Models']
)

# Filter data for the years between 2020 and 2025
test_data = df[df['Date'].dt.year.between(2020, 2025)]
test_size = test_data.shape[0]
train_data = df.Price[:- test_size]

st.write("\nWe cannot train on future data in time series data, so we consider the data from 2020 (the first year of COVID-19 pandemic) to now for testing and everything else for training.")

# Gold Price Training and Test Sets Plot
plt.figure(figsize = (15, 6), dpi = 150)
plt.rcParams['axes.facecolor'] = 'lightgray'
plt.rc('axes',edgecolor = 'white')

plt.plot(df.Date[:- test_size], df.Price[:- test_size], color = 'blue', lw = 2)
plt.plot(df.Date[- test_size:], df.Price[- test_size:], color = 'red', lw = 2)

plt.title('Gold Price Training and Test Sets', fontsize = 15)
plt.xlabel('Year', fontsize = 12)
plt.ylabel('Price', fontsize = 12)

plt.legend(
    ['Training set', 'Test set'],
    loc='upper left',
    prop={'size': 15},
    facecolor = 'white',
    edgecolor = 'black'
)

plt.grid(color = 'white')

plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()

st.pyplot(plt)


# Data Scaling
scaler = MinMaxScaler()
scaler.fit(df.Price.values.reshape(-1,1))

# Define Window Sizes
window_size_lstm = 4
window_size_gru = 3

# Prepare data for LSTM+Attention
train_data_lstm = scaler.transform(df.Price[:- test_size].values.reshape(-1, 1))
X_train_lstm, y_train_lstm = [], []
for i in range(window_size_lstm, len(train_data_lstm)):
    X_train_lstm.append(train_data_lstm[i - window_size_lstm : i, 0])
    y_train_lstm.append(train_data_lstm[i, 0])

test_data_lstm = scaler.transform(df.Price[- test_size - window_size_lstm:].values.reshape(-1, 1))
X_test_lstm, y_test_lstm = [], []
for i in range(window_size_lstm, len(test_data_lstm)):
    X_test_lstm.append(test_data_lstm[i - window_size_lstm : i, 0])
    y_test_lstm.append(test_data_lstm[i, 0])

X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm).reshape(-1, 1)
X_test_lstm, y_test_lstm = np.array(X_test_lstm), np.array(y_test_lstm).reshape(-1, 1)
X_train_lstm = X_train_lstm.reshape(-1, window_size_lstm, 1)
X_test_lstm = X_test_lstm.reshape(-1, window_size_lstm, 1)

# Prepare data for GRU
train_data_gru = scaler.transform(df.Price[:- test_size].values.reshape(-1, 1))
X_train_gru, y_train_gru = [], []
for i in range(window_size_gru, len(train_data_gru)):
    X_train_gru.append(train_data_gru[i - window_size_gru : i, 0])
    y_train_gru.append(train_data_gru[i, 0])

test_data_gru = scaler.transform(df.Price[- test_size - window_size_gru:].values.reshape(-1, 1))
X_test_gru, y_test_gru = [], []
for i in range(window_size_gru, len(test_data_gru)):
    X_test_gru.append(test_data_gru[i - window_size_gru : i, 0])
    y_test_gru.append(test_data_gru[i, 0])

X_train_gru, y_train_gru = np.array(X_train_gru), np.array(y_train_gru).reshape(-1, 1)
X_test_gru, y_test_gru = np.array(X_test_gru), np.array(y_test_gru).reshape(-1, 1)
X_train_gru = X_train_gru.reshape(- 1, window_size_gru, 1)
X_test_gru = X_test_gru.reshape(- 1, window_size_gru, 1)

# Define Models
def define_lstm_attention_model():
    input_seq = Input(shape = (window_size_lstm, 1))
    lstm_out = LSTM(64, return_sequences = True)(input_seq)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = LSTM(64, return_sequences = True)(lstm_out)

    attention = Attention()([lstm_out, lstm_out])
    attention = Flatten()(attention)
    attention = Dense(32, activation = 'relu')(attention)
    output = Dense(1)(attention)

    model = Model(inputs = input_seq, outputs = output)
    model.compile(loss = 'mean_squared_error', optimizer = 'Nadam')
    model.summary()

    return model

def define_gru_model():
    input_seq = Input(shape = (window_size_gru, 1))
    gru_out = GRU(64, return_sequences = True)(input_seq)
    gru_out = Dropout(0.2)(gru_out)
    gru_out = GRU(64, return_sequences = False)(gru_out)
    dense_out = Dense(32, activation = 'relu')(gru_out)
    output = Dense(1)(dense_out)

    model = Model(inputs = input_seq, outputs = output)
    model.compile(loss = 'mean_squared_error', optimizer = 'Nadam')
    model.summary()

    return model

# Initialize models
lstm_attention_model = define_lstm_attention_model()
gru_model = define_gru_model()

# Callbacks
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True, verbose = 1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, min_lr = 1e-6, verbose = 1)

# Train LSTM+Attention
start_time_lstm = time.time()
lstm_history = lstm_attention_model.fit(
    X_train_lstm, y_train_lstm,
    epochs = 200,
    batch_size = 64,
    validation_split = 0.1,
    callbacks = [early_stopping, reduce_lr],
    verbose = 1
)
end_time_lstm = time.time()

# Train GRU
start_time_gru = time.time()
gru_history = gru_model.fit(
    X_train_gru, y_train_gru,
    epochs = 200,
    batch_size = 64,
    validation_split = 0.1,
    callbacks = [early_stopping, reduce_lr],
    verbose = 1
)
end_time_gru = time.time()

# Model Evaluation

# Evaluate LSTM+Attention Model
lstm_results = lstm_attention_model.evaluate(X_test_lstm, y_test_lstm, verbose=1)
lstm_pred = lstm_attention_model.predict(X_test_lstm)
lstm_r2 = r2_score(y_test_lstm, lstm_pred)

# Evaluate GRU Model
gru_results = gru_model.evaluate(X_test_gru, y_test_gru, verbose=1)
gru_pred = gru_model.predict(X_test_gru)
gru_r2 = r2_score(y_test_gru, gru_pred)

# Inverse Transform Predictions
y_test_true_lstm = scaler.inverse_transform(y_test_lstm)
lstm_pred_true = scaler.inverse_transform(lstm_pred)
y_test_true_gru = scaler.inverse_transform(y_test_gru)
gru_pred_true = scaler.inverse_transform(gru_pred)

# Calculate Metrics
def calculate_metrics(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mape, mae, rmse

lstm_mape, lstm_mae, lstm_rmse = calculate_metrics(y_test_true_lstm, lstm_pred_true)
gru_mape, gru_mae, gru_rmse = calculate_metrics(y_test_true_gru, gru_pred_true)

# Calculate Residuals
lstm_residuals = y_test_true_lstm - lstm_pred_true
gru_residuals = y_test_true_gru - gru_pred_true

st.write("\nRESULTS")


if model_option == 'LSTM+Attention':
    # Print Results
    st.write("\nLSTM+Attention")
    st.write(f"Loss: {lstm_results}")
    st.write(f"R² Score: {lstm_r2}")
    st.write(f"Mean Absolute Percentage Error: {lstm_mape} %")
    st.write(f"Test Accuracy: {100 - lstm_mape} %")
    st.write(f"Mean Absolute Error: {lstm_mae} USD")
    st.write(f"Root Mean Square Error: {lstm_rmse} USD")
    st.write(f"Training time: {end_time_lstm - start_time_lstm} seconds")

    # Visualizing Results
    plt.figure(figsize = (15, 6), dpi = 150)
    plt.rcParams['axes.facecolor'] = 'lightgray'
    plt.rc('axes', edgecolor = 'white')

    plt.plot(df['Date'].iloc[- test_size:], y_test_true_lstm, color = 'red', lw = 2, label = 'True Values')
    plt.plot(df['Date'].iloc[- test_size:], lstm_pred_true, color = 'yellow', lw = 2, label = 'LSTM+Attention')
    
    plt.title('Model Performance on Gold Price Prediction', fontsize = 15)
    plt.xlabel('Year', fontsize = 12)
    plt.ylabel('Price', fontsize = 12)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(loc = 'upper left', prop={'size': 15}, facecolor = 'white', edgecolor = 'black')
    plt.grid(color = 'white')
    st.pyplot(plt)

    # Scatterplot: Residuals vs True Values
    plt.figure(figsize=(8, 5))

    # LSTM+Attention Scatterplot
    plt.scatter(y_test_true_lstm, lstm_residuals, color='blue', alpha=0.6, edgecolor='k')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title("LSTM+Attention: Residuals vs True Values", fontsize=16)
    plt.xlabel("True Values", fontsize=14)
    plt.ylabel("Residuals", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    st.pyplot(plt)

    # Histogram of LSTM+Attention
    plt.figure(figsize = (8, 5))
    sns.histplot(lstm_residuals, kde = True, bins = 30)
    plt.title("LSTM+Attention: Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()

    st.pyplot(plt)

elif model_option == 'GRU':
    # Print Results
    st.write("\nGRU:")
    st.write(f"Loss: {gru_results}")
    st.write(f"R² Score: {gru_r2}")
    st.write(f"Mean Absolute Percentage Error: {gru_mape} %")
    st.write(f"Test Accuracy: {100 - gru_mape} %")
    st.write(f"Mean Absolute Error: {gru_mae} USD")
    st.write(f"Root Mean Square Error: {gru_rmse} USD")
    st.write(f"Training Time: {end_time_gru - start_time_gru} seconds")

    # Visualizing Results
    plt.figure(figsize = (15, 6), dpi = 150)
    plt.rcParams['axes.facecolor'] = 'lightgray'
    plt.rc('axes', edgecolor = 'white')

    plt.plot(df['Date'].iloc[- test_size:], y_test_true_lstm, color = 'red', lw = 2, label = 'True Values')
    plt.plot(df['Date'].iloc[- test_size:], gru_pred_true, color = 'blue', lw = 2, label = 'GRU')

    plt.title('Model Performance on Gold Price Prediction', fontsize = 15)
    plt.xlabel('Year', fontsize = 12)
    plt.ylabel('Price', fontsize = 12)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(loc = 'upper left', prop={'size': 15}, facecolor = 'white', edgecolor = 'black')
    plt.grid(color = 'white')
    st.pyplot(plt)

    # Scatterplot: Residuals vs True Values
    plt.figure(figsize = (8, 5))

    # GRU Scatterplot
    plt.scatter(y_test_true_gru, gru_residuals, color='blue', alpha=0.6, edgecolor='k')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title("GRU: Residuals vs True Values", fontsize=16)
    plt.xlabel("True Values", fontsize=14)
    plt.ylabel("Residuals", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    st.pyplot(plt)

    # Histogram of GRU
    plt.figure(figsize = (8, 5))
    sns.histplot(gru_residuals, kde = True, bins = 30)
    plt.title("GRU: Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()

    st.pyplot(plt)

else:
    # Print Results
    st.write("\nLSTM+Attention")
    st.write(f"Loss: {lstm_results}")
    st.write(f"R² Score: {lstm_r2}")
    st.write(f"Mean Absolute Percentage Error: {lstm_mape} %")
    st.write(f"Test Accuracy: {100 - lstm_mape} %")
    st.write(f"Mean Absolute Error: {lstm_mae} USD")
    st.write(f"Root Mean Square Error: {lstm_rmse} USD")
    st.write(f"Training time: {end_time_lstm - start_time_lstm} seconds")

    st.write("\nGRU:")
    st.write(f"Loss: {gru_results}")
    st.write(f"R² Score: {gru_r2}")
    st.write(f"Mean Absolute Percentage Error: {gru_mape} %")
    st.write(f"Test Accuracy: {100 - gru_mape} %")
    st.write(f"Mean Absolute Error: {gru_mae} USD")
    st.write(f"Root Mean Square Error: {gru_rmse} USD")
    st.write(f"Training Time: {end_time_gru - start_time_gru} seconds")

    # Visualizing Results
    plt.figure(figsize = (15, 6), dpi = 150)
    plt.rcParams['axes.facecolor'] = 'lightgray'
    plt.rc('axes', edgecolor = 'white')

    plt.plot(df['Date'].iloc[- test_size:], y_test_true_lstm, color = 'red', lw = 2, label = 'True Values')
    plt.plot(df['Date'].iloc[- test_size:], lstm_pred_true, color = 'yellow', lw = 2, label = 'LSTM+Attention')
    plt.plot(df['Date'].iloc[- test_size:], gru_pred_true, color = 'blue', lw = 2, label = 'GRU')

    plt.title('Model Performance on Gold Price Prediction', fontsize = 15)
    plt.xlabel('Year', fontsize = 12)
    plt.ylabel('Price', fontsize = 12)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(loc = 'upper left', prop={'size': 15}, facecolor = 'white', edgecolor = 'black')
    plt.grid(color = 'white')
    st.pyplot(plt)

    # Scatterplot: Residuals vs True Values
    plt.figure(figsize = (14, 6), dpi = 150)

    # LSTM+Attention Scatterplot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_true_lstm, lstm_residuals, color = 'blue', alpha = 0.6)
    plt.axhline(0, color = 'red', linestyle ='--', linewidth = 1)
    plt.title("LSTM+Attention: Residuals vs True Values", fontsize = 14)
    plt.xlabel("True Values", fontsize = 12)
    plt.ylabel("Residuals", fontsize = 12)
    plt.grid(True)

    # GRU Scatterplot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_true_gru, gru_residuals, color = 'green', alpha = 0.6)
    plt.axhline(0, color = 'red', linestyle = '--', linewidth = 1)
    plt.title("GRU: Residuals vs True Values", fontsize = 14)
    plt.xlabel("True Values", fontsize = 12)
    plt.ylabel("Residuals", fontsize = 12)
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(plt)

    # Histogram of LSTM+Attention
    plt.figure(figsize = (8, 5))
    sns.histplot(lstm_residuals, kde = True, bins = 30)
    plt.title("LSTM+Attention: Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()

    st.pyplot(plt)

    # Histogram of GRU
    plt.figure(figsize = (8, 5))
    sns.histplot(gru_residuals, kde = True, bins = 30)
    plt.title("GRU: Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()

    st.pyplot(plt)