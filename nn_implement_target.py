import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load data
files = os.listdir('/Users/anhadsinghjaj/Desktop/odis_male_csv2')

data = []
for file in filter(lambda file: file.endswith('.csv') and not file.endswith('info.csv'), files):
    df = pd.read_csv(f'/Users/anhadsinghjaj/Desktop/odis_male_csv2/{file}')
    data.append(df)

data = pd.concat(data).reset_index(drop=True)
data['total_runs'] = data['runs_off_bat'].fillna(0) + data['extras'].fillna(0)

# First innings data
first_innings = data[data['innings'] == 1].copy()
first_innings['curr_runs'] = first_innings.groupby('match_id')['total_runs'].cumsum()
first_innings['is_wicket'] = first_innings['wicket_type'].notna().astype(int)
first_innings['curr_wickets'] = first_innings.groupby('match_id')['is_wicket'].cumsum()
first_innings['overs'] = first_innings.groupby('match_id').cumcount() // 6
first_innings = first_innings.merge(first_innings.groupby('match_id')['total_runs'].sum(), on='match_id', suffixes=('', '_final'))

# Select Only 3 Features
features = ['curr_runs', 'curr_wickets', 'overs']
X1 = first_innings[features].fillna(0).values
y1 = first_innings['total_runs_final'].values

# Scaling
scaler1 = MinMaxScaler()
X1_scaled = scaler1.fit_transform(X1)

# Reshape for LSTM
X1_lstm = X1_scaled.reshape(X1_scaled.shape[0], 1, X1_scaled.shape[1])

# Train-test split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1_lstm, y1, test_size=0.2, random_state=42)

# Updated LSTM Model for 3 Features
model1 = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, X1_train.shape[2]), activation='tanh'),
    Dropout(0.3),
    
    LSTM(32, return_sequences=False, activation='tanh'),
    BatchNormalization(),
    
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

# Compile Model
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])

# Callbacks to Improve Training
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# Train Model
model1.fit(X1_train, y1_train, epochs=100, batch_size=64, validation_data=(X1_test, y1_test), callbacks=callbacks, verbose=1)

# Evaluate Model
mse, mae = model1.evaluate(X1_test, y1_test, verbose=0)
print(f"\U0001F4C9 Improved LSTM Model - MSE: {mse:.2f}, MAE: {mae:.2f}")

# Save Model
model1.save("odi_target_predictor_lstm.h5")