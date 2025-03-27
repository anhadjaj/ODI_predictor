import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Load Data
files = os.listdir('/Users/anhadsinghjaj/Desktop/odis_male_csv2')

# IMPORTING THE MATCH WINNER DICTIONARY
match_id = {}
for file in filter(lambda file: file.endswith('info.csv'), files):
    df = pd.read_csv('/Users/anhadsinghjaj/Desktop/odis_male_csv2/' + file, on_bad_lines='warn')
    if len(df[df['version'] == 'winner']) == 0:
        continue
    else:
        match_id[int(file.split('_')[0])] = df[df['version'] == 'winner'].iloc[:, -1].values.tolist()[0]

# IMPORTING THE MATCH DATA
data = []
for file in filter(lambda file: not file.endswith('info.csv') and file.endswith('.csv'), files):
    df = pd.read_csv('/Users/anhadsinghjaj/Desktop/odis_male_csv2/' + file)
    data.append(df)

data = pd.concat(data).reset_index()
data['total_runs'] = data['runs_off_bat'] + data['extras']

first_innings_d = data[data['innings'] == 1]
data = pd.merge(data, first_innings_d.groupby(by='match_id').sum()['total_runs'], on='match_id')

chasing_data = data[data['innings'] == 2].copy()
chasing_data['curr_runs'] = chasing_data.groupby(['match_id'])['total_runs_x'].cumsum()
chasing_data['is_wicket'] = ~chasing_data['wicket_type'].isna()
chasing_data['curr_wickets'] = chasing_data.groupby(['match_id'])['is_wicket'].cumsum()

chasing_data['winner'] = chasing_data['match_id'].map(lambda x: match_id.get(x))
chasing_data['is_chase_suc'] = (chasing_data['winner'] == chasing_data['batting_team']).astype(int)

# Convert ball-by-ball data to overs
chasing_data['overs'] = chasing_data.groupby('match_id').cumcount() // 6

# Selecting relevant features
features = ['total_runs_y', 'curr_runs', 'curr_wickets', 'overs']
X = chasing_data[features].values
y = chasing_data['is_chase_suc'].values 

# Scaling Features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Neural Network Model (Optimized for Higher Accuracy)
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Neural Network Accuracy: {accuracy:.2f}")

# Save model
model.save("odi_chase_predictor_new.h5")
print("Optimized Model saved successfully!")