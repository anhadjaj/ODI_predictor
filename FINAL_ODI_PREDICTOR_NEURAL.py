import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow.keras.losses import MeanSquaredError

# Load data
files = os.listdir('/Users/anhadsinghjaj/Desktop/odis_male_csv2')

# Import match winner dictionary
match_id = {}
for file in filter(lambda file: file.endswith('info.csv'), files):
    df = pd.read_csv(f'/Users/anhadsinghjaj/Desktop/odis_male_csv2/{file}', on_bad_lines='warn')
    if not df[df['version'] == 'winner'].empty:
        match_id[int(file.split('_')[0])] = df[df['version'] == 'winner'].iloc[:, -1].values[0]

# Import match data
data = []
for file in filter(lambda file: file.endswith('.csv') and not file.endswith('info.csv'), files):
    df = pd.read_csv(f'/Users/anhadsinghjaj/Desktop/odis_male_csv2/{file}')
    data.append(df)

data = pd.concat(data).reset_index(drop=True)
data['total_runs'] = data['runs_off_bat'] + data['extras']

# Get user input for innings selection
inn = int(input("Enter the innings going on right now (1 or 2): "))

# ----------------------------------------
# **First Innings: Predict Final Score**
# ----------------------------------------
    # ----------------------------------------
# **First Innings: Predict Final Score**
# ----------------------------------------
if inn == 1:
    first_innings = data[data['innings'] == 1].copy()
    first_innings['curr_runs'] = first_innings.groupby('match_id')['total_runs'].cumsum()
    first_innings['is_wicket'] = ~first_innings['wicket_type'].isna()
    first_innings['curr_wickets'] = first_innings.groupby('match_id')['is_wicket'].cumsum()
    first_innings['overs'] = first_innings.groupby('match_id').cumcount() // 6
    first_innings = first_innings.merge(first_innings.groupby('match_id')['total_runs'].sum(), on='match_id', suffixes=('', '_final'))

    X1 = first_innings[['curr_runs', 'curr_wickets', 'overs']].values
    y1 = first_innings['total_runs_final'].values

    # Load the pre-trained LSTM model
    model1 = load_model("odi_target_predictor_lstm.h5", custom_objects={"mse": MeanSquaredError()})

    # Load and fit the scaler
    scaler1 = MinMaxScaler()
    X1_scaled = scaler1.fit_transform(X1)

    # Reshape for LSTM: (samples, time_steps=1, features=3)
    X1_scaled = X1_scaled.reshape((X1_scaled.shape[0], 1, X1_scaled.shape[1]))

    # Evaluate the model
    mse = model1.evaluate(X1_scaled, y1, verbose=0)
    print(f"📊 Model MSE: {mse}")

    while True:
        try:
            curr_runs = int(input("Enter the current runs: "))
            curr_wickets = int(input("Enter the current wickets lost: "))
            overs = float(input("Enter the over going on right now: "))

            # Scale input dynamically
            user_input = np.array([[curr_runs, curr_wickets, overs]])
            user_input_scaled = scaler1.transform(user_input)

            # Reshape for LSTM: (samples=1, time_steps=1, features=3)
            user_input_scaled = user_input_scaled.reshape((1, 1, 3))

            # Predict final score
            prediction = model1.predict(user_input_scaled)[0][0]
            print(f"🏏 **Predicted Final Score:** {int(prediction)}")

            if input("Press 'x' to exit or Enter to continue: ").lower() == 'x':
                break

        except ValueError:
            print("❌ Invalid input. Please enter numerical values.")


# ----------------------------------------
# **Second Innings: Predict Chase Success**
# ----------------------------------------
elif inn == 2:
    first_innings_d = data[data['innings'] == 1]
    data = pd.merge(data, first_innings_d.groupby('match_id')['total_runs'].sum(), on='match_id', suffixes=("", "_target"))

    chasing_data = data[data['innings'] == 2].copy()
    chasing_data['curr_runs'] = chasing_data.groupby('match_id')['total_runs'].cumsum()
    chasing_data['is_wicket'] = ~chasing_data['wicket_type'].isna()
    chasing_data['curr_wickets'] = chasing_data.groupby('match_id')['is_wicket'].cumsum()
    chasing_data['overs'] = chasing_data.groupby('match_id').cumcount() // 6

    chasing_data['winner'] = chasing_data['match_id'].map(match_id.get)
    chasing_data['is_chase_suc'] = (chasing_data['winner'] == chasing_data['batting_team']).astype(int)

    # Selecting relevant features
    features = ['total_runs_target', 'curr_runs', 'curr_wickets', 'overs']
    X = chasing_data[features].values
    y = chasing_data['is_chase_suc'].values  

    # Compute MinMaxScaler from dataset itself
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Load Neural Network Model
    model = load_model("odi_chase_predictor_new.h5")

    # Evaluate model
    y_pred = (model.predict(X_scaled) > 0.5).astype(int)
    accuracy = accuracy_score(y, y_pred)
    print(f" Neural Network Accuracy for 2nd Innings Chase Prediction: {accuracy:.2f}")

    # User input loop for predictions
    while True:
        try:
            target = int(input("Enter the target score: "))
            curr_runs = int(input("Enter the current runs: "))
            curr_wickets = int(input("Enter the current wickets lost: "))
            overs = float(input("Enter the over going on right now: "))

            # Scale input dynamically
            user_input = np.array([[target, curr_runs, curr_wickets, overs]])
            user_input_scaled = scaler.transform(user_input)

            # Predict win probability
            win_prob = model.predict(user_input_scaled)[0][0]
            chase_success = int(win_prob > 0.5)

            print("\n🔹 **Chase Prediction Results** 🔹")
            print(f"🔵 Probability of Successful Chase: {win_prob:.2%}")  # Format as percentage
            print(f"⚡ Prediction: {'Chase should be successful!' if chase_success else 'Chase might fail!'}\n")

            if input("Press 'x' to exit or Enter to continue: ").lower() == 'x':
                break

        except ValueError:
            print("Invalid input. Please enter numerical values.")

else:
    print("Invalid innings number! Please enter 1 or 2.")
