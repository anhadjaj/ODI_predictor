# 🏏 ODI Cricket Match Prediction
This project predicts match dynamics in One Day Internationals (ODIs), focusing on:
   -First Innings Final Score Prediction using an LSTM model.
   -Second Innings Chase Success Prediction using a neural network.
The system processes ball-by-ball data and predicts match progression with high accuracy, enabling real-time strategic insights for both innings.

> 📂 Dataset
Source: Raw CSVs from cricsheet.org (structured by match)
Includes detailed ball-by-ball records with:
   -match_id, innings, runs_off_bat, extras, wicket_type, etc.
Metadata extracted from *_info.csv files includes the match winner.

🧠 First Innings: Final Score Prediction
Predicts the final total runs a team is likely to score based on current match progress.

✅ Features Used:
curr_runs: Cumulative runs at any ball

curr_wickets: Cumulative wickets fallen

overs: Over number (derived from ball count)

⚙️ Model: LSTM
2 LSTM layers (64, 32 units) with dropout and batch normalization

Dense output layer for regression

Training Details:
Input Shape: (samples, 1, 3)

Scaler: MinMaxScaler()

Loss: Mean Squared Error (MSE)

MAE: Printed post-evaluation

Validation: 80/20 train-test split

📈 Evaluation:
Final model performance (example):

MSE: ~[value]

MAE: ~[value]

Input from users dynamically scaled and reshaped for live predictions.

🏃‍♂️ Second Innings: Chase Success Prediction
Predicts the probability of the batting second team winning using cumulative metrics and target score.

✅ Features Used:
total_runs_target: Target set by team batting first

curr_runs: Runs scored so far

curr_wickets: Wickets lost so far

overs: Over count

⚙️ Model: Dense Neural Network
Architecture: Dense(32) → Dropout → Dense(16) → Dropout → Dense(8) → Sigmoid output

Binary classification for chase success

Optimized for balanced accuracy and generalization

📦 Labels:
is_chase_suc: 1 if second innings team won, else 0

📈 Evaluation:
Accuracy: ~[example: 0.89–0.92 range]

Real-time predictions show live win probability for both teams based on inputs.

🔄 Preprocessing & Engineering
Merging Info: Match winner mapping from _info.csv files

Cumulative Metrics: Runs & wickets using groupby().cumsum()

Overs Calculation: cumcount() // 6

Model Input Scaling:

Scaler saved using pickle for future use

scaler_first.pkl and scaler_second.pkl respectively

📁 Files Generated:
odi_target_predictor_lstm.h5 – LSTM model for innings 1

odi_chase_predictor_new.h5 – Neural network for innings 2

scaler_first.pkl – First innings MinMaxScaler

scaler_second.pkl – Second innings MinMaxScaler

🔮 Future Extensions
Integrate with real-time live match APIs (e.g., Cricbuzz, ESPNcricinfo)

Include player-level impact (strike rate, economy)

Factor in venue conditions, pitch reports, and toss

Model uncertainty using probabilistic outputs or ensembles

Field placement impact analysis using reinforcement learning
