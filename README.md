# 🏏 ODI Cricket Match Prediction

This project predicts match dynamics in One Day Internationals (ODIs), focusing on:

- **First Innings Final Score Prediction** using an LSTM model.
- **Second Innings Chase Success Prediction** using a neural network.

The system processes ball-by-ball data and predicts match progression with high accuracy, enabling real-time strategic insights for both innings.

---

## 📂 Dataset

**Source:** Raw CSVs from [Cricsheet](https://cricsheet.org) (structured by match)

Includes detailed ball-by-ball records with:

- `match_id`
- `innings`
- `runs_off_bat`
- `extras`
- `wicket_type`, etc.

Metadata extracted from `*_info.csv` files includes the **match winner**.

---

## 🧠 First Innings: Final Score Prediction

> Predicts the final total runs a team is likely to score based on current match progress.

### ✅ Features Used

- `curr_runs`: Cumulative runs at any ball
- `curr_wickets`: Cumulative wickets fallen
- `overs`: Over number (derived from ball count)

### ⚙️ Model: LSTM

- 2 LSTM layers (64, 32 units) with dropout and batch normalization
- Dense output layer for regression

### Training Details

- **Input Shape:** (samples, 1, 3)
- **Scaler:** `MinMaxScaler()`
- **Loss:** Mean Squared Error (MSE)
- **Validation:** 80/20 train-test split

### 📈 Evaluation

- MSE: ~[value]
- MAE: ~[value]

Inputs from users are dynamically scaled and reshaped for live predictions.

---

## 🏃‍♂️ Second Innings: Chase Success Prediction

> Predicts the probability of the batting second team winning using cumulative metrics and the first innings target.

### ✅ Features Used

- `total_runs_target`: Target set by team batting first
- `curr_runs`: Runs scored so far
- `curr_wickets`: Wickets lost so far
- `overs`: Over count

### ⚙️ Model: Dense Neural Network

- Architecture: Dense(32) → Dropout → Dense(16) → Dropout → Dense(8) → Sigmoid
- Binary classification for chase success
- Optimized for balanced accuracy and generalization

### 📦 Labels

- `is_chase_suc`: 1 if second innings team won, else 0

### 📈 Evaluation

- Accuracy: ~ 82% (chasing model), MAE- 30 (target prediction model)

Live predictions show dynamic win probability as the second innings progresses.

---

## 🔄 Preprocessing & Feature Engineering

- **Match Winner Merge:** Mapped using `_info.csv` files
- **Cumulative Stats:** `groupby().cumsum()` for runs and wickets
- **Overs:** Computed using `cumcount() // 6`
- **Scaling:** `MinMaxScaler()` used and saved via `pickle`

---

## 📁 Files Generated

- `odi_target_predictor_lstm.h5` – LSTM model for innings 1
- `odi_chase_predictor_new.h5` – Neural network model for innings 2
- `scaler_first.pkl` – Scaler for first innings model
- `scaler_second.pkl` – Scaler for second innings model

---

## 🔮 Future Extensions

- Integrate with live APIs (e.g., Cricbuzz, ESPNcricinfo)
- Include player-level impact metrics (strike rate, economy)
- Use venue/pitch/toss conditions as features
- Add uncertainty modeling with probabilistic/ensemble methods
- Optimize field placement via reinforcement learning

---
