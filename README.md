# ODI_predictor
A neural network-based model to predict the results of One-Day International Cricket Matches.
----> 1st INNINGS 
1) We take 3 variables as input.
   -> 'Current runs' scored by the team batting first.
   -> 'Current wickets' lost by the team batting first.
   -> 'Current overs' elapsed.
2) Based on these inputs, the model will predict the final probable score that the team batting first will set.

----> 2nd INNINGS
1) We take 4 variables as input.
   -> The 'target' to chase down.
   -> The 'current runs' of the team chasing the target.
   -> The 'current wickets' lost by the team chasing the target.
   -> The 'current overs' elapsed.
2) Based on these inputs, the model will predict whether the chasing team will win or lose, along with the winning probability of the chasing team.


ABOUT THE UPLOADED FILES:
1) FINAL_ODI_PREDICTOR_NEURAL.py: The final python file where the inputs are taken and outputs are generated according to the 2 trained files, odi_chase_predictor_new.h5 and odi_target_predictor_lstm.h5.
2) nn_implem.py: The python file where the 2nd innings data was trained using neural Networks.
3) nn_implement_target.py: The python file where the 1st innings data was trained using an LSTM-based neural network approach.
4) odi_chase_predictor_new.h5: The file containing the trained data for the 2nd innings.
5) odi_target_predictor_lstm.h5: The file containing the trained data for the 1st innnings.
