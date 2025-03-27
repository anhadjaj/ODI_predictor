# ODI_predictor
A neural network based model to predict the results of One-Day International Cricket Matches.
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
2) Based on these inputs, the model will predict whether the chasing team will win or loose, along with the winning probability of the chasing team.
