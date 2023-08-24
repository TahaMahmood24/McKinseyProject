# McKinsey_Project

During Covid, McKinsey organized a data analytical competition among analysts for a grand prize of $25,000 for those who secured the top position. The position is determined by using the train data to train your ML Model and then utilising that model to crunch results for the test data. The one with the closest submission to the test data will be declared the winner.

## Objectives

In this project, the main objective was to predict the galaxies' energy output and then compare it with the actual energy output data.
Based on the result comparison, a score is assigned to your code with the relevant position in the leaderboard among other competitors.

## Methodology

The solutions are evaluated on two criteria: predicted future Index values and allocated energy from a newly discovered star.

 - Index predictions are evaluated using the RMSE metric
Energy allocation is also evaluated using the RMSE metric and has a set of known factors that need to be taken into account
Every galaxy has a certain limited potential for improvement in the index described by the following function:

- Potential for increase in the Index = -np.log(Index+0.01)+3

Likely index increase dependent on the potential for improvement and on extra energy availability is described by the following function:

- Likely increase in the Index = extra energy * Potential for increase in the Index **2 / 1000

## Constraints

- In total, there are 50,000 zillion DSML available for allocation.

- No galaxy should be allocated more than 100 zillion DSML or less than 0 zillion DSML

- Galaxies with a low existence expectancy index below 0.7 should be allocated at least 10% of the total energy available



## Submit Format 

| Variable             | Description                                                                |
| ----------------- | ------------------------------------------------------------------ |
| Index | Unique index from the test dataset in the ascending order |
| pred | Prediction for the index of interest |
| pred_opt | Optimal energy allocation |


## Result

I ran a series of codes and models to determine the best model output. Unfortunately, due to sheer competition & lack of experience, my best model, "XGBoost", secured the 525th position among the competitors. However, this marked the beginning for me in this field. This was the starting point of my journey in the field of Data Science, and this project helped me drive that passion within me.

