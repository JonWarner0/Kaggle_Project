# Kaggle Competition

https://www.kaggle.com/c/income-level-prediction-2021s/overview

Various algorithms used in the income-level-prediction competition where training and testing data are continous and categorical values.

Running each algorithm will follow the pattern:

    python3 <algorithm> <training> <testing> <flags>
    OUTPUT: results(algorithm).csv
    
Flags are dependent on the algorithm and checking the main of the .py files will describe the flags needed to effectively run the algorithm.

For algorithms needing strictly continuous values, use the ContMapping.py script to map the training and testing files into a continuous space. To run use:
    
    python3 ContMapping.py <training> <testing>
    OUTPUT: mappedTrainData.csv
            mappedTestData.csv

