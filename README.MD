# Zindi Tourist classification competition (solution at position 10)
Details about the competion is available at https://zindi.africa/competitions/ai4d-lab-tanzania-tourism-classification-challenge

The final solution was an ensemble of 4 models:
- Gradient boosting using catboost (boosting.py)
- A multilayer perceptron (mlp.py)
- A multi-task mlp (semi-supervised.py). The first task is the classification task for expedicture while the second task predicts a mask over the inputs. The mask determines which inputs are augmented using permutation (replacing input values with random values from the same column)
- Blending (Blending_MLP.py): The predictions of the previous 3 models are added to the inputs


The file stacking.py generates the final predictions by averaging the prediction of the above mentioned models.

## Running the scripts
1) Modify the file setting.py to specify the following variables:
   1) data_dir: path containing the data. ie "Train.csv" and "Test.csv"
   2) results_dir: path where intermediate outputs of the four models will be stored.
   3) log_dir: directory for storing logs (required by ray-tune)

2) Run boosting.py, mlp.py, and semi_supervised.py. Order not important
3) Run Blending_MLP.py
4) Run Stacking.py
