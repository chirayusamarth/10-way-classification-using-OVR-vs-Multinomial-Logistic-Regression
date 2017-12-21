# 10-way-classification-using-OVR-vs-Multinomial-Logistic-Regression

# Dataset

We will use mnist subset (images of handwritten digits from 0 to 9). As before, the dataset is stored in a JSON-formated
file mnist subset.json. You can access its training, validation, and test splits using the keys
‘train’, ‘valid’, and ‘test’, respectively. For example, suppose we load mnist subset.json to the
variable x. Then, x\['train'\] refers to the training set of mnist subset. This set is a list with two
elements: x\['train'\] \[0\] containing the features of size N (samples) ×D (dimension of features), and
x\['train'\]\[1\] containing the corresponding labels of size N.


# Cautions

Please do not import packages that are not listed in the provided code. Follow the instructions
in each section strictly to code up your solutions. Do not change the output format. Do
not modify the code unless we instruct you to do so.

# Multi-class Classification
You will modify 4 python functions in logistic prog.py. First, you will implement two functions
that train and test a one-versus-rest multi-class classification model. Second, you will implement
two functions that train and test a multinomial logistic regression model. Finally, you will run script q43.sh that trains and tests the two models using your implemented functions, and store your results to logistic res.json. TAs wrote functions to load, relabel, and sample the data for you, so you are not responsible for doing it. TAs have also pre-defined the hyperparameters and initializations in the template code. Moreover, you will use the AVERAGE of gradients from all training samples to update the parameters.

**One-versus-rest:**
Implement the code to solve the multi-class classification task with the one-versus-rest
strategy. That is, train 10 binary logistic regression models following the setting provided in class:
for each class Ck, k = 1,. . ., 10, we create a binary classification problem as follows:
- Re-label training samples with label Ck as positive (namely 1)
- Re-label other samples as negative (namely 0)


**Multinomial logistic regression:**
Implement the multinomial logistic regression, training a 10-way classifier (with the softmax
function) on mnist subset dataset. 
