import numpy as np
import perceptron
import csv
import random as rd

"""
read data: 
age - age in years
sex - (1 = male; 0 = female)
cp - chest pain type
trestbps - resting blood pressure (in mm Hg on admission to the hospital)
chol - serum cholestoral in mg/dl
fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg - resting electrocardiographic results
thalach - maximum heart rate achieved
exang - exercise induced angina (1 = yes; 0 = no)
oldpeak - ST depression induced by exercise relative to rest
slope - the slope of the peak exercise ST segment
ca - number of major vessels (0-3) colored by flourosopy
thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
target - 1 or 0
"""
with open('heart_new.csv', newline='') as f:
    data_list = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))

#shuffling data set - it is ordered by target
#np.random.seed(753818022)
#np.random.shuffle(data_list)

#split target from data
target = []
for i in range(len(data_list)):
    target = np.append(target, data_list[i].pop())

data = np.array(data_list)

#split target and data into train, validation and test sets

# dostosowywać do potrzeb i datasetu
train_X = data[:800]
train_y = target[:800]
print('Shape of training set: ' + str(train_X.shape))

# change y [1D] to Y [2D] sparse array coding class
n_examples = len(train_y)
labels = np.unique(train_y)
train_Y = np.zeros((n_examples, len(labels)))
for i in range(len(labels)):
    # Find examples with with a Label = lables(i)
    ix_tmp = np.where(train_y == labels[i])[0]
    train_Y[ix_tmp, i] = 1

# dostosowywać do potrzeb i datasetu
# walidacja na tą chwilę nie przeprowadzana
valid_X = data[250:277]
valid_y = target[250:277]
print('Shape of validation set: ' + str(valid_X.shape))

# change y [1D] to Y [2D] sparse array coding class
n_examples = len(valid_y)
labels = np.unique(valid_y)
valid_Y = np.zeros((n_examples, len(labels)))
for i in range(len(labels)):
    # Find examples with with a Label = lables(i)
    ix_tmp = np.where(valid_y == labels[i])[0]
    valid_Y[ix_tmp, i] = 1


# dostosowywać do potrzeb i datasetu
test_X = data[800:]
test_y = target[800:]
print('Shape of test set: ' + str(test_X.shape))

# change y [1D] to Y [2D] sparse array coding class
n_examples = len(test_y)
labels = np.unique(test_y)
test_Y = np.zeros((n_examples, len(labels)))
for i in range(len(labels)):
    # Find examples with with a Label = lables(i)
    ix_tmp = np.where(test_y == labels[i])[0]
    test_Y[ix_tmp, i] = 1


# Creating the MLP object
classifier = perceptron.Perceptron(layer_sizes = [13, 8, 5])
print(classifier)

# Training with Backpropagation and 400 iterations
iterations = 1000
loss = np.zeros([iterations,1])

for ix in range(iterations):
    classifier.train(train_X, train_Y, 1)
    Y_hat = classifier.solve(train_X)
    #print(Y_hat)
    y_tmp = np.argmax(Y_hat, axis=1)
    y_hat = labels[y_tmp]
    
    loss[ix] = (0.5)*np.square(y_hat - train_y).mean()

# Training Accuracy
Y_hat = classifier.solve(train_X)
#print(Y_hat)
y_tmp = np.argmax(Y_hat, axis=1)
y_hat = labels[y_tmp]

acc = np.mean(1 * (y_hat == train_y))
print('Training Accuracy: ' + str(acc*100))

# Test Accuracy
Y_hat = classifier.solve(test_X)
print(Y_hat)
y_tmp = np.argmax(Y_hat, axis=1)
y_hat = labels[y_tmp]

n_same = 0
for i in range(len(y_hat)):
    if (y_hat[i] > 0 and test_y[i] > 0) or (y_hat[i] < 1 and test_y[i] < 1):
        n_same += 1

print(n_same / len(y_hat))

acc = np.mean(1 * (y_hat == test_y))
print('Testing Accuracy: ' + str(acc*100)) 