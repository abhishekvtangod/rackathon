import csv
from utils import User, r
from net import Classifier
import numpy as np

np.random.seed(1)
print("1")
with open ('Train/data.csv', 'r') as f:
    training_data = [list(map(float, x)) for x in csv.reader(f)] 


meanList = []
stdList = []

labels = []
info = []


for item in training_data:
    if len(item[1:]) == 9: #number of statistics on each user
        labels.append(item[0]) #0 = human, 1 = bot
        info.append(item[1:]) #the user's information: do they have "bot" in the name? Are they repetitive? ect.


print("2")


def normalize(lists):
    '''
    Calculate the standard deviation and mean for all of the data; this
    is used to normalize new inputs and map the data to a lower range so that
    naturally larger datapoints do not overwhelm the neural network.
    '''
    newLists = []
    x = np.array(lists)
    for idx in range(len(lists[0])): #This loop goes through each column of the data
        y = np.array(x.T[idx])
        meanList.append(y.mean())
        stdList.append(y.std())
        new = (y-y.mean())/y.std() #Calculate new value using Standard Score (Z-Score)
        newLists.append(new.tolist())
    return np.array(newLists).T.tolist() 

info = normalize(info)

print("3")

def normalize_alone(single):
    '''
    Uses the known values to normalize new input, or a single list of
    a user's statistics.
    '''
    newList = []
    x = np.array(single)
    for idx in range(len(single)):
        y = np.array(x[idx])
        new = (y-meanList[idx])/stdList[idx]
        newList.append(new.tolist())
    return np.array(newList).T.tolist()


print("4")

training_data = []
for idx, item in enumerate(labels):
    training_data.append([item] + info[idx])


print("5")

NN = Classifier(9, 64, 1, 0.15)
for x in range(100):
    for item in training_data:
        print(x)
        NN.train(item[1:], item[0]) #Feed example --> Output Guess --> Backpropagate Error --> Adjust weights (see net.py for details)

print("6")

def isABot(input_user):
    Test = User(input_user)
    odds = NN.query( normalize_alone(Test.data[1:])) 
    return odds[0][0]

print("7")

isABot("redaniket")