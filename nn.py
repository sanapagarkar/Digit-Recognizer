import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('mnist_test.csv')
x=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values

one_hot_labels = np.zeros((9999, 10))
for i in range(9999):
    one_hot_labels[i, y[i]] = 1
    
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,one_hot_labels,test_size=0.2,random_state=0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

instances = x_train.shape[0]
attributes = x_train.shape[1]
hidden_nodes = 397 #(784+10)/2
output_labels = 10

wh = np.random.rand(attributes,hidden_nodes)
bh = np.random.randn(hidden_nodes)

wo = np.random.rand(hidden_nodes,output_labels)
bo = np.random.randn(output_labels)
lr = 0.05

error_cost = []

for epoch in range(10001):
#Feedforward
    #P1
    zh = np.dot(x_train, wh) + bh
    zh=zh/attributes
    zh=zh/attributes
    ah = sigmoid(zh)

    #P2
    zo = np.dot(ah, wo) + bo
    zo=zo/attributes
    zo=zo/attributes
    ao = softmax(zo)

#Back Propagation
#P1
    dcost_dzo = ao - y_train
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

    dcost_bo = dcost_dzo

#p2
    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = x_train
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    dcost_bh = dcost_dah * dah_dzh
    
    #Update weights
    wh -= lr * dcost_wh
    bh -= lr * dcost_bh.sum(axis=0)

    wo -= lr * dcost_wo
    bo -= lr * dcost_bo.sum(axis=0)
    tsum=0
    total=0
    if epoch % 200 == 0:
        print(epoch)
        loss = np.sum(-y_train * np.log(ao))
        print('Loss function value: ', loss)
        error_cost.append(loss)
        print("next")
for i in range(0,7999):
    for r in range(0,10):
        if(ao[i][r]>=0.5):
            ao[i][r]=1
        else:
            ao[i][r]=0
            
#calculating accuracy for train set to see how well it fit train set
count_train = ao.size - np.count_nonzero(ao == y_train)
count_train == 0, count / ao.size
total_train=ao.size/10
correct_train = total_train-count_train
accuracy_train = correct_train/total_train
accuracy_train*=100
print("Accuracy on train set ",accuracy_train)

xi=[]
for i in range (0, len(error_cost)):
    xi.append(i)
#visualizing cost function 
plt.scatter(xi, error_cost, color = 'red')
plt.title('Cost function')
plt.xlabel('i')
plt.ylabel('cost function value')
plt.show()

#test set
zh_test = np.dot(x_test, wh) + bh
zh_test=zh_test/attributes
zh_test=zh_test/attributes
ah_test = sigmoid(zh_test)

zo_test = np.dot(ah_test, wo) + bo
zo_test=zo_test/attributes
zo_test=zo_test/attributes
ao_test = softmax(zo_test)
loss_test = np.sum(-y_test * np.log(ao_test))
print('Loss function value for test : ', loss_test)

for i in range(0,2000):
    for r in range(0,10):
        if(ao_test[i][r]>=0.5):
            ao_test[i][r]=1
        else:
            ao_test[i][r]=0
#calculating accuracy for test set
count = ao_test.size - np.count_nonzero(ao_test == y_test)
count == 0, count / ao_test.size
total=ao_test.size/10
correct = total-count
accuracy = correct/total
accuracy*=100

print("Accuracy on test set ",accuracy)
