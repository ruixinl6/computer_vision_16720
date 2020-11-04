import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 80
# pick a batch size, learning rate
batch_size = 48
learning_rate = 0.2e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}
total_losses = []
avg_accs = []

# initialize layers here
initialize_weights(1024,64,params,'layer1')
initialize_weights(64,36,params,'output')


# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss = total_loss+loss
        total_acc = total_acc+acc
        delta1 = probs-yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)
        params['W' + 'output'] = params['W' + 'output']-learning_rate*params['grad_W' + 'output']
        params['b' + 'output'] = params['b' + 'output']-learning_rate*params['grad_b' + 'output']
        params['W' + 'layer1'] = params['W' + 'layer1']-learning_rate*params['grad_W' + 'layer1']
        params['b' + 'layer1'] = params['b' + 'layer1']-learning_rate*params['grad_b' + 'layer1']
        # training loop can be exactly the same as q2!
        
    total_acc = total_acc/len(batches)
    total_losses.append(total_loss)
    avg_accs.append(total_acc)
    
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
    # run on validation set and report accuracy! should be above 75%
valid_acc = None
h1 = forward(valid_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)

print('Validation accuracy: ',valid_acc)

import matplotlib.pyplot as plt
plt.plot(range(max_iters), np.atleast_1d(total_losses)/batch_size)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.show()

plt.plot(range(max_iters),avg_accs,'r',range(max_iters),valid_acc*np.ones((max_iters)),'b')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()



if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
from mpl_toolkits.axes_grid1 import ImageGrid

initialize_weights(1024, 64, saved_params, 'init')
weightsInit = saved_params['Winit']
fig = plt.figure()
grid = ImageGrid(fig, 111, (8,8))
for i in range(64):
    w = weightsInit[:, i].reshape(32, 32)
    grid[i].imshow(w)
plt.show()

weightsLearned = params['W' + 'layer1']
fig = plt.figure()
grid = ImageGrid(fig, 111, (8,8))
for i in range(64):
    w = weightsLearned[:, i].reshape(32, 32)
    grid[i].imshow(w)
plt.show()

# Q3.1.3
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
predict = np.argmax(probs, axis=1)
correct = np.argmax(valid_y, axis=1)
for i in range(correct.shape[0]):
    confusion_matrix[correct[i], predict[i]] += 1



import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
