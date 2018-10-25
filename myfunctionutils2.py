import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------- #
def softmax(y):
    e_y = np.exp(y - np.max(y))
    return e_y/e_y.sum(axis=0)

def make_y_from_x(X_i):
    y = X_i + [0]
    return y

def convert_name_to_indices(name, char_to_in):
    # Provide name as 'some_name'
    name_ind = []
    for a in name:
        name_ind.append(char_to_in[a])
    return name_ind

def convert_indices_to_onehot(name_ind):
    vocab_size = 27
    name_onehot = []
    Tx = len(name_ind)
    for i in name_ind:
        alpha_onehot = np.zeros((vocab_size,1))
        for j in range(vocab_size):
            if j == i:
                alpha_onehot[j] = 1
        name_onehot.append(alpha_onehot)
    name_onehot = np.array(name_onehot).reshape(Tx, vocab_size)
        
    return name_onehot
    
#------------------------------------------------------ MODEL ----------------------------------------------------------------#

def rnn_step_forward(xt, a_prev, parameters):
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']
    
    a_next = np.tanh(np.dot(Wax, xt)+ np.dot(Waa, a_prev) + ba)         #Remember, here xt is only 1 alphabet
    yt = softmax(np.dot(Wya, a_next) + by)
    
    return a_next, yt

def rnn_forward(X_i, Y_i, parameters):
    a, yhat = {}, {}
    loss = 0
    vocab_size = 27
    na = parameters['Waa'].shape[0]
    a[-1] = np.zeros((na, 1))  #allright
    X_i = np.array(X_i)
    
    Tx = X_i.shape[0]               # Remember, here X_i is only one example (1 whole name)
    for t in range(Tx):
        a[t], yhat[t] = rnn_step_forward((X_i[t]).reshape(vocab_size,1), a[t-1], parameters)
        loss -= np.log(yhat[t][Y_i[t],0])
                      
    cache = (yhat, a)       #Here i am adding X_i only becoz of backprop from other code, otherwise its useless
    return loss, cache
    
def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    
    na = parameters['Waa'].shape[0]
    
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + np.array(gradients['da_next']).reshape(na,1) # backprop into h
    daraw = (1 - a * a) * da
    gradients['dba'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients

def rnn_backward(X, Y, parameters, cache):
    gradients = {}
    vocab_size = 27
    (y_hat, a) = cache
    Waa, Wax, Wya, by, ba = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['ba']
    Tx = np.array(X).shape[0]
    na = parameters['Waa'].shape[0]
    
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['dba'], gradients['dby'] = np.zeros_like(ba), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
    
    for t in reversed(range(Tx)):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, np.array(X[t]).reshape(vocab_size,1), np.array(a[t]).reshape(na,1), np.array(a[t-1]).reshape(na,1))
    
    return gradients, a

def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['ba']  += -lr * gradients['dba']
    parameters['by']  += -lr * gradients['dby']
    return parameters


#--------------------------------------------------- VERY IMPORTANT NOTES ------------------------------------------------------
# - X_i and Y_i are about only 1 single name. You have to pass single single names to model.
# - for passing names to model, you will have to convert each names seperately as ONE_HOT just before passing it
# - so, pass X_i as one hot and Y_i as indices.
# - In given dino-model, they iterating in different way. it passing 5-10 times more iterations than n0. of name in dino-list.
#   so every names gets iterate over 5-10 times.