import numpy as np 
from sklearn.neural_network import MLPRegressor
import scipy.io as sio


h = sio.loadmat(r"C:\Users\massi\OneDrive\Desktop\Incendi Boschivi\variables.mat")
print(h['dati'].shape)
print(h['dati2'].shape)
x = np.load(r"C:\Users\massi\OneDrive\Desktop\Input_Ristorante100.npy")
y = np.load(r"C:\Users\massi\OneDrive\Desktop\Output_Ristorante100.npy")
x11 = np.transpose(h['dati2'])
y11 = np.transpose(h['dati'])
# x1 = h['dati2']
# y1 = h['dati']
x = x11
ind1 = [[2,3,4,13,19], [5,6,16,17,18], [9,14,15,20,21], [7,8,10,11,12]]

for type1 in range(4): 
    k_ind = []

    for k in range(4): 
        ind2 = ind1[type1]
        for kk in range(len(ind1[type1])):
            k_ind.append(ind2[kk] -2 + k*20)
    print(k_ind)
    x = x11[:10, k_ind]
    for parameters in range(5): #y11.shape[0]): 
        print(y11.shape)
        y12 = y11[parameters,k_ind]
        print(y12.shape)
        y = np.reshape(y12, (1, y12.shape[0]))
        print(x.shape)
        print(y.shape)
        # y = y1[:, :1]
        # x = x1
        err = 0
        err2 = 0
        min_err = 0
        max_err = 0
        from matplotlib import pyplot as plt

        montecarlo = 1 #100
        index = np.arange(x.shape[1])
        for n in range(montecarlo):
            indices = index
            np.random.shuffle(indices)
            P = int((0.9)*x.shape[1])
            x1 = x[:,indices]
            y1 = y[:,indices]
            x_train = np.transpose(x1[:,:P])
            y_train = np.transpose(y1[:,:P])
            print(x_train.shape)
            print(y_train.shape)
            x_val = np.transpose(x1[:,P:])
            y_val = np.transpose(y1[:,P:])
            # mlp = MLPRegressor(hidden_layer_sizes=(26,26,26),max_iter=1000,activation='relu', solver='adam')
            mlp = MLPRegressor(hidden_layer_sizes=(13,13,13),max_iter=1000,activation='relu', solver='adam')
            mlp.fit(x_train, y_train)

            y_pred = mlp.predict(x_val)
            y_val2 = np.reshape(y_val, (y_val.shape[0],))

            # plt.figure()
            # plt.plot( y_pred - y_val2)
            # plt.title("errore")

            err += np.mean(abs(y_pred - y_val2)**2)
            err2 += np.sqrt(err)
            r_quadro = 1 - np.mean(abs(y_pred - y_val)**2)/np.mean(abs(y_val - np.mean(y_val))**2)
            min_err += np.min(y_pred - y_val2)
            max_err += np.max(y_pred - y_val2)


        print(str(type1) + " " + str(parameters) + " errore medio : " + str(err/montecarlo))
        print(" R quadro : " + str(r_quadro))
        print(" errore minimo : " + str(min_err/montecarlo))

        print(" errore massimo : " + str(max_err/montecarlo))
