import sys, warnings, random, math

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

CLUSTERS = [["BHF","NVDA"]]
DEBUG = False

'''
GET_CLUSTERS:

- Reads cluster files and picks a threshold sample of the companies.
- args:
 . str path: path to the file where the clusters are split.
 . int threshold: number of companies to be used. '-1' for using all.
'''
def get_clusters(path,threshold,run_all):
    global CLUSTERS
    CLUSTERS = []
    with open(path,'r') as cluster_file:
        for line in cluster_file:
            cl = line.strip().split(' ')[1:]
            if threshold != -1 and len(cl) > threshold:
                cl = random.sample(cl,threshold)
            if len(cl) > 0:
                CLUSTERS.append(cl)
    if run_all:
        while len(CLUSTERS) > 1:
            CLUSTERS[0] += CLUSTERS[-1]
            CLUSTERS = CLUSTERS[:-1]
    if DEBUG:
        for i, cl in enumerate(CLUSTERS):
            print("cluster ",(i+1), ' ', cl)


'''
READ_FILE:

- Reads a company file and generate X's and Y's for regression.
- args:
 . str path: path to file where the company values are.
 . int period: number of days to be used to predict the next.
 . str type_trans: type of transformation performed in the data.
             'var' for numbers in variation (I_j = I_j - I_j-1).
             'inc' for number in incremental variation (I_j = I_j + (I_j-1 + I_j-2); I_0 = 0).
             NONE  for no transformation.
 . str attribute: attribute for prediction. 'open', 'high', 'low', 'close','volume' 
'''
def read_file(path, period, type_trans, attribute='open'):
    X = []
    Y = []

    att_index = 0
    period += 1

    with open(path,'r') as input_file:
        #get attribute index
        line = input_file.readline().strip()
        for i,att in enumerate(line.split(',')):
            if att == attribute:
                att_index = i

        # create X and Y arrays
        for i, line in enumerate(input_file):
            X.append([])
            line = line.strip().split(',')
            
            # adds the current value to all the vector before it
            for j in range(i,i-period,-1):
                if j >= 0:
                    if len(line[att_index]) != 0:
                        X[j].append(float(line[att_index]))
                    else:
                        X[j].append(X[j-1][-1])
                    
            # add labels
            if i >= period:
                if len(line[att_index]) != 0:
                    if type_trans == None:
                        Y.append(float(line[1]))
                    else:
                        Y.append([float(line[1])-X[len(Y)][-1]])
                else:
                    if type_trans == None:
                        Y.append([X[len(Y)][-1]])
                    else:
                        Y.append([0.0])

    # the last vectors will be incomplete so we ddiscard them
    X = X[:-period]

    # data transformation
    if type_trans == 'var' or type_trans == 'inc':
        for x in X:
            for id_x in range(len(x)-1,0,-1):
                x[id_x] -= x[id_x-1]

            if type_trans == 'inc':
                for id_x in range(2,len(x)):
                    x[id_x] += x[id_x-1]

    return X,Y

def classify(X_train,Y_train,X_test,clf):
    clf.fit(X_train,Y_train)
    return clf.predict(X_test)

def pocid_measure(Y_true, Y_pred):
    Y_pred = np.sign(Y_pred)
    Y_true = np.sign(Y_true)
    eq = [i for i, j in zip(Y_true, Y_pred) if i == j]

    return 100*(len(eq)/len(Y_pred))

def theil_measure(X, Y_true, Y_pred, Y_base):
    err1 = err1_ = 0.0
    err2 = err2_ = 0.0
    
    for i in range(1,len(Y_true)):
        err1_ += ((Y_pred[i]-Y_true[i])**2)
        err2_ += ((Y_base[i]-Y_true[i])**2)
        if X[i][-1] != 0:
            err1_ /= X[i][-1]
            err2_ /= X[i][-1]
            err1 += err1_
            err2 += err2_
    return math.sqrt(err1/err2)

def exact_measure(Y_true, Y_pred):
    eq = [i for i, j in zip(Y_true, Y_pred) if i == j]
    return len(eq)

def retransform(X, transform):
    if transform == 'var' or transform == 'inc':
        for x in X:
            for id_x in range(1,len(x)):
                if transform == 'inc':
                    x[id_x] += x[0]
                else:
                    x[id_x] += x[id_x-1]
    return X

def main(path_to_csv, period, transform, attribute, fold):
    if path_to_csv[-1] != '/': path_to_csv += '/'

    X = []
    Y = []

    X_cluster = []
    Y_cluster = []

    # read data
    print('reading files...')
    # read cluster file
    for i, cluster in enumerate(CLUSTERS):
        X_cluster.append([])
        Y_cluster.append([])
        for file in cluster:
            x, y = read_file(path=path_to_csv+file+'_data.csv', period=period, type_trans=transform, attribute=attribute)

            X += x
            Y += y
            X_cluster[i] += x
            Y_cluster[i] += y
            
    X = np.array(X)
    Y = np.array(Y)

    if DEBUG:
        print(X.shape)
        print(Y.shape)

    # CLUSTER EVALUATION    
    for i in range(len(CLUSTERS)):
        if len(CLUSTERS) > 0:
            print("evaluating cluster ",i)

        if fold:
            kf = KFold(n_splits=5)
            kf.get_n_splits(X,Y)
            
            svr_mse = svr_pocid = svr_exact = svr_theils = 0.0
            mlp_mse = mlp_pocid = mlp_exact = mlp_theils = 0.0
            base_mse = base_exact = 0.0

            for n_fold, (train_idx, test_idx) in enumerate(kf.split(X,Y)):
                print(n_fold,' fold.')

                svr_pred = classify(X[train_idx][:,1:], Y[train_idx].ravel(), X[test_idx][:,1:], clf=SVR(C=1.0, epsilon=0.2))
                mlp_pred = classify(X[train_idx][:,1:], Y[train_idx].ravel(), X[test_idx][:,1:], clf=MLPRegressor())
                
                # baseline predict
                if transform == None:
                    baseline_pred = X[test_idx][:,-1:].ravel()
                else:
                    baseline_pred = np.array([0.0]*len(Y[test_idx]))

                # evaluate SVR
                svr_mse += mean_squared_error(Y[test_idx], svr_pred)
                svr_pocid += pocid_measure(Y[test_idx], svr_pred)
                svr_exact += exact_measure(Y[test_idx], svr_pred)
                svr_theils += theil_measure(retransform(X[test_idx],transform), Y[test_idx], svr_pred, baseline_pred)

                # evaluate MLP
                mlp_mse += mean_squared_error(Y[test_idx], mlp_pred)
                mlp_pocid += pocid_measure(Y[test_idx], mlp_pred)
                mlp_exact += exact_measure(Y[test_idx], mlp_pred)
                mlp_theils += theil_measure(retransform(X[test_idx],transform), Y[test_idx], mlp_pred, baseline_pred)

                # evaluate BASELINE
                base_mse += mean_squared_error(Y[test_idx], baseline_pred)
                base_exact += exact_measure(Y[test_idx], baseline_pred)

            # evaluate SVR
            print('SVR MSE: ',      svr_mse/5)
            print('SVR POCID: ',    svr_pocid/5)
            print('SVR EXACT: ',    svr_exact/5)
            print("SVR THEIL'S U: ",svr_theils/5)

            # evaluate MLP
            print('MLP MSE: ',      mlp_mse/5)
            print('MLP POCID: ',    mlp_pocid/5)
            print('MLP EXACT: ',    mlp_exact/5)
            print("MLP THEIL'S U: ",mlp_theils/5)

            # evaluate BASELINE
            print('BASELINE MSE: ',  base_mse/5)
            print('BASELINE EXACT: ',base_exact/5)

        else:
            # split train/test - 20%
            print("LEN ",len(X_cluster[i]))
            try:
                X_train, X_test, Y_train, Y_test = train_test_split(np.array(X_cluster[i],dtype=np.float), np.array(Y_cluster[i],dtype=np.float), test_size=0.2)
            except:
                for L,x in enumerate(X_cluster[i]):
                    print(x, ' ', Y_cluster[i][L])
                exit()
            if DEBUG:
                print(X_train[:,1:].shape)
        
            print('fit predict SVR...')
            print("X_train")
            #print(X_train[:,1:])
            print(X_train.shape)
            print(type(X_train))
            print("Y")
            #print(Y_train)
            print(Y_train.shape)
            print(type(Y_train))
            print("X_test")
            #print(X_test)
            print(X_test[:,1:].shape)
            print(type(X_test[:,1:]))
            print("Y")
            #print(Y_test)
            print(Y_test.shape)
            print(type(Y_test))

            svr_pred = classify(X_train[:,1:], Y_train, X_test[:,1:], clf=SVR(C=1.0, epsilon=0.2))

            print('fit predict MLP...')
            mlp_pred = classify(X_train[:,1:], Y_train, X_test[:,1:], clf=MLPRegressor())
            
            # baseline predict
            if transform == None:
                baseline_pred = X_test[:,-1:].ravel()
            else:
                baseline_pred = np.array([0.0]*len(Y_test))

            # evaluate SVR
            print('SVR MSE: ',      mean_squared_error(Y_test, svr_pred))
            print('SVR POCID: ',    pocid_measure(Y_test, svr_pred))
            print('SVR EXACT: ',    exact_measure(Y_test, svr_pred))
            print("SVR THEIL'S U: ",theil_measure(retransform(X_test,transform), Y_test, svr_pred, baseline_pred))

            # evaluate MLP
            print('MLP MSE: ',      mean_squared_error(Y_test, mlp_pred))
            print('MLP POCID: ',    pocid_measure(Y_test, mlp_pred))
            print('MLP EXACT: ',    exact_measure(Y_test, mlp_pred))
            print("MLP THEIL'S U: ",theil_measure(retransform(X_test,transform), Y_test, mlp_pred, baseline_pred))

            # evaluate BASELINE
            print('BASELINE MSE: ',      mean_squared_error(Y_test, baseline_pred))
            print('BASELINE EXACT: ',    exact_measure(Y_test, baseline_pred))

            if DEBUG:
                for j in range(len(Y_test)):
                    if j < 5:
                        print(Y_test[j], ' ', svr_pred[j],' ',mlp_pred[j], ' ',baseline_pred[j])


if __name__ == '__main__':
    limit = -1
    period = 10
    path = '/home/rico/Documents/aplicacoes_mineracao_dados_am/sandp500/individual_stocks_5yr/individual_stocks_5yr/'
    cluster_file = None
    transform = None
    attribute = 'open'
    run_all = False
    fold = False

    for i,arg in enumerate(sys.argv):
        if arg == '-cfile' or arg == 'cfile':
            cluster_file = sys.argv[i+1]
        if arg == '-limit' or arg == 'limit':
            limit = int(sys.argv[i+1])
        if arg == '-path' or arg == 'path':
            path = sys.argv[i+1]
        if arg == '-p' or arg == 'p':
            period = int(sys.argv[i+1])
        if arg == '-t' or arg == 't':
            transform = sys.argv[i+1]
        if arg == '-att' or arg == 'att':
            attribute = sys.argv[i+1]
        if arg == '-debug' or arg == 'debug':
            DEBUG = True
        if arg == '-all' or arg == 'all':
            run_all = True
        if arg == '-fold' or arg == 'fold':
            fold = True

    if cluster_file != None:
        get_clusters(cluster_file, limit, run_all)
    
    main(path, period, transform, attribute, fold)
