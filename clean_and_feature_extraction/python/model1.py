import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import sys,os

def checkcommand():
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python model.py data-file')
    else:
        datafl = sys.argv[1]
        if os.path.isfile(datafl):
            return datafl
        else:
            raise SystemExit('Error: %s Not Exist!' % datafl)

def load(fl):
    print('load the %s......' % fl)
    df = pd.read_csv(fl)
    print('add num_attractions & norm scraped &weather......')
    #df = clean(df)
    print('The shape of raw data:', df.shape)
    print('Columns:')
    print(df.columns.values)
    return df

def clean(df):
    #df = df.sample(frac=0.1)
    #df.drop(columns='days_from_scraped',inplace=True)
    #df.drop(columns='availability_90',inplace=True)
    print('....')
    return df


def separate_training_test(dataset):
    #shuffle ids
    ids = dataset.id.unique()
    np.random.shuffle(ids)
    ids_train = ids[:int(0.8 * len(ids))]
    ids_test = ids[int(0.8 * len(ids)):]
    dataset_train = dataset[dataset['id'].isin(ids_train)].drop(columns='id')
    dataset_test = dataset[dataset['id'].isin(ids_test)].drop(columns='id')
    #get features and target
    x_train = dataset_train.drop(columns = 'price').values
    y_train = np.log(dataset_train['price'].values)
    
    x_test = dataset_test.drop(columns = 'price').values
    y_test = np.log(dataset_test['price'].values)
    
    return x_train,y_train,x_test,y_test

def xgboost(x_train,y_train,x_test,y_testi,cols):
    print('start xgboost training......')
    dtrain = xgb.DMatrix(x_train[:], label=y_train)
    dtest = xgb.DMatrix(x_test[:], label=y_test)
    # specify parameters via map
    param = {'max_depth':15, 'eta':0.3, 'lambda':1, 
             'silent':1, 'objective':'reg:linear', 
             #'tree_method':'gpu_hist', 
             'eval_metric':'rmse'}
    num_round = 20
    bst = xgb.train(param, dtrain, num_round)
    # make prediction
    preds = bst.predict(dtest)
    print('Training error:') 
    print(metrics.median_absolute_error(np.exp(y_test), np.exp(preds)))

    df_f = pd.DataFrame(sorted(bst.get_score().items(), key=lambda x: x[1],reverse=True)).set_index([0]) 
    fmax = df_f.max()
    df_f = (df_f/fmax).reset_index()
    df_f['features'] = df_f.apply(lambda x: cols[int(x[0][1:])] , axis=1)
    pd.set_option('display.max_rows',None)
    print(df_f)

if __name__=='__main__':
    fl = checkcommand()
    df = load(fl)
    x_train,y_train,x_test,y_test = separate_training_test(df)
    features  = list(df.columns.values)
    features.remove('price')
    features.remove('id')
    xgboost(x_train,y_train,x_test,y_test,features)
    print('finish!')
