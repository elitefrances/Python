import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from utils import *



def run_rgrs(df, regressors, features, targets):
    cols = [t.replace(' Target', '') for t in targets]
    pt = pd.DataFrame(index=['Train R^2', 'Test R^2', 'Train MSE', 'Test MSE'], columns=cols)
    for i, regressor in enumerate(regressors):
        target = targets[i]
        regressor.fit(df['Train'][features], df['Train'][target])
        train_y_pred = regressor.predict(df['Train'][features])
        test_y_pred = regressor.predict(df['Test'][features])
        r2_train = calculate_r2(train_y_pred, df['Train'][target])
        r2_test = calculate_r2(test_y_pred, df['Test'][target])
        mse_train = calculate_mse(train_y_pred, df['Train'][target])
        mse_test = calculate_mse(test_y_pred, df['Test'][target])
        pt[cols[i]] = [r2_train, r2_test, mse_train, mse_test]
    return pt


pathname = './data'
data = np.load(os.path.join(pathname, 'preprocessed_continuous_5.npy'))[0]

features_amount = [#'Security',
            #'Declared_Date',
            #'Dividend_Amount',
            #'Ex_Date',
            'ALTMAN_Z_SCORE',
            'ROC_WACC_RATIO',
            'WACC',
            'WACC_ECON_VALUE_ADDED',
            'WACC_EVA_SPREAD',
            'WACC_NOPAT',
            'Dividend_Frequency_Annual',
            #'Dividend_Frequency_Semi_Anl',
            'Dividend_Type_Final',
            'Dividend_Type_Interim',
            'Dividend_Type_Regular_Cash',
            'BICS_LEVEL_1_SECTOR_NAME_Communications',
            'BICS_LEVEL_1_SECTOR_NAME_Consumer_Discretionary',
            'BICS_LEVEL_1_SECTOR_NAME_Consumer_Staples',
            'BICS_LEVEL_1_SECTOR_NAME_Energy',
            'BICS_LEVEL_1_SECTOR_NAME_Financials',
            'BICS_LEVEL_1_SECTOR_NAME_Health_Care',
            'BICS_LEVEL_1_SECTOR_NAME_Industrials',
            'BICS_LEVEL_1_SECTOR_NAME_Materials',
            'BICS_LEVEL_1_SECTOR_NAME_Technology',
            'BICS_LEVEL_1_SECTOR_NAME_Utilities',
            'CNTRY_OF_RISK_CN',
            'CNTRY_OF_RISK_HK',
            #'Omit',
            #'New_Label',
            #'Ex_Date_Change',
            #'Ex_Date_Change_Target',
            #'Dividend_Amount_Target',
            #'Omit_Target',
            #'Ex_Date_Change_Lagged_1',
            #'Ex_Date_Change_Lagged_2',
            #'Ex_Date_Change_Lagged_3',
            #'Ex_Date_Change_Lagged_4',
            #'Ex_Date_Change_Lagged_5',
            'Dividend_Amount_Lagged_1',
            'Dividend_Amount_Lagged_2',
            'Dividend_Amount_Lagged_3',
            'Dividend_Amount_Lagged_4',
            'Dividend_Amount_Lagged_5',
            #'Omit_Lagged_1',
            #'Omit_Lagged_2',
            #'Omit_Lagged_3',
            #'Omit_Lagged_4',
            #'Omit_Lagged_5'
            ]


############ linear regression ############
from sklearn.linear_model import LinearRegression
rgs_lr = [LinearRegression()]
pt_lr = run_rgrs(data, regressors=rgs_lr, features=features_amount, targets=['Dividend_Amount_Target'])
############ linear regression ############


############ random forest ############
from sklearn.ensemble import RandomForestRegressor
rgs_rf = [RandomForestRegressor(max_depth=5, n_estimators=30)]
pt_rf = run_rgrs(data, regressors=rgs_rf, features=features_amount, targets=['Dividend_Amount_Target'])
############ random forest ############


############ LSTM ############
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

def prepare_LSTM_features(df, features, target, lag):
    _df = df.copy()
    _df[target+'_Lagged_0'] = _df[target+'_Target']
    
    x = np.zeros((len(_df), lag, len(features)-lag+1))
    #y = np.zeros((len(_df), lag, 1))
    y = np.zeros((len(_df), 1))
    for i in range(len(_df)):
        for k in range(lag):
            x[i,k,:] = _df.iloc[i][[f for f in features if target not in f] + [target+'_Lagged_%d'%(lag-k)]]
            #y[i,k,:] = _df.iloc[i][target+'_Lagged_%d'%(lag-k-1)]
        y[i,:] = _df.iloc[i][target+'_Target']
    return x, y

def run_LSTM(data, features, lag, target, optimizer='adam', num_cell=500, early_stop=False,
             loss='mean_squared_error', max_epoch=1000, bs=32, dropout=0, recurrent_dropout=0.0):
    train_x, train_y = prepare_LSTM_features(data['Train'], features, target, lag)
    test_x, test_y = prepare_LSTM_features(data['Test'], features, target, lag)
    
    callbacks = []
    if early_stop:
        callbacks += [EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')]
    
    ## Caution: calculate loss only on the last output
    model = Sequential()
    model.add(LSTM(num_cell, input_shape=(train_x.shape[1], train_x.shape[2]), 
                   return_sequences=False, dropout=dropout, recurrent_dropout=recurrent_dropout))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)

    history = model.fit(train_x, train_y, epochs=max_epoch, batch_size=bs, verbose=2, 
                        validation_data=(test_x, test_y), callbacks=callbacks)
    
    train_y_pred = model.predict(train_x)
    test_y_pred = model.predict(test_x)

    pt = pd.DataFrame(index=['Train R^2', 'Test R^2', 'Train MSE', 'Test MSE'], columns=[target])
    r2_train = calculate_r2(train_y_pred[:,0], train_y[:,0])
    r2_test = calculate_r2(test_y_pred[:,0], test_y[:,0])
    mse_train = calculate_mse(train_y_pred[:,0], train_y[:,0])
    mse_test = calculate_mse(test_y_pred[:,0], test_y[:,0])
    pt[target] = [r2_train, r2_test, mse_train, mse_test]
    
    return pt, history, model

def plot_loss(history, title=None):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.fill_between(np.arange(len(history.history['loss'])), history.history['loss'], history.history['val_loss'], facecolor="none", edgecolor='black', hatch="//", label='gap')
    plt.legend()
    if title:
        plt.title(title)
    plt.show(block=False)


nc = 200
bs = 64
dp = 0.3
rdp = 0.3

print("\n\n===============================\nnum_cells=%d\tbatch_sizes=%d" %(nc, bs))
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#optimizer = 'adam'
result_lstm = run_LSTM(data, features_amount, 5, 'Dividend_Amount', optimizer=optimizer, early_stop=True, 
                                   num_cell=nc, bs=bs, dropout=dp, recurrent_dropout=rdp)
plot_loss(result_lstm[1], title='dp=%.1f rdp=%.1f cell=%d' %(dp, rdp, nc))
############ LSTM ############


############ Print Results ############
print("\n===============================\nLinear Regression Results: ")
print(pt_lr)
print()

print("\n===============================\nRandom Forest Results: ")
print(pt_rf)
print()

print("\n===============================\nLSTM (num_cells=%d, batch_sizes=%d) Results: " %(nc, bs))
print(result_lstm[0])
print()
############ Print Results ############

model_lstm = result_lstm[2]
model_lstm.save_weights('./models/weights/pred_amount_lstm.h5')
print('pred_amount lstm model is saved in  ./models/weights/pred_amount_lstm.h5')