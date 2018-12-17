import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas_datareader import data as pdr
import numpy as np
import fix_yahoo_finance as yf
import datetime, math, time

today = str(datetime.date.today())
yf.pdr_override()

####################get data####################

ticker = input("Please enter a ticker:")
df = pdr.get_data_yahoo(ticker, '2008-01-03', today)#veryyyyyyyy unstable
df = df[['Open','High','Low','Close','Volume']]
weekdays = pd.date_range('2008-01-03', today, freq='B')
df = df.reindex(weekdays)
df = df.fillna(method='ffill')

####################calculate moving averages and other data####################

#df['50day'] = df.Close.rolling(50).mean().shift()
#df['200day'] = df.Close.rolling(200).mean().shift()
df['HL%'] = (df['High'] - df['Low']) / df['Low'] * 100
df['%Change'] = (df['Close'] - df['Open']) / df['Open'] * 100

df = df[['Close','HL%','%Change','Volume']]
df.fillna(-99999, inplace=True)
#backup = df
#print(df.iloc[20:60,0:3])

#####################analyze and train####################

length = len(df)
forecast_col = 'Close'
forecast_out = int(math.ceil(0.01*length))

df['label'] = df[forecast_col].shift(-forecast_out)
#print(df.tail())
#df.dropna(inplace=True)
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_data = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(df['label'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

prediction = clf.predict(X_data)
print(len(X),len(y))
print(prediction, accuracy)
print(forecast_out)

df['forecast'] = np.nan
today = df.iloc[-1].name
last = today.timestamp()
day = 86400
next = last + day

length1 = len(df.columns)
for i in prediction:
    nextday = datetime.datetime.fromtimestamp(next)
    next += day
    df.loc[nextday] = [np.nan for all in range(length1 - 1)] + [i]

#####################plot data####################

plt.figure(figsize=(15,10))
plt.grid(True)
plt.plot(df['Close'], label=ticker)
plt.plot(df['forecast'], label='future')
#plt.plot(df['50day'], label='50 day')
#plt.plot(df['200day'], label='200 day')
plt.plot(df['HL%'], label='HL %')
plt.plot(df['%Change'], label='% Change')
plt.legend(loc=2)
plt.xlabel('date')
plt.ylabel('price')
plt.show()



'''
def getStockData():
    stock = request.args.get('stock', default=None, type=None)
    quandl.ApiConfig.api_key = "qWcicxSctVxrP9PhyneG"
    allData = quandl.get('WIKI/'+stock)
    dataLength = 251
    allDataLength = len(allData)
    firstDataElem = math.floor(random.random()*(allDataLength-dataLength))
    mlData = allData[0:firstDataElem+dataLength]

    def FormatForModel(dataArray):
        dataArray = dataArray[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
        dataArray['HL_PCT'] = (dataArray['Adj. High'] - dataArray['Adj. Close']) / dataArray['Adj. Close'] * 100.0
        dataArray['PCT_change'] = (dataArray['Adj. Close'] - dataArray['Adj. Open']) / dataArray['Adj. Open'] * 100.0
        dataArray = dataArray[['Adj. Close', 'HL_PCT', 'PCT_change','Adj. Volume']]
        dataArray.fillna(-99999, inplace=True)
        return dataArray

    mlData = FormatForModel(mlData)

    forecast_col = 'Adj. Close'
    forecast_out = int(math.ceil(0.12*dataLength))

    mlData['label'] = mlData[forecast_col].shift(-forecast_out)
    mlData.dropna(inplace=True)

    X = np.array(mlData.drop(['label'],1))
    X = preprocessing.scale(X)
    X_data = X[-dataLength:]
    X = X[:-dataLength]
    data = mlData[-dataLength:]
    mlData = mlData[:-dataLength]
    y = np.array(mlData['label'])

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    prediction = clf.predict(X_data)
    data = data[['Adj. Close']]
    data = data.rename(columns={'Adj. Close':'EOD'})
    data['prediction'] = prediction[:]
    data = data.to_json(orient='table')
    return jsonify(data)
'''












