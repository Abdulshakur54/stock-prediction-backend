

from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import joblib
class PredictStock(APIView):

    def post(self, request):

            now = datetime.now()
            start = datetime(now.year - 10, now.month, now.day)
            end = now
            df = yf.download(ticker, start, end)

            # split data into training and testing dataset
            df_fulltrain, df_test = train_test_split(df, test_size=0.2, shuffle=False)
            # Load ML Model
            model = load_model("stock_prediction_model.keras")
            # Load saved MinMaxScaler
            scaler = joblib.load('stock_scaler.pkl')
            data_test = df_test.Close
            data_test_arr = scaler.fit_transform(data_test)
            X_test, y_test = [], []
            seq_num = 100
            for i in range(seq_num, len(data_test_arr)):
                X_test.append(data_test_arr[i - 100 : i])
                y_test.append(data_test_arr[i, 0])
            X_test, y_test = np.array(X_test), np.array(y_test)

            # prediction using the model
            y_predict_test = model.predict(X_test)
            return {
                'prediction': y_predict_test.tolist()
            }
           