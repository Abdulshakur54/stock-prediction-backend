from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from django.middleware.csrf import get_token
from stock_prediction.settings import HTTP_ONLY, HTTPS, SAME_SITE
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from datetime import datetime, timedelta
from .serializers import TickerSerializer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import os
from django.conf import settings
from sklearn.preprocessing import MinMaxScaler
from .cloudinary import Cloudinary
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import requests


class HttpOnlyTokenObtainPairView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        accessToken = response.data["access"]
        refreshToken = response.data["refresh"]
        res = Response()
        res.set_cookie(
            key="token",
            value=accessToken,
            httponly=HTTP_ONLY,
            secure=HTTPS,
            samesite=SAME_SITE,
            expires=datetime.now() + timedelta(minutes=15),
        )
        res.set_cookie(
            key="refresh",
            value=refreshToken,
            httponly=HTTP_ONLY,
            secure=HTTPS,
            samesite=SAME_SITE,
            expires=datetime.now() + timedelta(days=1),
        )

        csrfToken = get_token(request)
        res.set_cookie(
            key="csrf",
            value=csrfToken,
            httponly=False,
            secure=HTTPS,
            samesite=SAME_SITE,
            expires=datetime.now() + timedelta(days=1),
        )

        res.data = {"csrfToken": csrfToken}
        res.status_code = status.HTTP_200_OK

        return res


class HttpOnlyTokenRefreshView(TokenRefreshView):
    def post(self, request, *args, **kwargs):
        refreshToken = request.COOKIES.get("refresh")

        if refreshToken is None:
            return Response(
                {"error": "Refresh token not found"},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        else:
            request.data["refresh"] = refreshToken
            response = super().post(request, *args, **kwargs)
            accessToken = response.data["access"]
            res = Response()
            res.set_cookie(
                key="token",
                value=accessToken,
                httponly=HTTP_ONLY,
                secure=HTTPS,
                samesite=SAME_SITE,
            )
        res.status_code = status.HTTP_200_OK
        return res


class Logout(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        res = Response()
        res.delete_cookie(key="token", samesite=SAME_SITE)
        res.delete_cookie(key="refresh", samesite=SAME_SITE)
        res.delete_cookie(key="csrf", samesite=SAME_SITE)
        res.status_code = status.HTTP_200_OK
        return res


class PredictStock(APIView):

    def post(self, request):
        serializer = TickerSerializer(data=request.data)
        if serializer.is_valid():
            ticker = request.data["ticker"]
            now = datetime.now()
            start = datetime(now.year - 10, now.month, now.day)
            end = now
            df = yf.download(ticker, start, end)
            if df.empty:
                return Response(
                    {
                        "error": "No data found for the given ticker.",
                        "status": status.HTTP_404_NOT_FOUND,
                    }
                )
            df = df.reset_index()
            cld = Cloudinary()
            # plot Close prices
            plt.switch_backend("AGG")
            plt.figure(figsize=(12, 5))
            plt.plot(df.Close, label="Closing Price")
            plt.title(f"Closing price of {ticker}")
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.legend()
            # Save the plot to a file
            plot_img_path = f"{ticker}_plot.png"
            plot_img = cld.save_image(plt, plot_img_path)

            # 100 Days moving average
            ma100 = df.Close.rolling(100).mean()
            plt.switch_backend("AGG")
            plt.figure(figsize=(12, 5))
            plt.plot(df.Close, label="Closing Price")
            plt.plot(ma100, "r", label="100 DMA")
            plt.title(f"100 Days Moving Average of {ticker}")
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.legend()
            plot_img_path = f"{ticker}_100_dma.png"
            plot_100_dma = cld.save_image(plt, plot_img_path)

            # 200 Days moving average
            ma200 = df.Close.rolling(200).mean()
            plt.switch_backend("AGG")
            plt.figure(figsize=(12, 5))
            plt.plot(df.Close, label="Closing Price")
            plt.plot(ma100, "r", label="100 DMA")
            plt.plot(ma200, "g", label="200 DMA")
            plt.title(f"200 Days Moving Average of {ticker}")
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.legend()
            plot_img_path = f"{ticker}_200_dma.png"
            plot_200_dma = cld.save_image(plt, plot_img_path)

            # split data into training and testing dataset
            df_fulltrain, df_test = train_test_split(df, test_size=0.2, shuffle=False)
            scaler = MinMaxScaler(feature_range=(0, 1))
            # Load ML Model
            # model = load_model("stock_prediction_model.keras")

            data_test = df_test.Close
            data_test_arr = scaler.fit_transform(data_test)
            X_test, y_test = [], []
            seq_num = 100
            for i in range(seq_num, len(data_test_arr)):
                X_test.append(data_test_arr[i - 100 : i])
                y_test.append(data_test_arr[i, 0])
            X_test, y_test = np.array(X_test), np.array(y_test)
            url = 'https://abdulshakur54-stockprediction.hf.space'
            payload = {'ticker': ticker}
            headers = {'Content-Type': 'application/json'}
            res = requests.post(url, json=payload, headers=headers)
            prediction = res.json()['prediction']
            # prediction using the model
            y_predict_test = np.array(prediction)
            y_predict_test = scaler.inverse_transform(y_predict_test).flatten()
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            # Plot the final prediction
            plt.switch_backend("AGG")
            plt.figure(figsize=(12, 5))
            plt.plot(y_test, "b", label="Original Price")
            plt.plot(y_predict_test, "r", label="Predicted Price")
            plt.title(f"Final Prediction for {ticker}")
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.legend()
            plot_img_path = f"{ticker}_final_prediction.png"
            plot_prediction = cld.save_image(plt, plot_img_path)

            # Model Evaluation
            # Mean Squared Error (MSE)
            mse = mean_squared_error(y_test, y_predict_test)

            # Root Mean Squared Error (RMSE)
            rmse = np.sqrt(mse)

            # R-Squared
            r2 = r2_score(y_test, y_predict_test)

            return Response(
                {
                    "status": "success",
                    "plot_img": plot_img,
                    "plot_100_dma": plot_100_dma,
                    "plot_200_dma": plot_200_dma,
                    "plot_prediction": plot_prediction,
                    "mse": round(mse, 2),
                    "rmse": round(rmse, 2),
                    "r2": round(r2, 2),
                }
            )
        else:
            return Response(
                {"message": "Invalid Ticker"}, status=status.HTTP_400_BAD_REQUEST
            )
