from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from django.middleware.csrf import get_token
from stock_prediction.settings import HTTP_ONLY, HTTPS
from datetime import timedelta
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny


class HttpOnlyTokenObtainPairView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        response =  super().post(request, *args, **kwargs)
        accessToken = response.data['access']
        refreshToken = response.data['refresh']
        res = Response()
        res.set_cookie(
            key='token',
            value=accessToken,
            httponly=HTTP_ONLY,
            secure=HTTPS,
            samesite='None',
            expires=timezone.now() + timedelta(minutes=15)
        )
        res.set_cookie(
            key='refresh',
            value=refreshToken,
            httponly=HTTP_ONLY,
            secure=HTTPS,
            samesite='None',
            expires=timezone.now() + timedelta(days=1)
        )

        csrfToken =  get_token(request)
        res.set_cookie(
            key='csrf',
            value=csrfToken,
            httponly=HTTP_ONLY,
            secure=HTTPS,
            samesite='None',
            expires=timezone.now() + timedelta(minutes=15)
        )
        
        res.data = {'csrfToken': csrfToken}
        res.status_code = status.HTTP_200_OK

        return res
    
    
class HttpOnlyTokenRefreshView(TokenRefreshView):
    def post(self, request, *args, **kwargs):
        refreshToken = request.COOKIES.get('refresh')
        if refreshToken is None:
            return Response({'error': 'Refresh token not found'}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            request.data['refresh'] = refreshToken
            response =  super().post(request, *args, **kwargs)
            accessToken = response.data['access']
            res = Response()
            res.set_cookie(
            key='token',
            value=accessToken
        )
        res.set_cookie(
            key='refresh',
            value=refreshToken,
            httponly=HTTP_ONLY,
            secure=HTTPS
        )
        res.status_code = status.HTTP_200_OK
        return res
    
class Logout(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        res = Response()
        res.delete_cookie(
            key='token',
            samesite='None'
        )
        res.delete_cookie(
            key='refresh',
            samesite='None'
        )
        res.delete_cookie(
            key='csrf',
            samesite='None'
        )
        res.status_code = status.HTTP_200_OK
        return res


    
