from .serializers import UserSerializer
from django.contrib.auth.models import User
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework import generics
from rest_framework.response import Response
from rest_framework.views import APIView
from api.authentication import CookieJWTAuthentication

class RegistrationView(generics.CreateAPIView):
    serializer_class = UserSerializer
    queryset = User.objects.all()
    permission_classes = [AllowAny]



class ProtectedView(APIView):
    authentication_classes = [CookieJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response({'message': 'request is permitted'})

