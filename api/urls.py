from django.urls import path
from account import views as UserViews

urlpatterns = [
    path('register/', UserViews.RegistrationView.as_view(), name='register')
]
