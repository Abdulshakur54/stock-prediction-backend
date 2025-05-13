from django.urls import path
from account import views as UserViews
from .views import HttpOnlyTokenObtainPairView, HttpOnlyTokenRefreshView, Logout

urlpatterns = [
    path('register/', UserViews.RegistrationView.as_view(), name='register'),
    path('token/', HttpOnlyTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('refresh/', HttpOnlyTokenRefreshView.as_view(), name='token_refresh'),
    path('logout/', Logout.as_view(), name='logout'),
]
