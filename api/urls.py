from django.urls import path
from account import views as UserViews
from .views import HttpOnlyTokenObtainPairView, HttpOnlyTokenRefreshView, Logout, PredictStock

urlpatterns = [
    path('register/', UserViews.RegistrationView.as_view(), name='register'),
    path('protected/', UserViews.ProtectedView.as_view(), name='protected'),
    path('token/', HttpOnlyTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('refresh/', HttpOnlyTokenRefreshView.as_view(), name='token_refresh'),
    path('predict/', PredictStock.as_view(), name='predict'),
    path('logout/', Logout.as_view(), name='logout'),
]
