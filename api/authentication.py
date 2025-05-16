from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, AuthenticationFailed

class CookieJWTAuthentication(JWTAuthentication):
    def authenticate(self, request):
        # Read the access token from cookies
        accessToken = request.COOKIES.get('token')
        if not accessToken:
            return None

        try:
            validated_token = self.get_validated_token(accessToken)
            return self.get_user(validated_token), validated_token
        except InvalidToken as e:
            print('error: ', e)
            raise AuthenticationFailed('Invalid token')
        except Exception as e:
            print('Exception: ',e)

