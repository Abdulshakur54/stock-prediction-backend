from rest_framework import serializers

class TickerSerializer(serializers.Serializer):
    ticker = serializers.CharField(required=True, max_length=20)