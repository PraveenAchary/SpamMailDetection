from django.contrib import admin
from django.urls import path
from .views import predict

urlpatterns = [
    # Admin panel
    path('admin/', admin.site.urls),

    # API endpoint for spam prediction
    path('api/predict/', predict, name='predict'),
]