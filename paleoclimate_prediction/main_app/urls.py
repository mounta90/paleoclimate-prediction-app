from django.urls import path, include
from . import views

urlpatterns = [
    path("", views.home, name="main-home"),
    path("prediction", views.prediction, name="main-prediction")
]
