from django.urls import path, include
from . import views

urlpatterns = [
    path("", views.home, name="main-home"),
    path("results", views.results, name="main-results"),
    path("prediction", views.prediction, name="main-prediction"),
]
