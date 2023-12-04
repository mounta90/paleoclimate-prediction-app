from django.urls import path, include
from . import views

urlpatterns = [
    path("", views.home, name="main-home"),
    path("results", views.results, name="main-results"),
    path("prediction-penn", views.prediction_penn, name="main-prediction-penn"),
    path("prediction-wyo", views.prediction_wyo, name="main-prediction-wyo"),
]
