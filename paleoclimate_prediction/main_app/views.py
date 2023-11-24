from django.shortcuts import render

from .functions.linear_prediction import get_plot, linear_prediction
from .functions.random_forest_prediction import random_forest_prediction


# Create your views here.
def home(request):

    page_context = {
        "title": "Home",
    }
    return render(
        request=request,
        template_name="main_app/home.html",
        context=page_context,
    )


def prediction(request):
    prediction_result = linear_prediction()
    prediction_graph = get_plot([0, 1, 2, 3, 4], [0, 2, 4, 6, 8],)
    page_context = {
        "title": "Prediction",
        "prediction_title": "Prediction 1",
        "prediction_graph": prediction_graph,
        "prediction_result": prediction_result,
    }
    return render(request=request, template_name="main_app/prediction.html", context=page_context,)
