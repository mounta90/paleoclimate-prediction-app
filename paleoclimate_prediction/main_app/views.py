from django.shortcuts import render

from .functions.graphing import *
from .functions.prediction_models import *


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


def results(request):
    models_before_hpo: dict = model_comparison_before_hpo()
    models_after_hpo: dict = model_comparison_after_hpo()

    plot_before_hpo_values = [value for value in models_before_hpo.values()]
    plot_after_hpo_values = [value for value in models_after_hpo.values()]
    plot_names = [key for key in models_before_hpo.keys()]

    prediction_graph = get_bar_plot(names=plot_names, values1=plot_before_hpo_values, values2=plot_after_hpo_values,)

    page_context = {
        "title": "Prediction",
        "prediction_graph": prediction_graph,
        "prediction_models_before_hpo": models_before_hpo,
        "prediction_models_after_hpo": models_after_hpo,
    }

    return render(request=request, template_name="main_app/results.html", context=page_context,)


def prediction_penn(request):
    page_context = {
        "enthalpy": 30.12,
        "mat": 11.2,
        "gsp": 60,
        "sh": 5.9,
        "rh": 70,
        "enthalpy_prediction": make_enthalpy_prediction_penn,
        "mean_annual_temp_prediction": make_mat_prediction_penn,
        "growing_season_precipitation_prediction": make_gsp_prediction_penn,
        "specific_humidity_prediction": make_sh_prediction_penn,
        "relative_humidity_prediction": make_rh_prediction_penn,
    }

    return render(request=request, template_name="main_app/prediction.html", context=page_context)


def prediction_wyo(request):
    page_context = {
        "enthalpy_prediction": make_enthalpy_prediction_wyo,
        "mean_annual_temp_prediction": make_mat_prediction_wyo,
        "growing_season_precipitation_prediction": make_gsp_prediction_wyo,
        "specific_humidity_prediction": make_sh_prediction_wyo,
        "relative_humidity_prediction": make_rh_prediction_wyo,
    }

    return render(request=request, template_name="main_app/prediction.html", context=page_context)
