from django.shortcuts import render


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
