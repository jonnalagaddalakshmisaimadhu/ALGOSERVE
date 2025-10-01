from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload, name='upload'),
    path('algorithm-selection/', views.algorithm_selection, name='algorithm_selection'),
    path('solve/<str:problem>/', views.solve, name='solve'),
    path('results/', views.results, name='results'),
]
