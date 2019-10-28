from django.urls import path

from .views import HomePageView, CreatePostView

urlpatterns = [
    path('', CreatePostView.as_view(), name='add_post'),
    path('post/', HomePageView.as_view(), name='home'),
]
