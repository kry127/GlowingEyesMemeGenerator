from django.urls import path

from .views import HomePageView, CreatePostView, image_processing

urlpatterns = [
    path('', CreatePostView.as_view(), name='add_post'),
    path('post/', HomePageView.as_view(), name='home'),
    path('image_processing/', image_processing, name='process_image'),
]
