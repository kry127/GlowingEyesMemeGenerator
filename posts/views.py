# posts/views.py
import os

from django.views.decorators.csrf import csrf_exempt
from django.views.generic import ListView, CreateView # new
from django.urls import reverse_lazy # new
from django.contrib.staticfiles import finders

from opencv.image_converter import convert_image

from .forms import PostForm # new
from .models import Post


class HomePageView(ListView):
    model = Post
    template_name = 'posts/home.html'


class CreatePostView(CreateView): # new
    model = Post
    form_class = PostForm
    template_name = 'posts/post.html'
    success_url = reverse_lazy('home')

    def form_valid(self, form):
        response = super(CreatePostView, self).form_valid(form)
        # do something with self.object
        print(self.object)
        file_path = self.object.cover.path
        file_id = self.object.id
        file_dir = os.path.dirname(file_path)
        newfile_path = os.path.join(file_dir, f"tmpimg{file_id}.png")

        eyeglow_path = finders.find('images/eye_1.png')
        faceglow_path = finders.find('images/face_1.png')

        convert_image(file_path, newfile_path, options={"eyeglow_path": eyeglow_path, "faceglow_path": faceglow_path})

        # we can render something different here
        return response


@csrf_exempt
def image_processing():
    pass