# posts/views.py
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import ListView, CreateView # new
from django.urls import reverse_lazy # new

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
        file_path = self.object.cover.url
        file_id = self.object.id
        # we can render something different here
        return response


@csrf_exempt
def image_processing():
    pass