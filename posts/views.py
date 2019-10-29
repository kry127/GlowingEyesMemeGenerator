# posts/views.py
import os

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import ListView, CreateView # new
from django.urls import reverse_lazy # new
from django.contrib.staticfiles import finders

from opencv.image_converter import convert_image, make_collage

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

    # https://stackoverflow.com/questions/20877455/how-to-use-context-with-class-in-createview-in-django
    def get_context_data(self, **kwargs):
        ctx = super(CreatePostView, self).get_context_data(**kwargs)
        ctx['sparkle_images'] = [f"eye_{i}.png" for i in range(1, 10)]
        return ctx

    def form_valid(self, form):
        response = super(CreatePostView, self).form_valid(form)
        # do something with self.object
        print(self.object)
        file_path = self.object.cover.path
        file_id = self.object.id
        file_dir = os.path.dirname(file_path)
        newfile_name = f"tmpimg{file_id}.png"
        newfile_path = os.path.join(file_dir, newfile_name)

        eyeglow_path = finders.find('images/eye_1.png')
        faceglow_path = finders.find('images/face_1.png')
        eye_cascade_path = finders.find('xml/haarcascade_eye.xml')
        face_cascade_path = finders.find('xml/haarcascade_frontalface_default.xml')
        font_path = finders.find('fonts/road_sign.otf')


        sparkle_hue = int(self.request.POST.get('sparkle_hue', 0))
        randomize_color = self.request.POST.get('randomize_color', False)
        if randomize_color:
            randomize_color = True
        make_collage_flag = self.request.POST.get('make_collage', False)
        if make_collage_flag:
            make_collage_flag = True

        if "sparkle_type" in self.request.POST:
            eyeglow_path = os.path.join(os.path.dirname(eyeglow_path), self.request.POST["sparkle_type"])


        meme_text = self.object.title

        settings = {"eyeglow_path": eyeglow_path,
                     "faceglow_path": faceglow_path,
                     "eye_cascade": eye_cascade_path,
                     "face_cascade": face_cascade_path,
                     "sparkle_color": sparkle_hue,
                     "random_hue": randomize_color,
                     "meme_text": meme_text,
                     "meme_font": font_path}

        if make_collage_flag:
            make_collage(file_path, newfile_path, options=settings)
        else:
            convert_image(file_path, newfile_path, options=settings)

        url_base = os.path.dirname(self.object.cover.url)
        processed_image_url = f"{url_base}/{newfile_name}"

        # we can render something different here
        context={}
        context['processed_image_url'] = processed_image_url

        return render(self.request, 'posts/img_ready.html', context=context)


@csrf_exempt
def image_processing():
    pass