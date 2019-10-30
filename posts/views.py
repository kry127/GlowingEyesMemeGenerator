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
        ctx['sparkle_images'] = [f"eye_{i}.png" for i in range(1, 11)]
        return ctx

    def form_valid(self, form):
        # check field validity
        response = super(CreatePostView, self).form_valid(form)

        # get static files
        eyeglow_path = finders.find('images/eye_1.png')
        faceglow_path = finders.find('images/eye_10.png')
        eye_cascade_path = finders.find('xml/haarcascade_eye.xml')
        glasses_cascade_path = finders.find('xml/haarcascade_eye_tree_eyeglasses.xml')
        face_cascade_path = finders.find('xml/haarcascade_frontalface_default.xml')
        font_path = finders.find('fonts/road_sign.otf')

        if "sparkle_type" in self.request.POST:
            eyeglow_path = os.path.join(os.path.dirname(eyeglow_path), self.request.POST["sparkle_type"])


        # do something with self.object
        file_id = self.object.id

        # parse background file
        file_path = self.object.cover.path
        file_dir = os.path.dirname(file_path)
        newfile_name = f"tmpimg{file_id}.png"
        newfile_path = os.path.join(file_dir, newfile_name)

        # get sparkle file
        # if hasattr(self.object, "spark")
        try:
            eyeglow_path = self.object.spark.path
        except ValueError:
            pass


        sparkle_hue = int(self.request.POST.get('sparkle_hue', 0))
        resize_ratio = float(self.request.POST.get('resize_ratio', 1.0))
        substitude_eyes, substitude_face = False, False
        substitude_type = self.request.POST.get('substitude_type', 'eyes')
        if (substitude_type == 'eyes'):
            substitude_eyes = True
        elif (substitude_type == 'face'):
            substitude_face = True

        def get_bool_field(name):
            ret = self.request.POST.get(name, False)
            if ret:
                ret = True
            return ret

        resize_to_box = get_bool_field('resize_to_box')
        randomize_color = get_bool_field('randomize_color')
        make_collage_flag = get_bool_field('make_collage')

        meme_text = self.object.title

        settings = {"eyeglow_path": eyeglow_path,
                     "faceglow_path": eyeglow_path,
                     "eye_cascade": eye_cascade_path,
                     "glasses_cascade_path": glasses_cascade_path,
                     "face_cascade": face_cascade_path,
                     "substitude_eyes": substitude_eyes,
                     "substitude_face": substitude_face,
                     "sparkle_color": sparkle_hue,
                     "randomize_color": randomize_color,
                     "resize_to_box": resize_to_box,
                     "resize_ratio": resize_ratio,
                     "meme_text": meme_text,
                     "meme_font": font_path,}

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