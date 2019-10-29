from django import forms
from .models import Post


class PostForm(forms.ModelForm):

    class Meta:
        model = Post
        fields = ['title', 'cover']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'input'}),
            'cover': forms.FileInput(attrs={'class': 'my_input'})
        }