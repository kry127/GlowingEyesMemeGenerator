from django import forms
from .models import Post


class PostForm(forms.ModelForm):

    def __init__(self, *args, **kwargs):
        super(PostForm, self).__init__(*args, **kwargs)
        # setting not required fields
        self.fields['title'].required = False
        self.fields['spark'].required = False

    class Meta:
        model = Post
        fields = ['title', 'cover', 'spark']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'type': 'text', 'placeholder':'Enter a meme text!'}),
            'cover': forms.FileInput(attrs={'type': 'file', 'class': 'custom-file-input', 'id': 'inputGroupFile04','aria-describedby':'inputGroupFileAddon04'}),
            'spark': forms.FileInput(attrs={'type': 'file', 'class': 'custom-file-input','id': 'inputGroupFile04','aria-describedby':'inputGroupFileAddon04'})
        }