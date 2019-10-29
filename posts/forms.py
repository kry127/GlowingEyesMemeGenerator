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
            'title': forms.TextInput(attrs={'class': 'input'}),
            'cover': forms.FileInput(attrs={'class': 'my_input'}),
            'spark': forms.FileInput(attrs={'class': 'my_input'})
        }