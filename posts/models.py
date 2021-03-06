from django.db import models


class Post(models.Model):
    title = models.CharField(max_length=200, blank=True, default='')
    cover = models.ImageField(upload_to='images/')
    spark = models.ImageField(upload_to='images/')

    def __str__(self):
        return self.title


