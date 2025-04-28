from django.db import models

class TweetPrediction(models.Model):
    tweet_id = models.CharField(max_length=50, unique=True)
    prediction = models.CharField(max_length=1)  # '0' for real, '1' for fake
