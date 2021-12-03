---
toc: false
layout: post
description: A simple and immature gender recognizer by face images.
categories: [classification]
title: Gender recognizer
---

Recently I'm learning a course, [Practical Deep Learning for Coders](https://course.fast.ai/).\
Shout-out to [fast.ai](https://www.fast.ai/), they provides valuable neural nets course, resourses and dev tools.\
This gender revcognizer is a simple practice inspired by the first two lessons that i recently learned. Using [fastai](https://github.com/fastai/fastai) library.

---
At fist, I used [bing image search api collect data](https://www.microsoft.com/en-us/bing/apis/bing-image-search-api). But when i reviewed the data, I found those images were far from my actually needs.\
What i need is human face images, but most results contain many irrelevant elements like cars, trees, comics... Some images have more than one person.\
Technically, i can use some face cropping tools pretreat data, but as a beginner, i try to experience the whole process fist.\
So i found [Gender Classification Dataset](https://www.kaggle.com/cashutosh/gender-classification-dataset) from kaggle, and use it as my training and validation dataset.

![face images]({{ site.baseurl }}/images/face images.png "face images")

### import fastbook and fastai library
```python
!pip install -Uqq fastbook
import fastbook
from fastbook import *
from fastai.vision.widgets import *
```

### Dataloader and data augmentation
```python
genders = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
genders = genders.new(item_tfms=RandomResizedCrop(128, min_scale=0.5), batch_tfms=aug_transforms(mult=2))
dls = genders.dataloaders(path)
dls.valid.show_batch(max_n=8, nrows=2, unique=True)
```

### Train model
```python
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
```

### Test model
![gender predict]({{ site.baseurl }}/images/gender predict.png "gender predict")

### Thoughts
Data collecting is more challenging than i expected, and it can actually affect my training results.\
Biased data could cause ethical issues.
