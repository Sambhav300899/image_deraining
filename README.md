# Image Deraining
Image de-raining using a U-net like architecture. Image de-raining has been a problem gaining attention recently, this is due to the degredation in performance of models when used in rainy scenes. In this project I try out some of my methods to derain images. 

## Setup
### Download the github repo
```bash
git clone https://github.com/Sambhav300899/image_deraining.git
```

### Dataset
Download the dataset from https://github.com/jinnovation/rainy-image-dataset in your current repository

### Preparing the dataset
```bash
mkdir dataset
mv rainy-image-dataset/ground\ truth dataset/train
mv rainy-image-dataset/rainy\ image dataset/rainy_images
rm -rf rainy-image-dataset/
python3 make_dataset.py --config config.json
```
