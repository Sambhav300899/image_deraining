import os
import random
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

def generator(images_path = 'dataset/train', rainy_img_inputs = 'dataset/rainy_images', bs = 16):
    random.seed(42)
    np.random.seed(42)

    img_list = [os.path.join(images_path, name) for name in os.listdir(images_path)]
    rainy_img_list = [os.path.join(rainy_img_inputs, name) for name in os.listdir(rainy_img_inputs)]
    orig_imgs = img_list

    while True:
        X = []
        Y = []

        if len(img_list) < bs:
            img_list = orig_imgs

        gt_imgs = random.sample(img_list, bs)
        img_list = list(set.difference(set(gt_imgs), set(img_list)))

        '''
        generate o/p images according to ground truth, map input from the 14 rainy images in the dataset
        randomly select from the 14 images, p.s - every gt map has 14 corresponding o/p's so hardcode
        '''
        for gt_img in gt_imgs:
            rain_dir_path_and_img = os.path.join(rainy_img_inputs, gt_img.split('/')[-1])
            x_path = rain_dir_path_and_img.split('.')[0] + '_' + str(np.random.randint(1, 15)) + '.' + gt_img.split('.')[-1]

            img_y = load_img(gt_img, target_size = (256, 256))
            img_x = load_img(x_path, target_size = (256, 256))

            img_x = img_to_array(img_x)
            img_y = img_to_array(img_y)

            X.append(img_x)
            Y.append(img_y)

        yield [np.array(X), np.array(Y)]
