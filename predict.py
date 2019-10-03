import json
import argparse
from generator import *
from model import network
from keras.preprocessing.image import load_img, img_to_array, array_to_img

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--config', required = True, help = "path to config file")
    ap.add_argument('--img', required = False, help = 'path to input image', default = None)
    ap.add_argument('--model', required = False, help = 'path to model', default = None)

    args = vars(ap.parse_args())

    f = open(args['config'], 'r')
    json_data = json.load(f)
    f.close()

    gen = generator(json_data['data']['val_data'], bs = 1)

    if args['model'] == None:
        model = network(model_path = json_data['train']['final_model_name'])
    else:
        model = network(model_path = args['model'])

    if args['img'] == None:
        while True:
            orig = next(gen)[0][0].copy()
            img = model.predict(orig)
            img = array_to_img(img)

    else:
        img = load_img(args['img'], target_size = (256, 256))
        orig = img.copy()
        img = img_to_array(img)
        img = model.predict(img)
        img = array_to_img(img)

        orig.show()
        img.show()
