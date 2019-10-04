import os
import json
import shutil
import random
import argparse

random.seed(42)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help = 'path to config file', required = True)
    args = vars(ap.parse_args())

    f = open(args['config'], 'r')
    json_data = json.load(f)
    f.close()

    try :
        os.mkdir(json_data['data']['val_data'])
    except:
        if len(os.listdir(json_data['data']['val_data'])) == 0:
            pass
        else:
            print (json_data['data']['val_data'], 'already exists and is not empty')
            quit()

    gt_data_path = os.listdir(json_data['data']['train_data'])
    move_it = random.sample(gt_data_path, int(len(gt_data_path) *
                            json_data['train']['train_val_split']))

    for file_path in move_it:
        shutil.move(os.path.join(json_data['data']['train_data'], file_path),
                    os.path.join(json_data['data']['val_data'], file_path))
