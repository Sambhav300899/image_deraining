from generator import *
import cv2
import numpy as np

if __name__ == "__main__":
    gen = generator()

    while True:
        data = next(gen)
        data = np.array(data)
        #cv2.imshow('rain', data[0][0])
        #cv2.imshow('non_rain', data[1][0])
        #cv2.waitKey(0)
