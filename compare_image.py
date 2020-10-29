# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import re
import scipy.misc
import warnings
import face_recognition.api as face_recognition
import sys
from PIL import Image
import numpy
import imutils
import cv2

TOO_BIG_SIZE = 300

def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []

    basename = known_people_folder 
    img = face_recognition.load_image_file(known_people_folder)

    # # Scale down image if it's giant so things run a little faster
    # print("known image size: ", img.shape)
    # if img.shape[1] > TOO_BIG_SIZE:
    #     print("known image is too big, converting")
    #     new_height = TOO_BIG_SIZE
    #     new_width = int(round((TOO_BIG_SIZE / img.shape[1]) * img.shape[0], 0))
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         #unknown_image = scipy.misc.imresize(unknown_image, scale_factor)
    #         img = numpy.array(Image.fromarray(img).resize((new_height, new_width)))

    encodings = face_recognition.face_encodings(img)
    print("Vo len(encodings): ", len(encodings))
    if len(encodings) == 1:
        known_names.append(basename)
        known_face_encodings.append(encodings[0])   
    elif len(encodings) == 0:
        img_90 = imutils.rotate_bound(img, 90)
        print("Vo image rotated 90 degrees")
        encodings = face_recognition.face_encodings(img_90)
        if len(encodings) == 1:
            # img_90.save(known_people_folder)
            load_saved_image = Image.open(known_people_folder)
            np_img = numpy.asarray(load_saved_image)
            color_image = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            rotated_image = imutils.rotate_bound(color_image, 90)
            # rotated_image.save(known_people_folder)
            cv2.imwrite(known_people_folder,rotated_image)
            print("saved Vo image again")
            known_names.append(basename)
            known_face_encodings.append(encodings[0])   
    return known_names, known_face_encodings


def test_image(image_to_check, known_names, known_face_encodings):
    unknown_image = face_recognition.load_image_file(image_to_check)

    # # Scale down image if it's giant so things run a little faster
    # print("image to check size: ", unknown_image.shape)
    # if unknown_image.shape[1] > TOO_BIG_SIZE:
    #     print("image to check is too big, converting")
    #     new_height = TOO_BIG_SIZE
    #     new_width = int(round((TOO_BIG_SIZE / unknown_image.shape[1]) * unknown_image.shape[0], 0))
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         #unknown_image = scipy.misc.imresize(unknown_image, scale_factor)
    #         unknown_image = numpy.array(Image.fromarray(unknown_image).resize((new_height, new_width)))

    unknown_encodings = face_recognition.face_encodings(unknown_image)

    print("Chong len(unknown_encodings): ", len(unknown_encodings))
    if len(unknown_encodings)==1:
        for unknown_encoding in unknown_encodings:
            #result = face_recognition.compare_faces(known_face_encodings, unknown_encoding)
            #print("True") if True in result else print("False ")

            distance = face_recognition.face_distance(known_face_encodings, unknown_encoding)
            print("distance func", distance)

            if not distance:
                distance = 1
                return distance, "fail_vo"

        return distance, "ok"
    else:
        unknown_image_90  = imutils.rotate_bound(unknown_image, 90)
        print("image Chong rotated 90 degrees")
        unknown_encodings = face_recognition.face_encodings(unknown_image_90)
        
        if len(unknown_encodings)==1:
            load_saved_image = Image.open(image_to_check)
            np_img = numpy.asarray(load_saved_image)
            color_image = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            rotated_image = imutils.rotate_bound(color_image, 90)
            cv2.imwrite(image_to_check,rotated_image)
            print("saved Chong image again")
            for unknown_encoding in unknown_encodings:
                #result = face_recognition.compare_faces(known_face_encodings, unknown_encoding)
                #print("True") if True in result else print("False ")

                distance = face_recognition.face_distance(known_face_encodings, unknown_encoding)
                print("distance func", distance)

                if not distance:

                    distance = 1
                    return distance, "fail_vo"

            return distance, "ok"

        else:
            print("0","Many Faces or No Faces")
            distance = 1
            return distance, "fail_chong"

        



def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def main(known_people_folder, image_to_check):
    known_names, known_face_encodings = scan_known_people(known_people_folder)
    distance=test_image(image_to_check, known_names, known_face_encodings)
    print("distance main: ", distance)

    similarity = (1-distance[0])*100
    perfect_similarity = similarity
    status = distance[1]
    print("similarity", similarity)

    perfect_similarity = similarity + 30
    print("perfect similarity", perfect_similarity)
    if perfect_similarity > 100:
        perfect_similarity = 100

    similarity = int(similarity)
    perfect_similarity = int(perfect_similarity)

    print("final similarity, perfect_similarity: ", similarity, perfect_similarity)
    return similarity, perfect_similarity, status

if __name__ == "__main__":

  main(sys.argv[1],sys.argv[2])

