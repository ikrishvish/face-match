import tensorflow as tf
import numpy as np
from . import facenet
from .align import detect_face
import cv2
import os
from facematching.settings import BASE_DIR


images_dir = os.path.join(BASE_DIR, 'facematch', 'images')

# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160

thresh = 1.10  # set yourself to meet your requirement

sess = tf.Session()

# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
align_path = os.path.join(BASE_DIR, 'facematch', 'align')
pnet, rnet, onet = detect_face.create_mtcnn(sess, align_path)

# read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
model_path = os.path.join(BASE_DIR, 'facematch', '20170512-110547/20170512-110547.pb')
facenet.load_model(model_path)

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]


def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append(
                    {'face': resized, 'rect': [bb[0], bb[1], bb[2], bb[3]], 'embedding': getEmbedding(prewhitened)})
    return faces


def getEmbedding(resized):
    reshaped = resized.reshape(-1, input_image_size, input_image_size, 3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding


def compare2face(img1, img2):
    face1 = getFace(cv2.imread(img1))
    face2 = getFace(cv2.imread(img2))
    if face1 and face2:
        # calculate Euclidean distance
        dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
        similarity = "%.2f" % (1 / (1 + dist) * 100)
        if dist <= thresh:
            is_match = True
        else:
            is_match = False
        return similarity, is_match
    return False


def generate_embedding_dict_from_dir():
    file_embedding_dict = {}
    for subdir, dirs, files in os.walk(images_dir):
        for file in files:
            full_path = os.path.join(subdir, file)
            if full_path.endswith(('.jpg', '.Jpg', '.jpeg', '.png')):
                img = cv2.imread(full_path)
                face = getFace(img)
                file_embedding_dict[file] = face[0]['embedding']
    return file_embedding_dict


def compare2multiple(input_img):
    embed_dict = generate_embedding_dict_from_dir()
    img = cv2.imread(input_img)
    face = getFace(img)
    input_img_embed = face[0]['embedding']
    # print("input image embed: ", input_img_embed)

    compare_result_dict = {}
    for kf, vf in embed_dict.items():
        dist = np.sqrt(np.sum(np.square(np.subtract(input_img_embed, vf))))
        if dist <= thresh:
            similarity = "%.2f" % (1 / (1 + dist) * 100)
            if float(similarity) < 100:
                compare_result_dict[kf] = str(similarity)
    return compare_result_dict
