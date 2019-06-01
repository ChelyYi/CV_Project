from lxml import etree
import numpy as np
import os
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

# parameters that you should set before running this script
filter = ['aeroplane', 'car', 'chair', 'dog', 'bird']       # select class, this default should yield 1489 training and 1470 validation images
voc_root_folder = "../data/"  # please replace with the location on your laptop where you unpacked the tarball
image_size = 256    # image size that you will use for your network (input images will be resampled to this size), lower if you have troubles on your laptop (hint: use io.imshow to inspect the quality of the resampled images before feeding it into your network!)

# for segmentation, transform color image to target
CLASSES = ['background',
           'aeroplane','bicycle','bird','boat','bottle',
           'bus','car','cat','chair','cow',
           'diningtable','dog','horse','motorbike','person',
           'potted plant','sheep','sofa','train','tv/monitor']
# RGB color for each class
COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128,0,128], [0,128,128], [128,128,128], [64,0,0], [192,0,0],
            [64,128,0], [192,128,0], [64,0,128], [192,0,128],
            [64,128,128], [192,128,128], [0,64,0], [128,64,0],
            [0,192,0], [128,192,0], [0,64,128]]
CM2CLASS = np.zeros(256 ** 3) # every pixel 0~255，RGB 3 channel
for i,cm in enumerate(COLORMAP):
    CM2CLASS[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i # build index


# step1 - build list of filtered filenames
annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")
annotation_files = os.listdir(annotation_folder)
filtered_filenames = []
for a_f in annotation_files:
    tree = etree.parse(os.path.join(annotation_folder, a_f))
    if np.any([tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]):
        filtered_filenames.append(a_f[:-4])

def build_classification_dataset(list_of_files):
    """ build training or validation set

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, n_classes)
    """
    temp = []
    train_labels = []
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
            label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
            train_labels.append(len(temp[-1]) * [label_id])
    train_filter = [item for l in temp for item in l]

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if
                       f in file]
    x = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in image_filenames]).astype(
        'float32')
    # changed y to an array of shape (num_examples, num_classes) with 0 if class is not present and 1 if class is present
    y_temp = []
    for tf in train_filter:
        y_temp.append([1 if tf in l else 0 for l in temp])
    y = np.array(y_temp)

    return x, y

def get_classification_dataset():
    # step2 - build (x,y) for TRAIN/VAL (classification)
    classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Main/")
    classes_files = os.listdir(classes_folder)
    train_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if
                   filt in c_f and '_train.txt' in c_f]
    val_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if
                 filt in c_f and '_val.txt' in c_f]

    x_train, y_train = build_classification_dataset(train_files)
    print('%i training images from %i classes' % (x_train.shape[0], y_train.shape[1]))
    print(x_train.shape)
    print(y_train.shape)
    np.save('../data/x_train.npy', x_train)
    np.save('../data/y_train.npy', y_train)

    x_val, y_val = build_classification_dataset(val_files)
    print('%i validation images from %i classes' % (x_val.shape[0], y_val.shape[1]))
    print(x_val.shape)
    print(y_val.shape)
    np.save('../data/x_val.npy', x_val)
    np.save('../data/y_val.npy', y_val)

def build_segmentation_dataset(seg_file):
    """ build training or validation set for segmentation

    :param seg_file: filenames to build segmentation dataset
    :return: tuple with training data x np.ndarray of shape (n_images, image_size, image_size, 3) and
            segmentation result y np.ndarray of shape (n_images, image_size, image_size, 3)
    """
    with open(seg_file) as file:
        lines = file.read().splitlines()
        train_filter = [line.strip() for line in lines]

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if
                       f in file]
    x = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in image_filenames]).astype('float32')

    seg_folder = os.path.join(voc_root_folder, "VOC2009/SegmentationClass/")
    image_filenames = [os.path.join(seg_folder, file) for f in train_filter for file in os.listdir(seg_folder) if
                       f in file]
    y = np.array([get_target(img_f) for img_f in image_filenames])

    return x, y

def get_target(img_path):
    """
     For segmentation task, read color image, transform it to 3-d one-hot encodeing class target image.
    :param img_path: the image path
    :return: target: (image_size,image_size,21)
    """
    label_im = Image.open(img_path).convert('RGB')
    im = label_im.resize((image_size, image_size), Image.ANTIALIAS)

    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    label_map =  np.array(CM2CLASS[idx], dtype='int64') # 2-d class map, class of every pixel

    return label_map

def get_segmentation_dataset():
    train_file = os.path.join(voc_root_folder, "VOC2009/ImageSets/Segmentation/train.txt")
    x_train, y_train = build_segmentation_dataset(train_file)
    print('%i training images number' % (x_train.shape[0]))
    print(x_train.shape)
    print(y_train.shape)
    np.save('../data/seg/x_seg_train_.npy', x_train)
    np.save('../data/seg/y_seg_train_.npy', y_train)

    val_file = os.path.join(voc_root_folder, "VOC2009/ImageSets/Segmentation/val.txt")
    x_val, y_val = build_segmentation_dataset(val_file)
    print('%i validation images number' % (x_val.shape[0]))
    print(x_val.shape)
    print(y_val.shape)
    np.save('../data/seg/x_seg_val_.npy', x_val)
    np.save('../data/seg/y_seg_val_.npy', y_val)

def get_split_segmentation_dataset():
    train_file = os.path.join(voc_root_folder, "VOC2009/ImageSets/Segmentation/trainval.txt")
    x, y = build_segmentation_dataset(train_file)
    x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=10)
    print('%i training images number' % (x_train.shape[0]))
    print(x_train.shape)
    print(y_train.shape)
    np.save('../data/x_seg_train_.npy', x_train)
    np.save('../data/y_seg_train_.npy', y_train)

    print('%i validation images number' % (x_val.shape[0]))
    print(x_val.shape)
    print(y_val.shape)
    np.save('../data/x_seg_val_.npy', x_val)
    np.save('../data/y_seg_val_.npy', y_val)


if __name__ == '__main__':
    get_split_segmentation_dataset()