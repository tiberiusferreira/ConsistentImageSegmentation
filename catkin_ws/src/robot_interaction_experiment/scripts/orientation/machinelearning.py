from sklearn.linear_model import SGDClassifier
import numpy as np
import os
import random
from collections import Counter
import joblib
import os
import cv2
import pickle
import Queue
import time
import pylab as plot
import tsne
import copy
import math

'''
To be used in conjunction with the imgseg.py script. This script uses Machine Learning, in special the SGDC
classifier in the Sklearn toolkit which implements an stochastic gradient descent model and here a logistic regression
loss function is used.
Machine learning is used in order to guess the good orientation of the cube images.
By default it attempts to load the classifier from disk, but it can be trained using images in the LRN_PATH.
The images for trainning should have always the same orientation.
Histogram of Oriented Gradients is used as image features.'''
hog_list = list()
labels = list()
label = ''
saving_learn = 0
saving_test = 0
saved = 0
color = ''
n_bin = 10  # 4 number of orientations for the HoG
b_size = 64  # 15  block size
b_stride = 32
c_size = 32  # 15  cell size
rotation = -1
failure = 0
total = 0
tst_dsk_percentage = -1
LRN_PATH = 'LRN_IMGS/'
PARTIAL_LRN_PATH = 'PARTIAL_LRN/'
TST_PATH = 'TST_IMGS/'
PARTIAL_TST_PATH = 'PARTIAL_TST/'
HoG_PATH = 'HOG_N_LABELS/HOG_N_LABELS.pickle'
CLF_PATH = 'Classifier/clf.pkl'
# clf = BaggingClassifier(svm.SVC(probability=True), n_estimators=division, max_samples=1.0/30)
# clf = KNeighborsClassifier(n_neighbors=1000)
# clf = svm.SVC(probability=True)
clf = SGDClassifier(loss='log', random_state=10, shuffle=True)


def show_gui():
    window_name = 'Machine Learning'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Save Learn', window_name, 0, 1, save_lrn_callback)
    cv2.createTrackbar('Save Tst', window_name, 0, 1, save_tst_callback)
    cv2.createTrackbar('Tst from disk', window_name, 0, 1, test_from_disk)
    cv2.createTrackbar('Predict Label', window_name, 0, 1, hog_pred)
    cv2.createTrackbar('Save Classifier to Disk', window_name, 0, 1, save_classifier)
    cv2.createTrackbar('Save HoG to Disk', window_name, 0, 1, save_hog)




# just a callback which triggers the actual function to save the images for learning
def save_lrn_callback(value):
    save_imgs_learn(1)

# just a callback which triggers the actual function to save the images for testing
def save_tst_callback(value):
    save_imgs_test(1)

# reads images from the LRN_PATH ending with .png and using the format: cake_1_red1.png -> LABEL_ANYNUMBER_color.png
# and extract their HoGs and learns them
def learn_from_disk(value):
    global label
    global LRN_PATH
    global using_VGA
    i = 0
    for filename in os.listdir(LRN_PATH):
        if not filename.endswith(".png"):
            continue
        if (i % 20) == 0:
            start_time = time.time()
        label = filename.rsplit('_', 2)[0]
        image_read = cv2.imread(LRN_PATH + filename)
        store_hog(image_read)
        if (i % 200) == 0:
            print('Elapsed Time HoGing Image ' + str(i) + ' = ' + str(time.time() - start_time) + '\n')
        i += 1
    learn(1)
    print('Done')

# predicts the label of the image stored in the global variable img_clean_bgr_class and also outputs the probability
# matrix associated with it
def hog_pred(value):
    global img_clean_bgr_class
    well_oriented_img, confiance = rotate_n_resize(img_clean_bgr_class)
    fd = get_img_hog(well_oriented_img)
    global clf
    print(clf.predict(fd))
    print(clf.predict_proba(fd)[0])
    return clf.predict(fd)


# Returns the predicted label of the image
def label_pred(img):
    well_oriented_img, confiance = rotate_n_resize(img)
    fd = get_img_hog(well_oriented_img)
    global clf
    return clf.predict(fd)[0]


# attempts to load the HoG (and labels) from disk in the directory HoG_PATH.
# The hog is then stored in the global variable hog_list
# and the associated labels in the global variable labels
# noinspection PyBroadException
def load_hog(value):
    global hog_list
    global labels
    global HoG_PATH
    try:
        f = open(HoG_PATH)
        hog_tuple = pickle.load(f)
    except:
        print("Problem loading HoG from " + str(HoG_PATH) + " , is the file there?")
        return -1
    hog_list = hog_tuple[0]
    labels = hog_tuple[1]
    print('Loaded')
    database_indexes = range(len(labels))
    random.shuffle(database_indexes)
    temp = [hog_list[i] for i in database_indexes]
    hog_list = temp
    temp = [labels[i] for i in database_indexes]
    labels = temp
    return 1


# Use this function to generate a 2D scatter plot representing the classes based on their HoGs using the TSNE lib
def plot_2d_classes(value):
    global labels
    if 10 > len(labels):
        print ("Labels and HoG dont seem to have been loaded")
        print ("Trying to load them from disk")
        if not load_hog(1) == 1:
            print ("Could not load HoG, quitting")
            return
    nm_elements = int(raw_input('Plot this many elements (up to ' + str(len(labels)) + ') : '))
    new_labels = list()
    classes = np.unique(labels).tolist()
    for labell in labels[:nm_elements]:
        for unique_label in classes:
            if unique_label == labell:
                new_labels.append(classes.index(unique_label))
    y = tsne.tsne(np.array(hog_list[:nm_elements]))
    plot.scatter(y[:, 0], y[:, 1], 20, new_labels)
    plot.show()


# learns the data contained in the hog_list variable and labels variable into the clf classifier
def learn(value):
    global hog_list
    global labels
    global clf
    if 'hog_list' not in globals() or len(hog_list) < 2 or 'labels' not in globals() or len(labels) < 2:
        print ('Attempted to do learning on non-existant database.')
        exit(1)
    print('Learning')
    lrn_start_time = time.time()
    clf.fit(hog_list, labels)
    print('Done Learning')
    print('Elapsed Time Learning = ' + str(time.time() - lrn_start_time) + '\n')


# saves HoG (and labels) stored in variables hog_list and labels at the HoG_PATH
def save_hog(value):
    global hog_list
    global labels
    global HoG_PATH
    hog_tuple = (hog_list, labels)
    print ('Saving...')
    print('labels = ' + str(np.unique(hog_tuple[1])))
    with open(HoG_PATH, 'w') as f:
        pickle.dump(hog_tuple, f)
    print('Done')


# saves the classifier stored in the global variable clf into CLF_PATH
def save_classifier(value):
    global clf
    global CLF_PATH
    try:
        joblib.dump(clf, CLF_PATH)
    except:
        print ('Could not save.')
    print('Done')

# loads the classifier stored at CLF_PATH into the global variable clf
# noinspection PyBroadException
def load_classifier(value):
    global clf
    global CLF_PATH
    try:
        clf = joblib.load(CLF_PATH)
    except:
        print ("Loading failed")
        return -1
    print ("Loaded Classifier")
    return 1

# outputs useful information associated with the hog_list and labels
def hog_info(value):
    global labels
    global hog_list
    print('Current labels = ')
    myset = set(labels)
    print(str(myset))
    print('Current HoG size:')
    print(len(hog_list))
    print("Single HoG size: ")
    print(len(hog_list[0]))
    print (Counter(labels))

# rotates the img into the guessed orientatio, returns the imgs resized to (256,256) and the confidence associated with
# this guess
def rotate_n_resize(img):
    img_rotation, confidence = get_img_rot(img)
    if not img_rotation == 0:
        rows, cols, d = img.shape
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), img_rotation * 90, 1)
        img = cv2.warpAffine(img, m, (cols, rows))
    return cv2.resize(img, (256, 256)), confidence

# receives a list containing the detected images of the cubes in different sequential periods of time
# it then tries to guess if the objects changed too much or if they are more or less stable
def check_stability(objs_history):
    objs_hist_cp = copy.copy(objs_history)
    obj_center_list = list()
    nb_obj = 0
    stable = 1
    nb_time_snapshots = (len(objs_hist_cp))
    # obj_snapshot = list of object in a given time
    for index, obj_snapshot in enumerate(objs_hist_cp):
        if index == 0:
            nb_obj = len(obj_snapshot)
        else:
            # if the number of objects changed from one period of time to the other it is not stable
            if not nb_obj == len(obj_snapshot):
                # print ("Obj number not stable!")
                stable = 0
                return stable
    # sort the objects in each time snapshot by their center coordinates
    for index, obj_snapshot in enumerate(objs_hist_cp):  # for each period of time
        # sort objects by their y position and then by their x position
        sorted_by_y = sorted(obj_snapshot, key=lambda tup: tup[1][1])
        sorted_by_x = sorted(sorted_by_y, key=lambda tup: tup[1][0])
        objs_hist_cp[index] = sorted_by_x
    # get the same object center in the different time periods in a list and then append it to a list
    for obj in range(nb_obj):   # for each obj
        one_obj_center_hist = list()
        for index, obj_snapshot in enumerate(objs_hist_cp):  # for each time
            img_bgr8, center = obj_snapshot[obj]    # get a given obj in time
            one_obj_center_hist.append(center)      # append always the same obj from all time snapshots
        obj_center_list.append(one_obj_center_hist)
    # for each object check if the distance between itself and it's other time snapshots are below a threshold
    for obj in range(nb_obj):
        for one in range(nb_time_snapshots):
            x_one, y_one = obj_center_list[obj][one]
            for other in range(nb_time_snapshots):
                x_other, y_other = obj_center_list[obj][other]
                dist = math.sqrt(abs(x_one**2 - x_other**2) + abs(y_one**2 - y_other**2))
                if dist > 200:
                    stable = 0
    return stable

# initial setup: tries to load a classifier, if fails tries to load a hog_list and labels from disk and train it,
# if fails tries to start in debug mode
def setup():
    global setup_done
    if 'setup_done' not in globals():
        setup_done = False
    if not setup_done:
        show_gui()
        if not load_classifier(1) == 1:
            print ('Failed to load classifier, trying to load pre-calculated HoG')
            if not load_hog(1) == 1:
                print ('Failed to load pre-calculated HoG, trying to learn from images in disk')
                learn_from_disk(1)
            else:
                learn(1)
        print ('Got a working classifier!')
    setup_done = True


def objects_detector(uprightrects_tuples):
    global clf
    global saving_learn
    global saving_test
    global saved
    global using_VGA
    global iterations
    global obj_history
    global img_clean_bgr_class
    using_VGA = 0
    setup()
    if 'iterations' not in globals():
        iterations = 0
    else:
        iterations += 1
    if 'obj_history' not in globals():
        obj_history = list()
    if len(obj_history) < 7:
        obj_history.append(uprightrects_tuples)
        return
    else:
        index_last_obj = iterations % 7
        obj_history[index_last_obj] = uprightrects_tuples
    if check_stability(obj_history) == 0:
        return
    uprightrects_tuples = obj_history[2]
    final_imgs = list()
    for index, curr_tuple in enumerate(uprightrects_tuples):
        img_bgr8, center = curr_tuple
        w, l, d = np.shape(img_bgr8)
        img_clean_bgr_learn = img_bgr8.copy()
        if saving_learn == 1:
            save_imgs_learn(img_clean_bgr_learn)
            return
        if not using_VGA:
            img_bgr8 = img_bgr8[13:w - 5, 13:l - 8]
        else:
            img_bgr8 = img_bgr8[6:w - 2, 6:l - 4]
        img_clean_bgr_class = img_bgr8.copy()
        if saving_test == 1:
            save_imgs_test(img_clean_bgr_class)
            return
        img_clean_bgr_class = cv2.resize(img_clean_bgr_class, (128, 128), interpolation=cv2.INTER_AREA)  # resize image
        final, confiance = rotate_n_resize(img_clean_bgr_class)
        final_imgs.append(final)
    return final_imgs

# guesses the image orientation by using the position with highest confidence value, in order words the rotation which
# closest match a training image
def get_img_rot(img_bgr):
    global scaler
    best_rot = 0
    best_perc = 0
    for i in range(4):
        fd = get_img_hog(img_bgr)
        for percentage in clf.predict_proba(fd)[0]:
            if percentage > best_perc:
                best_perc = percentage
                best_rot = i
        rows, cols, d = img_bgr.shape
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        img_bgr = cv2.warpAffine(img_bgr, m, (cols, rows))
    return best_rot, best_perc

# calculates the img's HoG
def get_img_hog(img_bgr):
    img_bgr = cv2.resize(img_bgr, (128, 128), interpolation=cv2.INTER_AREA)  # resize image
    opencv_hog = cv2.HOGDescriptor((128, 128), (b_size, b_size), (b_stride, b_stride), (c_size, c_size), n_bin)
    h1 = opencv_hog.compute(img_bgr)
    fd = np.reshape(h1, (len(h1),))
    fd = fd.reshape(1, -1)
    return fd


# stores the img's HoG and label to hog_list and labels variables and also many variations of it's HoG
# the image is resized to (128, 128) in order to standardize it
def store_hog(img):
    global hog_list
    global labels
    w, l, d = np.shape(img)
    img_list = list()
    img_list.append((img[:, :]))  # no changes
    for i in range(1, 20, 1):
        img_list.append((img[0:w - i, :]))  # cut right
        img_list.append((img[i:, :]))  # cut left
        img_list.append((img[:, i:]))  # cut up
        img_list.append((img[:, 0:l - i]))  # cut down

        img_list.append((img[:, i:l - i]))  # cut down and up
        img_list.append((img[i:, 0:l - i]))  # cut down and left
        img_list.append((img[:w - i, 0:l - i]))  # cut down and right

        img_list.append((img[i:, i:]))  # cut up and left
        img_list.append((img[:w - i, i:]))  # cut up and right

        img_list.append((img[i:w - i, :]))  # cut left and right

        img_list.append((img[i:, i:l - i]))  # cut up and down and left
        img_list.append((img[:w - i, i:l - i]))  # cut up and down and right
        img_list.append((img[i:w - i, 0:l - i]))  # cut left and right and down
        img_list.append((img[i:w - i, i:]))  # cut left and right and up

        img_list.append((img[i:w - i, i:l - i]))  # cut up and down and left and right
    for imgs in img_list:
        imgs = cv2.resize(imgs, (128, 128), interpolation=cv2.INTER_AREA)  # resize image
        h1 = get_img_hog(imgs)
        h1 = h1.reshape(-1, )
        hog_list.append(h1)
        labels.append(label)


# reads images from the TST_PATH and guesses their orientation, after that prints useful information such as success rate
# and minimum confiance value
def test_from_disk(value):
    print('Testing from disk')
    start_time = time.time()
    global TST_PATH
    global total
    global tst_dsk_percentage
    global failure
    global lowest_conf
    total = 0
    failure = 0
    lowest_conf = 10000
    for filename in os.listdir(TST_PATH):
        if not filename.endswith(".png"):
            continue
        total += 1
        if (total % 20) == 0:
            start_time = time.time()
        correct_rotation = int(filename.rsplit('_', 3)[1])
        image = cv2.imread(TST_PATH + filename)
        image = cv2.resize(image, (128, 128))
        found_rot, confiance = get_img_rot(image)
        if confiance < lowest_conf:
            lowest_conf = confiance
        if not abs(correct_rotation - found_rot) < 0.5:
            failure += 1
            print (confiance)
        if (total % 400) == 0:
            print('Elapsed Time Testing Image ' + str(total) + ' = ' + str(time.time() - start_time) + '\n')
    tst_dsk_percentage = 100 * failure / total
    print('Failure = ' + str(tst_dsk_percentage) + '%')
    print('Failures = ' + str(failure))
    print('Elapsed Time Testing = ' + str(time.time() - start_time) + '\n')
    print ("Lowest conf = " + str(lowest_conf))
    print('Done')


# Function used to try different HoG parameters
def big_test(value):
    global n_bin
    global b_size  # Align to cell size
    global c_size
    global b_stride  # Multiple of cell size 8 8
    global labels
    global hog_list
    global label
    global failure
    failure_tot = 0
    tst_dsk_percentage_tot = 0
    global total
    global tst_dsk_percentage
    global clf
    global LRN_PATH
    global lowest_conf
    iterac = 10
    window = 128
    for bin_ in range(16, 2, -1):
        for i in range(iterac):
            start_time = time.time()
            n_bin = bin_
            labels = list()
            hog_list = list()
            print('Testing HoG')
            print('n_bin = ' + str(n_bin) + '\n')
            print('b_stride = ' + str(b_stride) + '\n')
            print('b_size = ' + str(b_size) + '\n')
            print('c_size = ' + str(c_size) + '\n')
            learn_from_disk(1)
            test_from_disk(1)
            failure_tot += failure
            tst_dsk_percentage_tot += tst_dsk_percentage
        failure_tot /= iterac
        tst_dsk_percentage_tot /= iterac
        failure = failure_tot
        tst_dsk_percentage = tst_dsk_percentage_tot
        tst_dsk_percentage_tot = 0
        with open('HoG_Trials.txt', 'a') as the_file:
            the_file.write('n_bin = ' + str(n_bin) + '\n')
            the_file.write('b_size = ' + str(b_size) + '\n')
            the_file.write('b_stride = ' + str(b_stride) + '\n')
            the_file.write('c_size = ' + str(c_size) + '\n')
            the_file.write('single HoG size = ' + str(len(hog_list[0])) + '\n')
            the_file.write('Failure = ' + str(failure) + '\n')
            the_file.write('Total = ' + str(total) + '\n')
            the_file.write('Percentage = ' + str(tst_dsk_percentage) + '\n')
            the_file.write('Lowest Conf = ' + str(lowest_conf) + '\n')
            the_file.write('Elapsed Time = ' + str(time.time() - start_time) + '\n\n\n')
        print('Written')
        print('Elapsed Time = ' + str(time.time() - start_time) + '\n')
        clf = SGDClassifier(loss='log')
    print('Big Test Done')

# saves images to LRN_PATH using the format cake_1_red1.png -> LABEL_ANYNUMBER_color.png
def save_imgs_learn(value):
    global label
    global color
    global saving_learn
    global LRN_PATH
    global saved
    if isinstance(value, int):
        label = str(raw_input('Label: '))
        color = str(raw_input('Color: '))
        saving_learn = 1
    else:
        # cv2.imwrite returns true if successful
        if not (cv2.imwrite(LRN_PATH + label + '_' + str(saved) + '_' + color + '.png', value)):
            print ('Could not write to ' + str(LRN_PATH) + '.')
            return -1
        saved += 1
        print(saved)
        if saved == 3:
            saving_learn = 0
            saved = 0
            print('Done saving')

# saves images to TST_PATH using the format label_(90*counter clockwise rotation degrees)_(image_number)_color.png
# ex: whale_1_11_lightblue.png
def save_imgs_test(value):
    global label
    global color
    global rotation
    global saving_test
    global TST_PATH
    global saved
    if isinstance(value, int):
        mode = str(raw_input('Label: '))
        label = mode
        color_ = str(raw_input('Color: '))
        color = color_
        rotation = str(raw_input('Rotation: '))
        saving_test = 1
    else:
        # cv2.imwrite return true if successful
        if not (cv2.imwrite(TST_PATH + label + '_' + str(rotation) + '_' + str(saved) + '_' + color + '.png', value)):
            print ('Could not write to ' + str(LRN_PATH) + '.')
            return -1
        saved += 1
        print(saved)
        if saved == 20:
            saving_test = 0
            saved = 0
            print('Done saving')

