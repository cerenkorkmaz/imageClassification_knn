#21995445
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


training_data = []                  #array to hold all of the images
labels = []                         #array to hold label of images
path = 'train/'

for img in os.listdir(path):
    pic = cv2.imread(os.path.join(path, img)) #read the images
    pic = cv2.resize(pic, (32, 32))        #resize
    label = img.split(" ")[0]              #save the first word of image name as label - COVID,NORMAL,Viral
    labels.append(label)
    training_data.append(pic)


# Grayscale
def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray


# Gabor Filter
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get half size
    d = K_size // 2

    # prepare kernel
    gabor = np.zeros((K_size, K_size), dtype=np.float32)

    # each value
    for y in range(K_size):
        for x in range(K_size):
            # distance from center
            px = x - d
            py = y - d

            # degree -> radian
            theta = angle / 180. * np.pi

            # get kernel x
            _x = np.cos(theta) * px + np.sin(theta) * py

            # get kernel y
            _y = -np.sin(theta) * px + np.cos(theta) * py

            # fill kernel
            gabor[y, x] = np.exp(-(_x ** 2 + Gamma ** 2 * _y ** 2) / (2 * Sigma ** 2)) * np.cos(
                2 * np.pi * _x / Lambda + Psi)

    # kernel normalization
    gabor /= np.sum(np.abs(gabor))

    return gabor


# Use Gabor filter to act on the image
def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get shape
    H, W = gray.shape

    # padding
    gray = np.pad(gray, (K_size // 2, K_size // 2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)

    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y: y + K_size, x: x + K_size] * gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


# Use 6 Gabor filters with different angles to perform feature extraction on the image
def Gabor_process(img):
    # get shape
    H, W, _ = img.shape

    # gray scale
    gray = BGR2GRAY(img).astype(np.float32)

    # define angle
    # As = [0, 45, 90, 135]
    As = [0, 30, 60, 90, 120, 150]

    # prepare plot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        _out = Gabor_filtering(gray, K_size=9, Sigma=1, Gamma=1.2, Lambda=1, angle=A)

        # add gabor filtered image
        out += _out

    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out


# Canny Edge dedection
def Canny_edge(img):
    # Canny Edge
    canny_edges = cv2.Canny(img, 50, 100)
    return canny_edges


filtered = []
for i in range(len(training_data)):
    img = Gabor_process(training_data[i])   #apply gabor filter to images
    # img = Canny_edge(img)                 #uncomment to apply canny (gabor gives better res so I commented this)
    filtered.append(img)                    #append filtered pictures into array

knn_acc = []                                #array to hold all of the accuracies when knn function called


def predict_knn(s_labels, k):
    c_count, n_count, v_count = 0, 0, 0
    for i in range(k):                      #to look for first k neighbors
        if s_labels[i] == "COVID":          #find what is the label and increment the related variable
            c_count += 1                    #to see how many occurances every label has
        elif s_labels[i] == "NORMAL":
            n_count += 1
        elif s_labels[i] == "Viral":
            v_count += 1

    arr = [c_count, n_count, v_count]       #append occurances to find which one is bigger
    pred = max(arr)                         #find the biggest label
    if pred == c_count:
        predicted = "COVID"                 #return the biggest label as prediction
    elif pred == n_count:
        predicted = "NORMAL"
    elif pred == v_count:
        predicted = "Viral"
    else:
        predicted = s_labels[0]

    return predicted


def k_nearest_neighbor(train, t_label, test, k_test_label, k):
    p_labels = []                                       #array to hold predicted labels

    for i in range(len(test)):
        distances = []                                  #array to hold euclidean distances
        for j in range(len(train)):
            dist = np.linalg.norm(test[i] - train[j])   #find euclidean distance between every test and train image
            distances.append(dist)
        sorted_labels = [t_label for _, t_label in sorted(zip(distances, t_label))] #sort labels according to increasing distance
        distances.sort()                                #sort distances in increasing order
        p_labels.append(predict_knn(sorted_labels, k))  #predict labels and hold them in an array

    knn_acc.append(calculate_accuracy(p_labels, k_test_label)) #calculate accuracy and hold them in an array


def predict_w_knn(s_labels, k, dist):
    c_sum, n_sum, v_sum = 0, 0, 0
    for i in range(k):                                  #the simplest way to calculate weight is
        if dist[i] != 0:                                #inversing the distance and add them for every label
            if s_labels[i] == "COVID":                  #so for all labels we inverse their distances and hold the sum
                c_sum += 1 / dist[i]
            elif s_labels[i] == "NORMAL":
                n_sum += 1 / dist[i]
            elif s_labels[i] == "Viral":
                v_sum += 1 / dist[i]

    arr = [c_sum, n_sum, v_sum]                         #find the biggest sum
    pred = max(arr)
    if pred == c_sum:                                   #predict and return the label according to biggest sum
        predicted = "COVID"
    elif pred == n_sum:
        predicted = "NORMAL"
    elif pred == v_sum:
        predicted = "Viral"
    else:
        predicted = s_labels[0]

    return predicted


wKnn_acc = []                                               #array to hold w-knn accuracies


def weighted_knn(train, t_label, test, k_test_label, k):
    pw_labels = []                                          #array to hold predicted labels
    for i in range(len(test)):
        distances = []
        for j in range(len(train)):
            dist = np.linalg.norm(test[i] - train[j])       #calculate euclidean distance between every taste and train image
            distances.append(dist)
        sorted_labels = [t_label for _, t_label in sorted(zip(distances, t_label))] #sort the labels according to increasing distances
        distances.sort()                                    #sort the distances according to increasing order
        pw_labels.append(predict_w_knn(sorted_labels, k, distances))  #predict labels and store

    wKnn_acc.append(calculate_accuracy(pw_labels, k_test_label)) #calculate accuracy and store


def calculate_accuracy(predicted_l, test_l):
    correct_prediction, accuracy = 0, 0
    total = 0
    for i in range(len(predicted_l)):                       #for every predicted label
        if predicted_l[i] == test_l[i]:                     #check if they are same with test labels
            correct_prediction += 1                         #if they are same increase the correct prediction value
        total += 1                                          #to see how many predictions was made increase this everytime
    accuracy = 100 * (correct_prediction / total)           #calculate percentage accuracy
    return accuracy


def mean_accuracy(acc):
    mean_acc = 0                                            #to find mean average because we run
    for i in range(len(acc)):                               #many test using kfold
        mean_acc += acc[i]                                  #add every accuracy
    mean_acc /= splits                                      #divide the sum with kfold split value and return
    return mean_acc


def kfold_cross(splits, k):
    kf = KFold(splits, random_state=None, shuffle=True)     #kfold function to apply
    for train, test in kf.split(filtered):
        k_train = []                                        #holds splitted train images
        k_test = []                                         #holds splitted test images
        k_train_label = []                                  #holds labels for splitted train images
        k_test_label = []                                   #holds labels for splitted test images
        for i in test:
            k_test.append(filtered[i])                      #append filtered images
            k_test_label.append(labels[i])
        for j in train:
            k_train.append(filtered[j])
            k_train_label.append(labels[j])

        k_nearest_neighbor(k_train, k_train_label, k_test, k_test_label, k) #predict with knn algorithm
        weighted_knn(k_train, k_train_label, k_test, k_test_label, k)       #predict with w-knn algorithm


k = 3                                                       #k value for nearest neighbors
splits = 5                                                  #kfold split value
kfold_cross(splits, k)                                      #runs the tests
print('K-fold splits:', splits, '\nk=', k)
print('Mean accuracy with knn algorithm is:', mean_accuracy(knn_acc))           #find and print mean accuracy for knn
print('Mean accuracy with weighted knn algorithm is:', mean_accuracy(wKnn_acc)) #find and print mean accuracy for wknn
