import numpy as np
import os


def load_data(file_dir, file_name):
    file_path = os.path.join(file_dir, file_name)
    data = np.zeros((1, 1024))
    with open(file_path, 'r') as f:
        for i in range(32):
            line = f.readline()
            for j in range(32):
                data[0, i*32+j] = int(line[j])
    label = int(file_name.split('_')[0])
    return data, label


def handwrite():
    train_dir = 'trainingDigits'
    train_filename = os.listdir(train_dir)  # return a list, which include all file name under the train_dir
    train_filenum = len(train_filename)
    train_data = np.zeros((train_filenum, 1024))
    train_label = []
    # read train data and label
    for i in range(train_filenum):
        train_data[i, :], label = load_data(train_dir, train_filename[i])
        train_label.append(label)

    test_dir = 'testDigits'
    test_filename = os.listdir(test_dir)
    test_filenum = len(test_filename)
    a_test_data = np.zeros((1, 1024))
    real = 0
    for i in range(test_filenum):
        a_test_data, test_label = load_data(test_dir, test_filename[i])
        diff_data = np.tile(a_test_data, (train_filenum, 1)) - train_data
        diff_data_square = diff_data**2
        temp = diff_data_square.sum(axis=1)
        distance = temp**0.5
        sort = distance.argsort()
        predict = {}
        for j in range(10):
            predict[j] = 0
        for j in range(3):          # k of knn is 3
            predict[train_label[sort[j]]] += 1
        pre_result = max(predict, key=predict.get)
        print("the classifier came back with: %d, the real answer is: %d" % (pre_result, test_label))
        if pre_result == test_label:
            real += 1
    print(real/test_filenum)


if __name__ == '__main__':
    handwrite()