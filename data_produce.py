import os
import random
import pandas as pd
from PIL import Image
from random import sample, choices

length = 150
height = 40

def image_concat(x, data, len):
    im_ans = Image.new('RGB', (length, height))
    string = ''
    size = length / x
    for i in range(x):
        ran = random.randint(0, len - 1)
        angle = random.randint(0, 15)
        im = Image.open(data.loc[ran, 'path'])
        im.rotate(angle)
        im.resize((int(size), height))
        string += str(data.loc[ran, 'tag'])
        im_ans.paste(im, (i * int(size), 0))

    return im_ans, string

if __name__ == '__main__':
    # train_file = open('../../dataset/MNIST/rawtrain.txt')
    # train_lines = train_file.readlines()
    # pd_train = pd.DataFrame(columns=['path', 'tag'])
    # sum_train = 0
    # for i in train_lines:
    #     j = i.split('\n')[0]
    #     ls = j.split(' ')
    #     pd_train.loc[sum_train, 'path'] = ls[0]
    #     l1 = ls[1].split('(')[1]
    #     l2 = l1.split(')')[0]
    #     pd_train.loc[sum_train, 'tag'] = l2
    #     sum_train += 1
    # pd_train.to_csv("../../dataset/MNIST/rawtrain.csv", index=0)

    # test_file = open('../../dataset/MNIST/rawtrain.txt')
    # test_lines = test_file.readlines()
    # pd_test = pd.DataFrame(columns=['path', 'tag'])
    # sum_test = 0
    # for i in test_lines:
    #     j = i.split('\n')[0]
    #     ls = j.split(' ')
    #     pd_test.loc[sum_test, 'path'] = ls[0]
    #     l1 = ls[1].split('(')[1]
    #     l2 = l1.split(')')[0]
    #     pd_test.loc[sum_test, 'tag'] = l2
    #     sum_test += 1
    # pd_test.to_csv("../../dataset/MNIST/rawtest.csv", index=0)

    pd_train = pd.read_csv("../../dataset/MNIST/rawtrain.csv")
    pd_test = pd.read_csv("../../dataset/MNIST/rawtest.csv")
    pd_trainset = pd.DataFrame(columns=['path', 'tag'])
    pd_testset = pd.DataFrame(columns=['path', 'tag'])

    for i in range(15000):
        a = random.randint(3, 6)
        image, string = image_concat(a, pd_train, 60000)
        image.save('../../dataset/MNIST/mul-digits/train/{}_{}.jpg'.format(string, i))

        pd_trainset.loc[i, 'path'] = '../../dataset/MNIST/mul-digits/train/{}_{}.jpg'.format(string, i)
        pd_trainset.loc[i, 'tag'] = string
    pd_trainset.to_csv("../../dataset/MNIST/trainset.csv", index=0)

    for i in range(2500):
        a = random.randint(3, 6)
        image, string = image_concat(a, pd_test, 10000)
        image.save('../../dataset/MNIST/mul-digits/test/{}_{}.jpg'.format(string, i))
        pd_testset.loc[i, 'path'] = '../../dataset/MNIST/mul-digits/test/{}_{}.jpg'.format(string, i)
        pd_testset.loc[i, 'tag'] = string
    pd_testset.to_csv("../../dataset/MNIST/testset.csv", index=0)