import glob

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class dataProcess(object):

    def __init__(self, out_rows, out_cols, data_path="data/train/image", label_path="data/train/label",
                 test_path="data/test/image",
                 npy_path="npy_data", img_type="png"):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    # 将data/train/image里面的图片加载进来并数字化存储到npy_data下
    def create_train_data(self):
        i = 0
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)
        # 返回所有匹配的文件路径列表，它只有一个参数pathname，定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径。
        imgs = glob.glob("./data/train/image/*.png")
        # imgs = glob.glob(self.data_path + "/*." + self.img_type)

        #调试，修正
        imgs.sort(key=lambda x: int(x.split('\\')[1][:-4]))
        # print(len(imgs))
        for img in imgs:
            img.replace('\\', '/')
        print(imgs)

        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)  # 创建ndarray对象
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imgname in imgs:
            # print(imgname)
            #这里获取所有图片名称
            midname = imgname[imgname.rindex("\\") + 1:]
            # print(midname)
            img = load_img(self.data_path + "/" + midname, grayscale=True)
            # print(image)
            label = load_img(self.label_path + "/" + midname, grayscale=True)
            img = img_to_array(img)  # 图片转为数组，转换后元素类型是浮点型
            label = img_to_array(label)
            imgdatas[i] = img
            imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)  # 存储到npy_data文件夹路径下
        print(imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def create_test_data(self):
        i = 0
        print('-' * 30)
        print('Creating test images...')
        print('-' * 30)
        imgs = glob.glob(self.test_path + "/*." + self.img_type)
        imgs.sort(key=lambda x: int(x.split('\\')[1][:-4]))
        # print(len(imgs))
        for img in imgs:
            img.replace('\\', '/')
        print(imgs)
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imgname in imgs:
            # midname = imgname[imgname.rindex("/") + 1:]
            midname = imgname[imgname.rindex("\\") + 1:]
            img = load_img(self.test_path + "/" + midname, grayscale=True)
            img = img_to_array(img)
            imgdatas[i] = img
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

    def load_train_data(self):
        print('-' * 30)
        print('load train images...')
        print('-' * 30)
        # train
        imgs_train = np.load(
            self.npy_path + "/imgs_train.npy")  # imgs_train = np.load(self.npy_path+"\\imgs_train.npy")
        # train的label
        imgs_mask_train = np.load(
            self.npy_path + "/imgs_mask_train.npy")  # imgs_mask_train = np.load(self.npy_path+"\\imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')  # 转换为float32
        imgs_mask_train = imgs_mask_train.astype('float32')
        # 二值化操作，0、255
        imgs_train /= 255
        # mean = imgs_train.mean(axis = 0)
        # imgs_train -= mean
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('---------------------------load test images---------------------------')
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        print(self.npy_path + "/imgs_test.npy")
        print(imgs_test)
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test


if __name__ == "__main__":
    mydata = dataProcess(512, 512)
    mydata.create_train_data()
    mydata.create_test_data()
