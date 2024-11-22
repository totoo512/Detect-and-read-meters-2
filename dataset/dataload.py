import copy
import cv2
import torch
import numpy as np
from PIL import Image


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class TextInstance(object):
    """
    TextInstance的label属性保存的不是具体的分类，具体分类信息保存在text属性中。

    points: 遮罩对应的几何图形顶点的集合
    orient: 方向
    text: 对应的分类标签
    label: ?
    bottoms:
    e1:
    e2:
    """

    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.bottoms = None
        self.e1 = None
        self.e2 = None

        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        self.points = np.array(points)

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(object):

    def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training

    @staticmethod
    def make_text_region(polygon, mask):
        """
            根据 polygon 和 mask 绘制对应的遮罩区域
        """
        cv2.fillPoly(mask, [polygon.points.astype(np.int32)], 1)  # make text_region

        return mask

    def get_training_data(self, image, polygons, transcripts, image_id, image_path):
        """
            获取训练数据

            image: 图像矩阵
            polygons: TextInstance list
            transcripts: 标签 list
        """
        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))

        pointer_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        dail_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        text_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)

        if polygons is not None:
            for i, polygon in enumerate(polygons):
                if polygon.text == '1':
                    pointer_mask = self.make_text_region(polygon, pointer_mask)

                if polygon.text == '2':
                    dail_mask = self.make_text_region(polygon, dail_mask)

                if polygon.text == 'number':
                    text_mask = self.make_text_region(polygon, text_mask)
                    bboxs = polygon.points.reshape((1, 8))

        train_mask = np.ones(image.shape[:2], np.uint8)
        # # to pytorch channel sequence
        # HWC => CHW
        image = image.transpose(2, 0, 1)

        if not self.is_training:  # test condition

            meta = {
                'image_id': image_id,
                'Height': H,
                'Width': W,
                'trans': transcripts
            }

            return image, pointer_mask, dail_mask, text_mask, train_mask, meta

        # image = torch.from_numpy(image).float()
        # pointer_mask = torch.from_numpy(pointer_mask).long()
        # train_mask = torch.from_numpy(train_mask).long()
        # dail_mask = torch.from_numpy(dail_mask).long()
        # text_mask=torch.from_numpy(text_mask).long()
        # bboxs = torch.from_numpy(bboxs).float()

        # mapping=np.array([0])
        # mapping = torch.from_numpy(mapping).long()
        # transcripts=np.array(transcripts)

        return image, pointer_mask, dail_mask, text_mask, train_mask, bboxs, transcripts

    def __len__(self):
        raise NotImplementedError()
