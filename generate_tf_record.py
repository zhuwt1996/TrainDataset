from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image #此处在Anaconda中需要引入pillow，引入PIL会因为版本冲突报错
from utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# 将分类名称转成ID号
def class_text_to_int(row_label):

    if row_label == 'person':
        return 1
    if row_label == 'bicycle':
        return 2
    if row_label == 'car':
        return 3
    if row_label == 'bus':
        return 4
    if row_label == 'umbrella':
        return 5
    if row_label == 'suitcase':
        return 6
    if row_label == 'bottle':
        return 7
    if row_label == 'cup':
        return 8
    if row_label == 'chair':
        return 9
    if row_label == 'bed':
        return 10
    if row_label == 'dining table':
        return 11
    if row_label == 'toilet':
        return 12
    if row_label == 'tv':
        return 13
    if row_label == 'laptop':
        return 14
    if row_label == 'mouse':
        return 15
    if row_label == 'remote':
        return 16
    if row_label == 'keyboard':
        return 17
    if row_label == 'cell phone':
        return 18
    if row_label == 'microwave':
        return 19
    if row_label == 'oven':
        return 20
    if row_label == 'sink':
        return 21
    if row_label == 'refrigerator':
        return 22
    if row_label == 'book':
        return 23
    if row_label == 'clock':
        return 24

    else:
        print('NONE: ' + row_label)
        None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def generate_tf_example(group, path):
    print(os.path.join(path, '{}'.format(group.filename)))
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = (group.filename + '.jpg').encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)

        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(csv_input, output_path, imgPath):
    writer = tf.python_io.TFRecordWriter(output_path)
    # 通过读取csv文件得到多个group
    groups = split(pd.read_csv(csv_input), 'filename')
    for group in groups:
        #根据图片路径和group得到example
        example = generate_tf_example(group, imgPath)
        # 将example序列化以后写入文件，所有example写入文件以后生成tf_record文件
        writer.write(example.SerializeToString())
    writer.close()
    print('成功创建tf_record文件: {}'.format(output_path))


if __name__ == '__main__':
    # 图片集的路径
    imgPath = r'E:\datas\datas\img'
    # 训练文件输出路径和训练集csv文件路径
    output_path = 'data/train.record'
    csv_input = 'data/train.csv'
    main(csv_input, output_path, imgPath)
    # 验证文件输出路径和验证集csv文件路径
    output_path = 'data/eval.record'
    csv_input = 'data/eval.csv'
    main(csv_input, output_path, imgPath)
