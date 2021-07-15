#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


# Part of the code is referred from: https://github.com/charlesq34/pointnet

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data_modelnet(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label, None


def load_data_mixamo(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    input = np.load(os.path.join(DATA_DIR, 'abla_binary.npy'))
    data = np.repeat(input[:, 0][None, :, :], 32, axis=0)
    color = np.repeat(input[:, 1][None, :, :], 32, axis=0)
    return data, None, color


def load_data(partition, dataset='modelnet40'):
    assert dataset in ['modelnet40', 'mixamo']
    if dataset == 'modelnet40':
        return load_data_modelnet(partition)
    else:
        return load_data_mixamo(partition)


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class CustomDataset(Dataset):
    def __init__(self, num_points, partition='train', gaussian_noise=False, unseen=False, factor=4,
                 dataset='modelnet40', use_color=False):
        if dataset == 'modelnet40' and use_color:
            raise Exception('ModelNet40 does not support color. Please set use_color to false.')
        self.data, self.label, self.color = load_data(partition, dataset)
        self.num_points = num_points  # TODO: Subsample points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        if unseen:
            self.label = self.label.squeeze()
        self.factor = factor
        self.use_color = use_color
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label >= 20]
                self.label = self.label[self.label >= 20]
                if use_color:
                    self.color = self.color[self.label >= 20]
            elif self.partition == 'train':
                self.data = self.data[self.label < 20]
                self.label = self.label[self.label < 20]
                if use_color:
                    self.color = self.color[self.label < 20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        permutation1 = np.random.permutation(len(pointcloud1.T))
        pointcloud1 = pointcloud1.T[permutation1].T

        permutation2 = np.random.permutation(len(pointcloud2.T))
        pointcloud2 = pointcloud2.T[permutation2].T

        if self.use_color:
            color = self.color[item][:self.num_points]
            color1 = color[permutation1].T
            color2 = color[permutation2].T
        else:
            color1, color2 = np.empty(0), np.empty(0)

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32'), color1.astype('float32'), \
               color2.astype('float32')

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = CustomDataset(1024)
    test = CustomDataset(1024, 'test')
    for data in train:
        print(len(data))
        break
