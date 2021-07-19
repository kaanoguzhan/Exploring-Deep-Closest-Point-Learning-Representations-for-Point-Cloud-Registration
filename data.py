#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import trimesh


# Part of the code is referred from: https://github.com/charlesq34/pointnet

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data_modelnet(partition):
    # download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(*[BASE_DIR, 'data', 'modelnet40'])
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


def load_data_mixamo(partition, num_points, different_sampling):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(*[BASE_DIR, 'data', 'mixamo', 'objfiles'])
    #input = np.load(os.path.join(DATA_DIR, 'abla_binary.npy'))
    #data = np.repeat(input[:, 0][None, :, :], 32, axis=0)
    #color = np.repeat(input[:, 1][None, :, :], 32, axis=0)
    data, color = [], []

    # [(x,3),(x,3)]
    if partition == "train":
        npy_files = glob.glob(DATA_DIR + "/*.npy")
    elif partition == "test":
        test_size = 5
        npy_files = glob.glob(DATA_DIR + "/*.npy")
        npy_files = np.array(npy_files)
        rng_test = np.arange(len(npy_files))
        npy_files = npy_files[rng_test[:test_size]]

    for file in npy_files:
        tmp = np.load(file)
        rng = np.arange(len(tmp[:, 0]))
        if not different_sampling:
            np.random.shuffle(rng)
            data.append(tmp[rng[:num_points], 0])
            color.append(tmp[rng[:num_points], 1])
        else:
            # Return all points
            data.append(tmp[:, 0])
            color.append(tmp[:, 1])

    data = np.array(data)
    color = np.array(color)

    # print(data.shape)
    # print(color.shape)
    #data = np.concatenate(data,axis=0)
    #color = np.concatenate(color,axis=0)

    return data, None, color

def load_data_tumrgbd(partition, num_points, different_sampling):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(*[BASE_DIR, 'data', 'tumrgbd'])
    #input = np.load(os.path.join(DATA_DIR, 'abla_binary.npy'))
    #data = np.repeat(input[:, 0][None, :, :], 32, axis=0)
    #color = np.repeat(input[:, 1][None, :, :], 32, axis=0)
    data, color = [], []

    # [(x,3),(x,3)]
    partition= "test"
    if partition == "train":
        ply_files = glob.glob(DATA_DIR + "/*.ply")
    elif partition == "test":
        test_size = 5
        ply_files = glob.glob(DATA_DIR + "/*.ply")
        ply_files = np.array(ply_files)
        #shuffle here
        rng_test = np.arange(len(ply_files))
        ply_files = ply_files[rng_test[:test_size]]

    for file in ply_files:


        tmp = trimesh.load_mesh(file)
        tmp = np.stack([tmp.vertices,tmp.visual.vertex_colors[:,:3]],axis=1)

        rng = np.arange(len(tmp[:, 0]))
        if not different_sampling:
            np.random.shuffle(rng)
            data.append(tmp[rng[:num_points], 0])
            color.append(tmp[rng[:num_points], 1])
        else:
            # Return all points
            data.append(tmp[:, 0])
            color.append(tmp[:, 1])

    data = np.array(data)
    color = np.array(color)

    # print(data.shape)
    # print(color.shape)
    #data = np.concatenate(data,axis=0)
    #color = np.concatenate(color,axis=0)

    return data, None, color


def load_data(partition, different_sampling, dataset='modelnet40', num_points=1024):
    assert dataset in ['modelnet40', 'mixamo','tumrgbd']
    if dataset == 'modelnet40':
        return load_data_modelnet(partition)
    elif dataset == 'mixamo':
        return load_data_mixamo(partition, num_points, different_sampling)
    elif dataset == 'tumrgbd':
        return load_data_tumrgbd(partition, num_points, different_sampling)




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
                 dataset='modelnet40', use_color=False, different_sampling=False):

        if dataset == 'modelnet40' and use_color:
            raise Exception('ModelNet40 does not support color. Please set use_color to false.')
        self.different_sampling = different_sampling
        self.data, self.label, self.color = load_data(partition=partition, different_sampling=different_sampling,
                                                      dataset=dataset, num_points=num_points)
        self.num_points = num_points  # TODO: Subsample points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        if unseen:
            self.label = self.label.squeeze()
        self.factor = factor
        self.use_color = use_color
        if self.unseen:
            # simulate testing on first 20 categories while training on last 20 categories
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

    def __getitem__(self, index):
        pointcloud = self.data[index]
        permutation = np.random.permutation(len(pointcloud))
        pointcloud = pointcloud[permutation[:self.num_points]]
        if self.use_color:
            color = self.color[index][permutation[:self.num_points]]

        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)

        # Fixed random seed for "Validation" and "Test" sets
        old_random_state = np.random.get_state()  # Store current random state
        if self.partition != 'valid':
            np.random.seed(index)
        if self.partition != 'test':
            np.random.seed(10000000 + index)

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

        permutation1 = np.random.permutation(len(pointcloud1.T))[:self.num_points]
        permutation2 = np.random.permutation(len(pointcloud2.T))[:self.num_points]

        pointcloud1 = pointcloud1[:, permutation1]
        pointcloud2 = pointcloud2[:, permutation2]

        if self.use_color:
            color1 = color[permutation1].T
            color2 = color[permutation2].T
        else:
            color1, color2 = np.empty(0), np.empty(0)

        # Restore stored random seed
        if self.partition in ['valid', 'test']:
            np.random.set_state(old_random_state)

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
            translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
            euler_ab.astype('float32'), euler_ba.astype('float32'), color1.astype('float32'), \
            color2.astype('float32')

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = CustomDataset(1024, 'train')
    test = CustomDataset(1024, 'test')
    for data in train:
        print(len(data))
        break
