from __future__ import absolute_import

import os
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)
from random import shuffle
import time
from torch.utils.data import DataLoader

from clustercontrast.evaluators import extract_features
from clustercontrast.utils.data import Preprocessor


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        super().__init__(data_source)
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, i_pid, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if not select_indexes:
                    continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])

        return iter(ret)


class RandomMultipleGallerySamplerNoCam(Sampler):
    def __init__(self, data_source, num_instances=4):
        super().__init__(data_source)

        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])
            _, i_pid, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            index = self.pid_index[pid_i]

            select_indexes = No_index(index, i)
            if not select_indexes:
                continue
            if len(select_indexes) >= self.num_instances:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
            else:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

            for kk in ind_indexes:
                ret.append(index[kk])

        return iter(ret)

class GraphSampler(Sampler):
    def __init__(self, data_source, img_path, transformer, model,  batch_size=64, num_instance=4,
                 gal_batch_size=256, prob_batch_size=256, save_path=None, verbose=False,mode =None):
        super(GraphSampler, self).__init__(data_source)
        self.data_source = data_source
        self.img_path = img_path
        self.transformer = transformer
        self.model = model
        self.mode = mode
        self.batch_size = batch_size
        self.num_instance = num_instance
        self.gal_batch_size = gal_batch_size
        self.prob_batch_size = prob_batch_size
        self.save_path = save_path
        self.verbose = verbose

        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_pids = len(self.pids)
        for pid in self.pids:
            shuffle(self.index_dic[pid])

        self.sam_index = None
        self.sam_pointer = [0] * self.num_pids

    def make_index(self):
        start = time.time()
        self.graph_index()
        if self.verbose:
            print('\nTotal GS time: %.3f seconds.\n' % (time.time() - start))

    def calc_distance(self, dataset):
        data_loader = DataLoader(
            Preprocessor(dataset, self.img_path, transform=self.transformer),
            batch_size=32, num_workers=0,
            shuffle=False, pin_memory=True)

        if self.verbose:
            print('\t GraphSampler: ', end='\t')
        model = copy.deepcopy(self.model).cuda().eval()

        # features = extract_features(model, data_loader, num_features, fea_height, fea_width, self.verbose)
        features, _= extract_features(model, data_loader, print_freq=50, mode=self.mode)
        # features_g_rgb = torch.stack(features_g_rgb)
        # features_p_rgb = torch.stack(features_p_rgb)
        # features_p_rgb = features_p_rgb.mean(2)
        if self.verbose:
            print('\t GraphSampler: \tCompute distance...', end='\t')
        start = time.time()
        features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(data_loader.dataset.dataset)], 0)

        # dist = pairwise_distance(matcher, features, features, self.gal_batch_size, self.prob_batch_size, self.verbose)
        dist = np.matmul(features, np.transpose(features))
        if self.verbose:
            print('Time: %.3f seconds.' % (time.time() - start))

        return dist

    def graph_index(self):
        sam_index = []
        for pid in self.pids:
            index = np.random.choice(self.index_dic[pid], size=1)[0]
            sam_index.append(index)

        dataset = [self.data_source[i] for i in sam_index]
        dist = self.calc_distance(dataset)

        with torch.no_grad():
            dist = torch.tensor(dist) + torch.eye(self.num_pids) * 1e15
            topk = self.batch_size // self.num_instance - 1
            _, topk_index = torch.topk(dist.cuda(), topk, largest=False)
            topk_index = topk_index.cpu().numpy()

        if self.save_path is not None:
            filenames = [fname for fname, _, _, _ in dataset]
            test_file = os.path.join(self.save_path, 'gs%d.npz' % self.epoch)
            np.savez_compressed(test_file, filenames=filenames, dist=dist.cpu().numpy(), topk_index=topk_index)

        sam_index = []
        for i in range(self.num_pids):
            id_index = topk_index[i, :].tolist()
            id_index.append(i)
            index = []
            for j in id_index:
                pid = self.pids[j]
                img_index = self.index_dic[pid]
                len_p = len(img_index)
                index_p = []
                remain = self.num_instance
                while remain > 0:
                    end = self.sam_pointer[j] + remain
                    idx = img_index[self.sam_pointer[j]: end]
                    index_p.extend(idx)
                    remain -= len(idx)
                    self.sam_pointer[j] = end
                    if end >= len_p:
                        shuffle(img_index)
                        self.sam_pointer[j] = 0
                assert (len(index_p) == self.num_instance)
                index.extend(index_p)
            sam_index.extend(index)

        sam_index = np.array(sam_index)
        sam_index = sam_index.reshape((-1, self.batch_size))
        np.random.shuffle(sam_index)
        sam_index = list(sam_index.flatten())
        self.sam_index = sam_index

    def __len__(self):
        if self.sam_index is None:
            return self.num_pids
        else:
            return len(self.sam_index)

    def __iter__(self):
        self.make_index()
        return iter(self.sam_index)