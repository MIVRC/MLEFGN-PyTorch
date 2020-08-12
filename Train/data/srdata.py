import os
import glob

from data import common
import pickle
import numpy as np
import imageio

import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_edge, list_lr = self._scan()
        if args.ext.find('bin') >= 0:
            # Binary files are stored in 'bin' folder
            # If the binary file exists, load it. If not, make it.
            list_hr, list_edge, list_lr = self._scan()
            self.images_hr = self._check_and_load(
                args.ext, list_hr, self._name_hrbin()
            )
            self.images_edge = self._check_and_load(
                args.ext, list_edge, self._name_edgebin()
            )
            self.images_lr = [
                self._check_and_load(args.ext, l, self._name_lrbin(s)) \
                for s, l in zip(self.scale, list_lr)
            ]
        else:
            if args.ext.find('img') >= 0 or benchmark:
                self.images_hr, self.images_edge, self.images_lr = list_hr, list_edge, list_lr
            elif args.ext.find('sep') >= 0:
                os.makedirs(
                    self.dir_hr.replace(self.apath, path_bin),
                    exist_ok=True
                )
                os.makedirs(
                    self.dir_edge.replace(self.apath, path_bin),
                    exist_ok=True
                )
                for s in self.scale:
                    os.makedirs(
                        os.path.join(
                            self.dir_lr.replace(self.apath, path_bin),
                            'X{}'.format(s)
                        ),
                        exist_ok=True
                    )
                
                self.images_hr, self.images_edge, self.images_lr = [], [], [[] for _ in self.scale]
                for h in list_hr:
                    b = h.replace(self.apath, path_bin)
                    b = b.replace(self.ext[0], '.pt')
                    self.images_hr.append(b)
                    self._check_and_load(
                        args.ext, [h], b, verbose=True, load=False
                    )

                for e in list_edge:
                    g = e.replace(self.apath, path_bin)
                    g = g.replace(self.ext[0], '.pt')
                    self.images_edge.append(g)
                    self._check_and_load(
                        args.ext, [e], g, verbose=True, load=False
                    )

                for i, ll in enumerate(list_lr):
                    for l in ll:
                        b = l.replace(self.apath, path_bin)
                        b = b.replace(self.ext[1], '.pt')
                        self.images_lr[i].append(b)
                        self._check_and_load(
                            args.ext, [l], b,  verbose=True, load=False
                        )

        if train:
            self.repeat \
                = args.test_every // (len(self.images_hr) // args.batch_size)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_edge = sorted(
            glob.glob(os.path.join(self.dir_edge, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, self.ext[1]
                    )
                ))

        return names_hr, names_edge, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_edge = os.path.join(self.apath, 'EDGE')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png', '.png')

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.pt'.format(self.split)
        )

    def _name_edgebin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_EDGE.pt'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.pt'.format(self.split, scale)
        )

    def _check_and_load(self, ext, l, f, verbose=True, load=True):
        if os.path.isfile(f) and ext.find('reset') < 0:
            if load:
                if verbose: print('Loading {}...'.format(f))
                with open(f, 'rb') as _f: ret = pickle.load(_f)
                return ret
            else:
                return None
        else:
            if verbose:
                if ext.find('reset') >= 0:
                    print('Making a new binary: {}'.format(f))
                else:
                    print('{} does not exist. Now making binary...'.format(f))
            b = [{
                'name': os.path.splitext(os.path.basename(_l))[0],
                'image': imageio.imread(_l)
            } for _l in l]
            with open(f, 'wb') as _f: pickle.dump(b, _f)
            return b

    def __getitem__(self, idx):
        lr, edge, hr, filename = self._load_file(idx)
        lr, edge, hr = self.get_patch(lr, edge, hr)
        lr, edge, hr = common.set_channel(lr, edge, hr, n_channels=self.args.n_colors)
        lr_tensor, edge_tensor, hr_tensor = common.np2Tensor(
            lr, edge, hr, rgb_range=self.args.rgb_range
        )

        return lr_tensor, edge_tensor, hr_tensor, filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_edge = self.images_edge[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        if self.args.ext.find('bin') >= 0:
            filename = f_hr['name']
            hr = f_hr['image']
            edge = f_edge['image']
            lr = f_lr['image']
        else:
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            if self.args.ext == 'img' or self.benchmark:
                hr = imageio.imread(f_hr)
                edge = imageio.imread(f_edge)
                lr = imageio.imread(f_lr)
            elif self.args.ext.find('sep') >= 0:
                with open(f_hr, 'rb') as _f: hr = np.load(_f)[0]['image']
                with open(f_edge, 'rb') as _f: edge = np.load(_f)[0]['image']
                with open(f_lr, 'rb') as _f: lr = np.load(_f)[0]['image']

        return lr, edge, hr, filename

    def get_patch(self, lr, edge, hr):
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, edge, hr = common.get_patch(
                lr,
                edge,
                hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi_scale=multi_scale
            )
            if not self.args.no_augment:
                lr, edge, hr = common.augment(lr, edge, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]
            edge = edge[0:ih * scale, 0:iw * scale]

        return lr, edge, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

