import json
import math

import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
import model.metric as module_metrics
from functools import partial, update_wrapper

from utils import plot
from torch import nn

from torch.utils.data._utils.collate import *


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def ifnone(a, b):
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


def evaluate(class_dict, test_df, preds, path:str=None, save_fig:bool=False, patient_based=True):
    # ---- tile-based ----
    keys = ['Patient_ID', 'Label', 'Prediction', 'Probability']
    values = [
        test_df.Patient_ID,
        test_df.Label.map(lambda x: class_dict.get(x)),
        np.argmax(preds, axis=1),
        [np.array(x) for x in preds]
    ]

    path_obj = Path(path) if path is not None else Path('.')
    tile_df = pd.DataFrame(dict(zip(keys, values)))

    # save stuff for outside plotting
    tile_df.to_csv(path_obj/"tile_df.csv", index=False)
    write_json(class_dict, path_obj/"class_mapping.json")

    plot.plot(class_dict, tile_df, path, '_tile', save_fig)

    tile_df_for_return = tile_df
    if patient_based:
        # ---- patient-based ----
        tmp_df = tile_df.groupby('Patient_ID')
        df_patients = {x: tmp_df.get_group(x).reset_index(drop=True) for x in tmp_df.groups.keys()}

        patient_ids = []
        labels = []
        majority = []
        probability = []
        prediction = []

        for k,v in df_patients.items():
            assert len(v.Patient_ID.unique()) == 1
            patient_ids.append(v.Patient_ID[0])
            assert len(v.Label.unique()) == 1
            labels.append(v.Label[0])

            majority.append(v.Prediction.value_counts().idxmax())
            probability.append(v.Probability.sum() / len(v))
            prediction.append(np.argmax(probability[-1]))

        d = {
            'Patient_ID': patient_ids,
            'Label': labels,
            'Majority_Decision':majority,
            'Probability': probability,
            'Prediction': prediction
        }
        patient_df = pd.DataFrame(d)
        patient_df.to_csv(path_obj/"patient_df.csv", index=False)
        plot.plot(class_dict, patient_df, path, '_patient', save_fig)

    return tile_df_for_return


class MetricTracker:
    def __init__(self, *keys, writer = None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def add_metric(self, *keys):
        self._data = self._data.append(pd.DataFrame(data=0, index=keys, columns=self._data.columns))

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1, **kwargs):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def update_tensorboard(self, key, **kwargs):
        if self.writer:
            self.writer.add_scalar(key, self.avg(key), **kwargs)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class EpochMetricTracker(MetricTracker):
    def __init__(self, *keys, writer=None):
        super().__init__(*keys, writer)

    def update(self, key, value, n=1, **kwargs):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def epoch_update(self, key, **kwargs):
        if self.writer:
            self.writer.add_scalar(key, self.avg(key), **kwargs)


class ConfusionTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=('TP', 'TN', 'FP', 'FN') + keys, columns=['total'])
        self.metric_fnts = []
        for key in keys:
            ftn = partial(getattr(module_metrics, key))
            update_wrapper(ftn, getattr(module_metrics, key))
            self.metric_fnts.append(ftn)
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, output, target):
        with torch.no_grad():
            pred = torch.argmax(output, dim=1).detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            assert pred.shape[0] == len(target)
            neq = np.not_equal(pred, target)
            eq = np.equal(pred, target)
            tp = np.logical_and (eq, pred == 1).sum()
            tn = np.logical_and (eq, pred == 0).sum()
            fp = np.logical_and(neq, pred == 1).sum()
            fn = np.logical_and(neq, pred == 0).sum()
            if self.writer is not None:
                self._data.total['TP'] += tp
                self.writer.add_scalar('TP', self._data.total['TP'])
                self._data.total['TN'] += tn
                self.writer.add_scalar('TN', self._data.total['TN'])
                self._data.total['FP'] += fp
                self.writer.add_scalar('FP', self._data.total['FP'])
                self._data.total['FN'] += fn
                self.writer.add_scalar('FN', self._data.total['FN'])

    def result(self):
        for met in self.metric_fnts:
            self._data.total[met.__name__] = met(self._data.total['TP'], self._data.total['TN'],
                                                 self._data.total['FP'], self._data.total['FN'])
            self.writer.add_scalar(met.__name__, self._data.total[met.__name__])

        return dict(self._data.total)


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)
