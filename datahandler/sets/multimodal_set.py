import torch
from torch.utils import data
from PIL import Image
import random


class BasicMixDataset(data.Dataset):
    def __init__(self, patients, df_dicts, class_dict, labels, tfms, mutation_df, data_root):
        super(BasicMixDataset, self).__init__()
        self.patients = patients
        self.df_dicts = df_dicts
        self.class_dict = class_dict
        self.labels = labels
        self.tfms = tfms
        self.mutation_df = mutation_df
        self.data_root = data_root

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        patient = self.patients[item]

        labels = [self.class_dict[self.df_dicts[0][patient][self.labels[0]].unique()[0]]]
        for label in self.labels[1:]:
            labels.append(self.df_dicts[0][patient][label].unique()[0])

        data = []
        for d in self.df_dicts:
            df = d[patient]
            items = random.sample(list(df.index), 1)
            img_list = []

            for i in items:
                try:
                    img = Image.open(self.data_root + df.Path[i])
                except (OSError, ValueError) as err:
                    raise err(f'{df.Path[i]} is damaged....')
                img = img.convert('RGB')
                img_list.append(self.tfms(img))

            data.append(torch.stack(img_list).squeeze())

        if self.mutation_df is not None:
            data.append(torch.Tensor(self.mutation_df.loc[patient]))

        return data, labels


class BasicMixValidDataset(data.Dataset):
    def __init__(self, dataframe, valid_columns, class_dict, labels, tfms, mutation_df, data_root):
        super(BasicMixValidDataset, self).__init__()
        self.mutation_df = mutation_df
        self.dataframe = dataframe
        self.class_dict = class_dict
        self.labels = labels
        self.tfms = tfms
        self.valid_columns = valid_columns
        self.data_root = data_root

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        labels = [self.class_dict.get(self.dataframe[self.labels[0]][item])]
        patient = self.dataframe['Patient_ID'][item]

        for label in self.labels[1:]:
            labels.append(self.dataframe[label][item])

        data = []
        for p in self.valid_columns:
            path = self.data_root + self.dataframe[p][item]
            try:
                img = Image.open(path)
            except (OSError, ValueError) as err:
                raise err(f'{path} is damaged...')
            img = self.tfms(img.convert('RGB'))
            data.append(img)

        if self.mutation_df is not None:
            data.append(torch.Tensor(self.mutation_df.loc[patient]))

        return data, labels