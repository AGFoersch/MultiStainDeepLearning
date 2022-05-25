from base import BaseDataLoader
from datahandler.sets import *
import pandas as pd
from torch.utils.data import DataLoader


class BasicMixDataLoader(BaseDataLoader):
    def __init__(self, dataframes, labels, dataframe_valid=None, valid_columns=None, mutation_df=None,
                 batch_size=1, shuffle=True, num_workers=0, tfms=None, data_root=""):
        super(BasicMixDataLoader, self).__init__(tfms=tfms, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.tfms = tfms
        self.labels = labels
        self.dataframes = []
        self.train_dataframes = []
        self.mutation_df = pd.read_csv(mutation_df).set_index('Patient_ID') if mutation_df is not None else None
        if data_root == "" or data_root.endswith("/"):
            self.data_root = data_root
        else:
            self.data_root = data_root + "/"

        for df_path in dataframes:
            df = pd.read_csv(df_path)
            self.dataframes.append(df)

            tmp_df = df.groupby('Set')
            train_df = tmp_df.get_group('TRAIN').reset_index(drop=True)
            self.train_dataframes.append(train_df)

        self.train_dicts, self.train_patients = self._create_dict(self.train_dataframes)
        self.classes, self.class_dict = self._create_class_dict()
        self.train_dataset = BasicMixDataset(
            self.train_patients, self.train_dicts, self.class_dict, self.labels, self.tfms, self.mutation_df,
            self.data_root
        )

        if dataframe_valid is not None:
            self.valid_dataframe = pd.read_csv(dataframe_valid)
            self.valid_path = valid_columns
            self.valid_dataset =  BasicMixValidDataset(
                self.valid_dataframe, valid_columns, self.class_dict, self.labels, self.valid_tfms, self.mutation_df,
                self.data_root
            )
        else:
            try:
                self.valid_dataframes = []
                for df in self.dataframes:
                    tmp_df = df.groupby('Set')
                    valid_df = tmp_df.get_group('VALID').reset_index(drop=True)
                    self.valid_dataframes.append(valid_df)
                self.valid_dicts, self.valid_patients = self._create_dict(self.valid_dataframes)
                self.valid_dataset = BasicMixDataset(
                    self.valid_patients, self.valid_dicts, self.class_dict, self.labels, self.valid_tfms,
                    self.mutation_df, self.data_root
                )
            except:
                self.valid_dataset = None

        self._generate_dataloader(dataset=self.train_dataset, batch_size=batch_size, shuffle=shuffle)

    def split_validation(self):
        if self.valid_dataset is None: return None
        else: return DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                shuffle=False)

    def _create_dict(self, dataframes):
        patients = dataframes[0].Patient_ID.unique()
        dicts = []

        for dataframe in dataframes:
            tmp_dict = {}
            tmp_df = dataframe.groupby('Patient_ID')
            for patient in patients:
                tmp_dict[patient] = tmp_df.get_group(patient).reset_index(drop=True)
            dicts.append(tmp_dict)

        return dicts, patients

    def _create_class_dict(self):
        class_dict = {}
        classes = []
        if len(self.labels) == 2:
            for c in self.dataframes[0][self.labels[0]].unique():
                class_dict.update({c: int(c.__contains__('1'))})
                classes.append(c)
        else:
            for i,c in enumerate(self.dataframes[0][self.labels[0]].unique()):
                class_dict.update({c : i})
                classes.append(c)

        return classes, class_dict