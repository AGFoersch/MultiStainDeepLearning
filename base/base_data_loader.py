from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):
    def __init__(self, tfms, batch_size=None, shuffle:bool=True, num_workers:int=1):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.tfms = tfms
        self.valid_tfms = self.tfms.base

    def _generate_dataloader(self, **kwargs):
        super().__init__(num_workers=self.num_workers, **kwargs)

    def split_validation(self):
        raise NotImplementedError

    def _create_class_dict(self):
        raise NotImplementedError