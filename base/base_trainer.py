import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from tqdm import tqdm
from utils import inf_loop, MetricTracker


class BaseTrainer:
    """
    Base class for all trainers
    """
    metric_ftns: list

    def __init__(self, model, criterion, metric_ftns, optimizer, config,
                 lr_scheduler, data_loader, valid_data_loader):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = 50 # number of steps after which tensorboard is updated
        self.cm_step = 300 # number of steps after which confusion matrix is updated in tensorboard


        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)

        self.criterion = criterion.to(self.device)
        self.metric_ftns_epoch = metric_ftns[0]
        self.metric_ftns_running = metric_ftns[1]
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer.get('save_period', self.epochs)
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.len_epoch = cfg_trainer.get('len_epoch', None)
        self.val_period = cfg_trainer.get('val_period', 1)
        self.start_checkpointing_at = cfg_trainer.get("start_checkpointing_at", 1)
        self.do_evaluation = cfg_trainer.get('evaluation', False)

        self._data_loader = self.data_loader
        self.log = {}

        if self.len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)

        # configuration to monitor model performance and save best
        if not (self.monitor == "off" or len(self.monitor) == 0):
            self.disabled_metric_idx = []
            self.early_stop = cfg_trainer.get('early_stop', inf)
            # this ensures that the old way of passing self.monitor (as a string rather than a list) still works.
            if isinstance(self.monitor, str):
                self.monitor = [self.monitor]
            self.mnt_modes   = [None for i in range(len(self.monitor))]
            self.mnt_metrics = [None for i in range(len(self.monitor))]
            self.mnt_bests   = [None for i in range(len(self.monitor))]
            for i in range(len(self.monitor)):
                self.mnt_modes[i], self.mnt_metrics[i] = self.monitor[i].split()
                assert self.mnt_modes[i] in ["min", "max"]
                self.mnt_bests[i] = inf if self.mnt_modes[i] == "min" else -inf
        else:
            self.monitor = "off"
            self.mnt_bests = []

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir
        self.log_dir = config.log_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.writer.add_model(self.model)
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns_epoch],
                                           *[m.__name__ for m in self.metric_ftns_running], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns_epoch],
                                           *[m.__name__ for m in self.metric_ftns_running], writer=self.writer)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        # images = next(iter(self._data_loader))[0]
        # images = images.to(self.device)
        self.model.eval()
        # self.writer.write_graph(images)
        self.writer.write_weight_histograms()
        self.model.train()
        not_improved_counts = [0 for i in range(len(self.monitor))]
        for epoch in tqdm(range(self.start_epoch, self.epochs + 1), total=self.epochs, desc='Epoch'):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            self.print_logger(log)

            # save best checkpoints according to tracked metrics
            if self.monitor != "off" and epoch % self.val_period == 0 and epoch >= self.start_checkpointing_at:
                # update the values in mnt_bests first so they're all up to date for every checkpoint
                improved = [None for m_idx in range(len(self.monitor))]
                for m_idx in range(len(self.monitor)):
                    if m_idx in self.disabled_metric_idx:
                        continue
                    try:
                        improved[m_idx] = (self.mnt_modes[m_idx] == "min" and log[self.mnt_metrics[m_idx]] < self.mnt_bests[m_idx]) or \
                                          (self.mnt_modes[m_idx] == "max" and log[self.mnt_metrics[m_idx]] > self.mnt_bests[m_idx])
                    except KeyError:
                        self.logger.warning(f"Warning: Metric {self.mnt_metrics[m_idx]} not found. Disabling tracking of this metric.")
                        self.disabled_metric_idx += [m_idx]
                        improved[m_idx] = False
                    if improved[m_idx]:
                        self.mnt_bests[m_idx] = log[self.mnt_metrics[m_idx]]
                # now that the values are up to date, we can save checkpoints with the correct metric values
                for m_idx in range(len(self.monitor)):
                    if m_idx in self.disabled_metric_idx:
                        continue
                    if improved[m_idx]:
                        not_improved_counts[m_idx] = 0
                        self._save_checkpoint(epoch, save_epoch=False, metric_idx=m_idx)
                    else:
                        not_improved_counts[m_idx] += 1
                metrics_not_improving = [not_improved_counts[m_idx] > self.early_stop for m_idx in range(len(self.monitor))]
                if all(metrics_not_improving):
                    self.logger.info(f"Validation performance didn't improve for {self.early_stop} epochs. Stopping training.")
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_epoch=True)

        if self.do_evaluation:
            best_model_paths = list(self.checkpoint_dir.glob("model_best*.pth"))
            if len(best_model_paths) == 0:
                self.logger.warning("No model file with file name pattern model_best*.pth found.")
                return
            if len(best_model_paths) > 1:
                self.logger.warning("Best model selection is ambiguous. Please evaluate your models manually via test.py.")
                return
            else:
                self._resume_checkpoint(best_model_paths[0])
                self.evaluate()

    def print_logger(self, log):
        # print logged informations to the screen
        for key, value in log.items():
            self.logger.info(f'    {key:30s}: {value:3.4f}')
            self.log.update({key: self.log.get(key, []) + [value]})
        self.logger.info('----' * 15)

        torch.save(self.log, str(self.log_dir / f'metric_logger.pth'))

    def evaluate(self):
        pass

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_epoch=True, metric_idx=None):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_epoch: If True, save a checkpoint with the file name f'checkpoint-epoch{epoch}.pth'
        :param metric_idx: Index of a specific metric in self.monitor, etc., if this checkpoint is supposed to be the
                           best checkpoint for that metric.
        """
        arch = type(self.model).__name__
        lr_sched_dict = None if self.lr_scheduler is None else self.lr_scheduler.state_dict()
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': lr_sched_dict,
            'monitor_best': self.mnt_bests,
            # 'config': self.config
        }
        if save_epoch:
            filename = str(self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth')
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if metric_idx is not None:
            metric_str = self.mnt_metrics[metric_idx]
            best_path = str(self.checkpoint_dir / f'model_best_{metric_str}.pth')
            torch.save(state, best_path)
            self.logger.info(f"Saving current best: model_best_{metric_str}.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_bests = checkpoint['monitor_best']
        # For compatibility with checkpoints created before implementing tracking multiple metrics
        if not isinstance(self.mnt_bests, list):
            self.mnt_bests = [self.mnt_bests]

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        # load optimizer state from checkpoint
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.lr_scheduler is not None:
            try:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            except:
                self.logger.warning("Could not resume lr_scheduler. Continuing anyway, but be careful if training.")

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))