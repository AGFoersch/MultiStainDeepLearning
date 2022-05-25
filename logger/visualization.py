import importlib
import statistics
from datetime import datetime
from queue import Queue
from threading import Thread, Event
from abc import ABC, abstractmethod
from torch import nn
from time import sleep
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils import plot_confusion_matrix, ifnone
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TensorboardWriter():
    def __init__(self, log_dir, logger, enabled):
        self.writer = SummaryWriter(str(log_dir)) if enabled else None
        self.hist_writer = HistogramTBWriter()
        self.stats_writer = ModelStatsTBWriter()
        self.img_writer = ImageTBWriter()
        self.graph_writer = GraphTBWriter()

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding', 'add_figure'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def add_model(self, model:nn.Module):
        self.model = model

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def write_model_stats(self, name='model_stats', model=None):
        model = ifnone(model, self.model)
        self.stats_writer.write(model=model, iteration=self.step, tbwriter=self.writer, name=name)

    def write_weight_histograms(self, name='model', model=None):
        model = ifnone(model, self.model)
        self.hist_writer.write(model=model, iteration=self.step, tbwriter=self.writer, name=name)

    def write_image(self, tag, data):
        self.img_writer.write(image=data, iteration=self.step, tbwriter=self.writer, name=tag)

    def write_embedding(self, features, class_labels, images):
        self.writer.add_embedding(features, metadata=class_labels, label_img=images)

    def write_graph(self, data):
        self.graph_writer.write(model=self.model, tbwriter=self.writer, input_to_model=data)

    def write(self):
        self.write_model_stats()
        self.write_weight_histograms()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if kwargs.get('sub_tag', False):
                        sub_tag = kwargs.get('sub_tag')
                        tag = f'{sub_tag}/{tag}'
                        kwargs.pop('sub_tag')
                    elif name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError(f"type object has no attribute {name}")
            return attr

    def plot_confusion(self, cm, target_names:[str], name:str):
        """
        Takes a matplotlib figure handle and converts it using
        canvas and string-casts to a numpy array that can be
        visualized in TensorBoard using the add_image function

        source: http://martin-mundt.com/tensorboard-figures/
        """
        fig = plot_confusion_matrix(cm, target_names, return_fig=True)
        self.add_figure(f'{name}', fig)

        # # Draw figure on canvas
        # fig.canvas.draw()
        #
        # # Convert the figure to numpy array, read the pixel values and reshape the array
        # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #
        # # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
        # img = img / 255.0
        # img = np.rollaxis(img, 2, 0)
        #
        # # Add figure in numpy "image" to TensorBoard writer
        # plt.close(fig)
        # self.write_image(tag=f'{name}', data=img)


class UnNormalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class TBWriteRequest(ABC):
    "A request object for Tensorboard writes.  Useful for queuing up and executing asynchronous writes."
    def __init__(self, iteration:int, tbwriter: SummaryWriter):
        super().__init__()
        self.tbwriter = tbwriter
        self.iteration = iteration

    @abstractmethod
    def write(self)->None: pass


class AsyncTensorboardWriter():
    def __init__(self):
        super().__init__()
        self.stop_request = Event()
        self.queue = Queue()
        self.thread = Thread(target=self._queue_processor, daemon=True)
        self.thread.start()

    def request_write(self, request: TBWriteRequest):
        if self.stop_request.isSet(): return
        self.queue.put(request)

    def _queue_processor(self):
        while not self.stop_request.isSet():
            while not self.queue.empty():
                if self.stop_request.isSet(): return
                request = self.queue.get()
                request.write()
            sleep(0.2)

    def close(self):
        self.stop_request.set()
        self.thread.join()

    def __enter__(self): pass

    def __exit__(self, exc_type, exc_val, exc_tb): self.close()

asyncTB = AsyncTensorboardWriter()

class HistogramTBRequest(TBWriteRequest):
    "Request object for model histogram writes to Tensorboard."
    def __init__(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str):
        super(HistogramTBRequest, self).__init__(iteration=iteration, tbwriter=tbwriter)
        self.params = [(name, values.clone().detach().cpu()) for (name, values) in model.named_parameters()]
        self.name = name

    def _write_histogram(self, param_name:str, values):
        "Writes single model histogram to Tensorboard"
        splitted = param_name.split('.')
        tag = self.name + f'/{".".join(splitted[:-1])}/' + splitted[-1]
        self.tbwriter.add_histogram(tag, values, self.iteration)

    def write(self) ->None:
        for param_name, values in self.params: self._write_histogram(param_name, values)


class HistogramTBWriter():
    def __init__(self): super().__init__()

    def write(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str='model'):
        request = HistogramTBRequest(model=model, iteration=iteration, tbwriter=tbwriter, name=name)
        asyncTB.request_write(request)


class ModelStatsTBRequest(TBWriteRequest):
    "Request object for model gradient statistics writes to Tensorboard."
    def __init__(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str):
        super().__init__(tbwriter=tbwriter, iteration=iteration)
        self.gradients = [x.grad.clone().detach().cpu() for x in model.parameters() if x.grad is not None]
        self.name = name

    def _add_gradient_scalar(self, name:str, scalar_value)->None:
        "Writes a single scalar value for a gradient statistic to Tensorboard."
        tag = self.name + '/gradients/' + name
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=self.iteration)

    def _write_avg_norm(self, norms:[])->None:
        "Writes the average norm of the gradients to Tensorboard."
        avg_norm = sum(norms)/len(self.gradients)
        self._add_gradient_scalar('avg_norm', scalar_value=avg_norm)

    def _write_median_norm(self, norms:[])->None:
        "Writes the median norm of the gradients to Tensorboard."
        median_norm = statistics.median(norms)
        self._add_gradient_scalar('median_norm', scalar_value=median_norm)

    def _write_max_norm(self, norms:[])->None:
        "Writes the maximum norm of the gradients to Tensorboard."
        max_norm = max(norms)
        self._add_gradient_scalar('max_norm', scalar_value=max_norm)

    def _write_min_norm(self, norms:[])->None:
        "Writes the minimum norm of the gradients to Tensorboard."
        min_norm = min(norms)
        self._add_gradient_scalar('min_norm', scalar_value=min_norm)

    def _write_num_zeros(self)->None:
        "Writes the number of zeroes in the gradients to Tensorboard."
        gradient_nps = [x.data.numpy() for x in self.gradients]
        num_zeros = sum((np.asarray(x) == 0.0).sum() for x in gradient_nps)
        self._add_gradient_scalar('num_zeros', scalar_value=num_zeros)

    def _write_percent_zeros(self):
        "Writes the percentage of zeroes in the gradients to Tensorboard."
        gradient_nps = [x.data.numpy() for x in self.gradients]
        ges_gradients = sum(x.numel() for x in self.gradients)
        percentage_zeros = sum((np.asarray(x) == 0.0).sum() for x in gradient_nps) / ges_gradients
        self._add_gradient_scalar('percentage_zeros', scalar_value=percentage_zeros)

    def _write_avg_gradient(self)->None:
        "Writes the average of the gradients to Tensorboard."
        avg_gradient = sum(x.data.mean() for x in self.gradients)/len(self.gradients)
        self._add_gradient_scalar('avg_gradient', scalar_value=avg_gradient)

    def _write_median_gradient(self)->None:
        "Writes the median of the gradients to Tensorboard."
        median_gradient = statistics.median(x.data.median() for x in self.gradients)
        self._add_gradient_scalar('median_gradient', scalar_value=median_gradient)

    def _write_max_gradient(self)->None:
        "Writes the maximum of the gradients to Tensorboard."
        max_gradient = max(x.data.max() for x in self.gradients)
        self._add_gradient_scalar('max_gradient', scalar_value=max_gradient)

    def _write_min_gradient(self)->None:
        "Writes the minimum of the gradients to Tensorboard."
        min_gradient = min(x.data.min() for x in self.gradients)
        self._add_gradient_scalar('min_gradient', scalar_value=min_gradient)

    def write(self)->None:
        "Writes model gradient statistics to Tensorboard."
        if len(self.gradients) == 0: return
        norms = [x.data.norm() for x in self.gradients]
        self._write_avg_norm(norms=norms)
        self._write_median_norm(norms=norms)
        self._write_max_norm(norms=norms)
        self._write_min_norm(norms=norms)
        self._write_num_zeros()
        self._write_percent_zeros()
        self._write_avg_gradient()
        self._write_median_gradient()
        self._write_max_gradient()
        self._write_min_gradient()

class ModelStatsTBWriter():
    "Writes model gradient statistics to Tensorboard."
    def write(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str='model_stats')->None:
        "Writes model gradient statistics to Tensorboard."
        request = ModelStatsTBRequest(model=model, iteration=iteration, tbwriter=tbwriter, name=name)
        asyncTB.request_write(request)

class ImageTBRequest(TBWriteRequest):
    "Request object for model image output writes to Tensorboard."
    def __init__(self, image, iteration:int, tbwriter:SummaryWriter, name:str):
        super().__init__(tbwriter=tbwriter, iteration=iteration)
        self.image = image
        self.name = name

    def write(self)->None:
        "Writes original, generated and real(target) images to Tensorboard."
        self.tbwriter.add_image(self.name, self.image, self.iteration)


# If this isn't done async then this is noticeably slower
class ImageTBWriter():
    "Writes model image output to Tensorboard."
    def __init__(self): super().__init__()

    def write(self, image, iteration:int, tbwriter:SummaryWriter, name:str)->None:
        "Writes training and validation batch images to Tensorboard."
        request = ImageTBRequest(image=image, iteration=iteration, tbwriter=tbwriter, name=name)
        asyncTB.request_write(request)


class GraphTBRequest(TBWriteRequest):
    "Request object for model histogram writes to Tensorboard."
    def __init__(self, model:nn.Module, tbwriter:SummaryWriter, input_to_model):
        super().__init__(tbwriter=tbwriter, iteration=0)
        self.model,self.input_to_model = model,input_to_model

    def write(self)->None:
        "Writes single model graph to Tensorboard."
        self.tbwriter.add_graph(model=self.model, input_to_model=self.input_to_model)


class GraphTBWriter():
    "Writes model network graph to Tensorboard."
    def write(self, model:nn.Module, tbwriter:SummaryWriter, input_to_model)->None:
        "Writes model graph to Tensorboard."
        request = GraphTBRequest(model=model, tbwriter=tbwriter, input_to_model=input_to_model)
        asyncTB.request_write(request)