import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from captum.attr import GuidedGradCam, LayerGradientXActivation
from captum.attr import visualization as viz

import utils
from base import BaseTrainer


class BasicMultiTrainer(BaseTrainer):
    """
    A rewrite of the former BasicMultiTrainer class.
    This trainer can:
        - handle both unimodal and multimodal models (although both should use the MultiModel class)
        - handle training via both Cox and Cross Entropy loss
        - generally automatically determine whether we are in the C-Index or the classification case
          and act accordingly
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader:DataLoader,
                 valid_data_loader:DataLoader=None, lr_scheduler=None):
        super(BasicMultiTrainer, self).__init__(
            model, criterion, metric_ftns, optimizer, config, lr_scheduler, data_loader, valid_data_loader
        )
        self.freeze = self.config["trainer"].get("freeze", False)
        if self.freeze:
            for model in self.model.models:
                utils.dfs_freeze(model)
        self.unfreezing = self.config["trainer"].get("unfreeze_after", np.inf)

        self.train_metrics.add_metric("loss_epoch")
        self.valid_metrics.add_metric("loss_epoch")

        """
        Tell classification / cindex cases apart by looking at the loss function.
        These two cases behave differently in the following ways:
        - Loss: CoxLoss for CIndex, CrossEntropyLoss for Classification
        - Labels: Two components for CIndex (survival status, survival time), one for classification (the class)
        - Model structure: One output component for CIndex, c for classification (where c is the number of classes)
        - Handling of model output: CIndex output needs a sigmoid. Classification outputs need a softmax or argmax.
        - Evaluation: Afaik, there is no way to access accuracies/probabilities in the cindex case. As such, utils'
                      evaluate method shouldn't be run in that case.
        """
        if self.criterion._get_name() == "CoxLoss":
            self.is_classification = False
        else:
            self.is_classification = True
            if self.criterion._get_name() != "CrossEntropyLoss":
                self.logger.warning("The trainer assumes you want to train a classifier, but you're not using "
                                    "CrossEntropyLoss. Is this intentional?")

        if self.lr_scheduler is not None:
            supported_lr_schedulers = [
                "ReduceLROnPlateau", "LambdaLR", "MultiplicativeLR", "StepLR", "MultiStepLR", "ExponentialLR"
            ]
            lr_sched_name = self.lr_scheduler._get_name()
            if lr_sched_name not in supported_lr_schedulers:
                self.logger.warning(f"{lr_sched_name} is not explicitely supported by this trainer. "
                                    "Please ensure that your scheduler's step should be called at the end of each EPOCH, "
                                    "not batch, then feel free to ignore this warning.")

        # Determine some info about the output and label shapes. This should help
        # speed up epochs a little since we can preallocate appropriate amounts of space instead
        # of dynamically appending data to arrays or lists.
        # @ me: _train_epoch iterates over self.data_loader, all validation stuff over self.valid_data_loader
        self.train_num_data = len(self.data_loader.dataset)
        self.train_num_batches = len(self.data_loader)
        self.val_num_data = len(self.valid_data_loader.dataset)
        self.val_num_batches = len(self.valid_data_loader)
        self.label_components = 1 if self.is_classification else 2
        self.output_components = self.model.num_classes if self.is_classification else 1
        self.num_modalities = len(self.model.models)

        # Calculate class weights from the training data if applicable.
        # The train loader iterates over patients rather than individual samples, so the class weights are calculated
        # based on the number of patients for each class.
        # If you don't want class weight calculation to happen, you have to explicitely state that in your config file.
        if self.is_classification:
            compute_cw = self.config["trainer"].get("compute_class_weights", True)
            if compute_cw:
                unique_classes   = self.data_loader.train_dataframes[0][self.data_loader.labels[0]].unique()
                label_column_name = self.data_loader.labels[0]
                first_pid = self.data_loader.train_patients[0]
                class_per_pid = self.data_loader.train_dicts[0][first_pid][label_column_name].to_numpy()[:1]
                for pid in self.data_loader.train_patients[1:]:
                    current_class = self.data_loader.train_dicts[0][pid][label_column_name].to_numpy()[:1]
                    class_per_pid = np.concatenate([class_per_pid, current_class])
                weights = compute_class_weight("balanced", classes=unique_classes, y=class_per_pid)
                tqdm.write(f"Applying class weights for loss calculation: {weights}")
                self.criterion.weight = torch.tensor(weights).float().to(self.device)

        ###### Visualization stuff ######
        # NOTE: You might need to reduce the batch size to run the visualizations without running out of GPU memory.
        # Halving the batch size compared to training worked for me.
        ### general settings ###
        try:
            vis_options = self.config["trainer"]["vis_options"]
            self.make_vis = vis_options.get("enabled", False)
        except:
            self.make_vis = False
        if self.make_vis:
            self.plot_info = vis_options.get("plot_info", True)
            self.img_size = vis_options.get("img_size", 5) # rough image size (a la matplotlib's figsize) for each individual image in the final figures
            self.img_per_col = vis_options.get("img_per_col", 1) # Images are arranged in a grid. How many images per column?
            ### guided gradcam ###
            try:
                ggradcam_config = vis_options["guided_gradcam"]
                self.make_ggradcam = ggradcam_config.get("enabled", False)
            except:
                self.make_ggradcam = False
            if self.make_ggradcam:
                # The necessary captum objects are instanciated in init_ggradcam when the time comes. Instanciating
                # them switches out the model's forward method, so I'll do it later.
                # ---------
                # Label index to backpropagate. If None, use the predicted label.
                # This only applies to the classification case
                self.ggradcam_backprop_label = ggradcam_config.get("backprop_label", None)
                # If True: Backpropagate the sigmoided hazard. If False: Backpropagate 1 - (sigmoided hazard)
                self.ggradcam_use_hazard_max = ggradcam_config.get("use_hazard_max", True)
                self.include_impacts = ggradcam_config.get("include_impacts", True)
                if self.include_impacts:
                    fusion_layers = self.model.get_fusion_output_layers()
                    if self.is_classification:
                        self.actxgrad_maker = LayerGradientXActivation(self.model.regular_forward, fusion_layers)
                    else:
                        self.actxgrad_maker = LayerGradientXActivation(self.model.forward_for_impact_survival, fusion_layers)
                self.ggradcam_cmap = ggradcam_config.get("colormap", "coolwarm")


    def init_ggradcam(self):
        """
        This method switches out the model's regular forward method with the forward_for_gradcam method.
        Keep this in mind if you want to keep doing stuff with the model, maybe switch it back once gradcams
        are done.
        """
        backprop_layers = self.model.get_last_activ_layers()
        if self.is_classification:
            self.model.forward = self.model.forward_for_gradcam_classification
        else:
            self.model.forward = self.model.forward_for_gradcam_survival
        self.ggradcam_makers = [GuidedGradCam(self.model, layer, self.device_ids) for layer in backprop_layers]


    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()

        # this collects either the sigmoided hazard or the argmaxed classification result.
        if self.is_classification:
            pred_all = torch.zeros((self.train_num_data,), dtype=torch.long)
        else:
            pred_all = torch.zeros((self.train_num_data,))
        labels_all = torch.zeros((self.label_components, self.train_num_data))

        if self.unfreezing == epoch:
            if self.freeze:
                for model in self.model.models:
                    utils.dfs_unfreeze(model)
                self.logger.info("Unfreezed unimodal models!")
            else:
                for model in self.model.models:
                    utils.dfs_freeze(model)
                self.logger.info("Freezed unimodal models!")

        batch_bar = tqdm(total=self.train_num_batches, desc="Batches", position=1)
        # offset used for indexing into the _all tensors. I'm using this rather than doing
        # batch_idx*batch_size since the last batch of an epoch can potentially have
        # size != batch_size.
        offset = 0
        for batch_idx, (inputs, labels) in enumerate(self.data_loader):
            bs = len(labels[0])
            #### Transfer data to GPU, gather labels ####
            for i in range(self.num_modalities):
                # this omits transferring genetic data if it's unused, which should be fine I think.
                # If you see this comment while debugging, maybe it's not :)
                inputs[i] = inputs[i].to(self.device)
            for i in range(self.label_components):
                labels_all[i, offset:offset+bs] = labels[i]
                labels[i] = labels[i].to(self.device) # all labels[i] are already torch.tensor at this point

            #### Optimization step, gather output ####
            self.optimizer.zero_grad()
            output = self.model(*inputs)
            if self.is_classification:
                # applies cross entropy
                loss = self.criterion(output, labels[0])
            else:
                # applies coxloss
                loss = self.criterion(labels[0], labels[1], output, self.device)
            loss.backward()
            self.optimizer.step()

            #### gather hazard/class predictions ####
            if self.is_classification:
                pred_all[offset:offset+bs] = torch.argmax(output.detach(), dim=1).cpu()
            else:
                pred_all[offset:offset+bs] = torch.nn.Sigmoid()(output[:,0].detach()).cpu()

            #### logging - running metrics ####
            self.writer.set_step((epoch-1)*self.train_num_data + offset)
            self.train_metrics.update("loss", loss.item(), n=bs)
            for met in self.metric_ftns_running:
                self.train_metrics.update(met.__name__, met(output, labels), n=bs)
                self.train_metrics.update_tensorboard(met.__name__)
            if batch_idx % self.log_step == 0:
                self.train_metrics.update_tensorboard("loss")
                self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], sub_tag="lr")
                self.writer.write_model_stats()
            if batch_idx % self.cm_step == 0 and self.is_classification:
                cm = confusion_matrix(labels_all[0, :offset+bs].long(), pred_all[:offset+bs])
                self.writer.plot_confusion(cm, self.data_loader.classes, "confusion_matrix")

            offset += bs
            batch_bar.set_postfix({"loss": loss.item()})
            batch_bar.update(1)
            if batch_idx == self.len_epoch:
                break

        #### logging - epoch metrics ####
        self.writer.set_step(epoch)
        # labels_all[0] (classes or survival status) may need to be integer for everything to
        # work properly, not 100% sure (but it definitely doesn't hurt, so I'll just do it)
        if self.is_classification:
            labels_all = labels_all[0].long()
        else:
            labels_all = [labels_all[0].long(), labels_all[1]]
        for met in self.metric_ftns_epoch:
            self.train_metrics.update(met.__name__, met(pred_all, labels_all))
            self.train_metrics.update_tensorboard(met.__name__)
        self.train_metrics.update("loss_epoch", self.train_metrics.avg("loss"))
        self.train_metrics.update_tensorboard("loss_epoch")
        log = self.train_metrics.result()
        if self.is_classification:
            cm = confusion_matrix(labels_all, pred_all)
            self.writer.plot_confusion(cm, self.data_loader.classes, name="confusion_matrix/train")
        self.writer.write_weight_histograms()
        self.writer.write_model_stats()

        if self.do_validation and epoch % self.val_period == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_"+k: v for k,v in val_log.items()})
            if self.lr_scheduler is not None:
                if self.lr_scheduler._get_name() == "ReduceLROnPlateau":
                    self.lr_scheduler.step(log[self.mnt_metrics[0]])

        if self.lr_scheduler is not None:
            if self.lr_scheduler._get_name() != "ReduceLROnPlateau":
                self.lr_scheduler.step()

        return log


    def _valid_epoch(self, epoch):
        """
        If you feel like there's context/comments missing, check out _train_epoch. Much
        of this is very similar to what's happening there.
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            pred_all = torch.zeros((self.val_num_data,))
            labels_all = torch.zeros((self.label_components, self.val_num_data))

            batch_bar = tqdm(total=self.val_num_batches, desc="Validation", position=1)
            offset = 0
            for batch_idx, (inputs, labels) in enumerate(self.valid_data_loader):
                bs = len(labels[0])
                for i in range(self.num_modalities):
                    inputs[i] = inputs[i].to(self.device)
                for i in range(self.label_components):
                    labels_all[i, offset:offset + bs] = labels[i]
                    labels[i] = labels[i].to(self.device)

                output = self.model(*inputs)
                if self.is_classification:
                    loss = self.criterion(output, labels[0])
                    pred_all[offset:offset+bs] = torch.argmax(output.detach(), dim=1).cpu()
                else:
                    loss = self.criterion(labels[0], labels[1], output, self.device)
                    pred_all[offset:offset+bs] = torch.nn.Sigmoid()(output[:,0].detach()).cpu()

                self.writer.set_step((epoch-1)*self.val_num_data + offset, "valid")
                self.valid_metrics.update("loss", loss.item(), n=bs)
                for met in self.metric_ftns_running:
                    self.valid_metrics.update(met.__name__, met(output, labels), n=bs)

                offset += bs
                batch_bar.set_postfix({"loss": loss.item()})
                batch_bar.update(1)

            self.valid_metrics.update_tensorboard("loss")
            for met in self.metric_ftns_running:
                self.valid_metrics.update_tensorboard(met.__name__)

            if self.is_classification:
                labels_all = labels_all[0].long()
            else:
                labels_all = [labels_all[0].long(), labels_all[1]]
            self.writer.set_step(epoch, "valid")
            for met in self.metric_ftns_epoch:
                self.valid_metrics.update(met.__name__, met(pred_all, labels_all))
                self.valid_metrics.update_tensorboard(met.__name__)
            self.valid_metrics.update("loss_epoch", self.valid_metrics.avg("loss"))
            self.valid_metrics.update_tensorboard("loss_epoch")
            if self.is_classification:
                cm = confusion_matrix(labels_all, pred_all)
                self.writer.plot_confusion(cm, self.data_loader.classes, name="confusion_matrix/valid")

            return self.valid_metrics.result()


    def get_preds(self):
        self.model.eval()
        with torch.no_grad():
            # CAREFUL: for the classification case, pred_all contains the softmaxed output
            # rather than the argmaxed one.
            if self.is_classification:
                pred_all = torch.zeros((self.val_num_data, self.output_components))
            else:
                pred_all = torch.zeros((self.val_num_data,))
            batch_bar = tqdm(total=self.val_num_batches, desc="Predict", position=1)
            offset = 0
            for batch_idx, (inputs, labels) in enumerate(self.valid_data_loader):
                bs = len(labels[0])
                for i in range(self.num_modalities):
                    inputs[i] = inputs[i].to(self.device)
                output = self.model(*inputs)
                if self.is_classification:
                    pred_all[offset:offset+bs] = torch.softmax(output, dim=1).cpu()
                else:
                    pred_all[offset:offset+bs] = torch.nn.Sigmoid()(output[:,0]).cpu()
                offset += bs
                batch_bar.update(1)
        return pred_all.numpy()


    def evaluate(self):
        tqdm.write("Evaluation...")
        eval_dir = self.log_dir/"evaluation"
        self.model.eval()
        try:
            df = self.data_loader.valid_dataframe
        except:
            tqdm.write("The dataloader does not contain a validation dataframe. Exiting evaluation now.")
            return
        preds = self.get_preds()
        ###### Metric curves via utils' evaluate method ######
        if self.is_classification:
            df_for_evaluate = df.rename(columns={self.data_loader.labels[0]: "Label"})
            utils.evaluate(self._data_loader.class_dict, df_for_evaluate, preds, path=eval_dir, save_fig=True)
        else:
            tqdm.write("utils.evaluate not available for C-Index case. Not creating metric curves and such.")
        self.save_eval_df(df, preds, eval_dir)
        ###### Visualization stuff ######
        if self.make_vis:
            if self.make_ggradcam:
                tqdm.write("Creating guided grad-cam images...")
                self.create_ggradcam(eval_dir, df, preds)


    def create_ggradcam(self, eval_dir, val_df, preds):
        self.init_ggradcam() # this switches the model's forward method to forward_to_captum
        ggradcam_dir = eval_dir/"guided_gradcams"
        ggradcam_dir.mkdir(exist_ok=True, parents=True)

        offset = 0
        batch_bar = tqdm(total=self.val_num_batches, desc="Guided Grad-Cam batch...", position=1)
        for batch_idx, (inputs, labels) in enumerate(self.valid_data_loader):
            bs = len(labels[0])
            relevant_preds = torch.tensor(preds[offset:offset+bs])
            if self.is_classification:
                if self.ggradcam_backprop_label is None:
                    backprop_idx = torch.argmax(relevant_preds, dim=1)  # backprop predicted labels
                else:
                    backprop_idx = torch.tensor([self.ggradcam_backprop_label]*bs)  # backprop chosen label
            else:
                if self.ggradcam_use_hazard_max:
                    backprop_idx = torch.tensor([0]*bs) # backprop sigmoided output
                else:
                    backprop_idx = torch.tensor([1]*bs) # backprop 1 - (sigmoided output)
            backprop_idx = backprop_idx.to(self.device)
            attributions = []
            for inp_idx in range(len(inputs)):
                inputs[inp_idx] = inputs[inp_idx].to(self.device)
            for m_idx, modality in enumerate(self.data_loader.valid_dataset.valid_columns):
                inputs[m_idx].requires_grad = True
                other_tensors = [(i, tensor) for i, tensor in enumerate(inputs) if i != m_idx]
                # shape before permutation: (bs, 3, y_size, x_size)
                attrib = self.ggradcam_makers[m_idx].attribute(
                    inputs[m_idx], additional_forward_args=(m_idx, other_tensors), target=backprop_idx
                ).detach().cpu() # shape (bs, ch, y, x)
                # If all values in attrib are too close to zero, captum's visualize method throws an error while
                # normalizing the image. Solution if this is the case: Add a very small positive value to one
                # pixel, add a very small negative value to another pixel of the image.
                # This way, captum should hopefully use that for normalization, making the image simply appear grey
                # (other than those two pixels, which are hopefully not all that apparent visually)
                eps = 1E-7
                for attrib_idx, attrib_img in enumerate(attrib):
                    if torch.allclose(attrib_img, torch.tensor([0.0])):
                        # try this instead in case two pixels are not enough?
                        #max_y = attrib_img.shape[1]-1
                        #attrib[attrib_idx,:,0,:] = eps
                        #attrib[attrib_idx,:,max_y,:] = -eps
                        attrib[attrib_idx,:,0,0] = eps
                        attrib[attrib_idx,:,0,1] = -eps
                attributions += [attrib.permute(0,2,3,1).numpy()]
                inputs[m_idx].requires_grad = False

            #### Impact score stuff ####
            if self.include_impacts:
                for inp_idx in range(len(inputs)):
                    inputs[inp_idx].requires_grad = True
                actxgrad = self.actxgrad_maker.attribute(tuple(inputs), target=backprop_idx) # list of tensors (1 for each layer, aka each modality)
                impacts = torch.zeros((len(actxgrad), *actxgrad[0].shape))
                for i in range(len(actxgrad)):
                    impacts[i] = actxgrad[i]
                # Doing this should be the same as first ReLU-ing the gradient and then multiplying with the activation
                # because the activation was already ReLU'd here (see model.fusion.outputs)
                impacts = torch.maximum(impacts, torch.tensor([0]))
                impacts = torch.mean(impacts, dim=(2,)) # mean over the output neurons - new shape: (num_modalities, batch_size)
                impacts = impacts.detach().cpu()
                # softmaxing doesn't really make sense for the scales the impacts are in, so let's not do that.
                # normalized_impacts = torch.softmax(impacts, dim=0)
                # next attempt: normalize by their sum. The impacts are all non-negative, so this should be fine
                # as long as this sum is not zero. And if it is - well, too bad.
                normalized_impacts = impacts / torch.sum(impacts, dim=(0,), keepdim=True)
            else:
                impacts = None
                normalized_impacts = None

            self.plot_ggradcam(
                attributions, offset, backprop_idx, relevant_preds, impacts, normalized_impacts, val_df, ggradcam_dir
            )
            batch_bar.update(1)
            offset += bs
            torch.cuda.empty_cache()
        self.model.forward = self.model.regular_forward # undo the forward method change from self.init_ggradcam


    def plot_ggradcam(self, attributions, offset, backprop_idx, preds, impacts, normalized_impacts, val_df, save_dir):
        num_imgs = attributions[0].shape[0]
        num_modalities = len(attributions)
        nrows = num_modalities // self.img_per_col
        ncols_outer = self.img_per_col
        ncols_all = 2*self.img_per_col
        label_names = self.data_loader.labels
        label2idx = self.data_loader.class_dict

        if self.is_classification:
            argmaxed_preds = torch.argmax(preds, dim=1)

        for i in range(num_imgs):
            df_idx = offset + i
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols_all, squeeze=False, figsize=(ncols_all*(self.img_size+2), nrows*(self.img_size+2))
            )
            patient_id = val_df.iloc[df_idx]["Patient_ID"]
            if self.plot_info:
                title_patient = f"patient ID: {patient_id}"
                title_labels = "real labels: "
                for lnum, lname in enumerate(label_names):
                    lstr = val_df.iloc[df_idx][lname]
                    if lnum == 0:
                        lidx = label2idx[lstr]
                        title_labels += f"{lname}: {lstr} ({lidx}), "
                    else:
                        title_labels += f"{lname}: {lstr}, "
                title_labels = title_labels[:-2] # remove trailing comma and whitespace
                if self.is_classification:
                    title_pred = f"predicted label (prob): {argmaxed_preds[i]} ({preds[i, argmaxed_preds[i]]:.3f})"
                    title_backprop = f"backprop of: {backprop_idx[i]}"
                else:
                    title_pred = f"predicted hazard (sigmoided): {preds[i]:.3f}"
                    if self.ggradcam_use_hazard_max:
                        title_backprop = "backprop of sigmoided hazard"
                    else:
                        title_backprop = "backprop of 1 - (sigmoided hazard)"
                title = f"{title_patient}\n{title_labels}\n{title_pred}\n{title_backprop}"
                fig.suptitle(title)
            df_dict = {"ggradcam_img_name": []}
            for m_idx, modality in enumerate(self.data_loader.valid_dataset.valid_columns):
                df_dict[f"{modality}"] = []
                df_dict[f"{modality}_impact_normalized"] = []
                df_dict[f"{modality}_impact_raw"] = []

            for m_idx, modality in enumerate(self.data_loader.valid_dataset.valid_columns):
                col_offset = 2*(m_idx % ncols_outer)
                row = m_idx // ncols_outer

                viz.visualize_image_attr(
                    attributions[m_idx][i], sign="all", show_colorbar=self.plot_info, cmap=self.ggradcam_cmap, use_pyplot=False,
                    fig_size=(self.img_size, self.img_size), plt_fig_axis=(fig, axes[row, col_offset+0])
                )

                img_path = val_df.iloc[df_idx][modality]
                axes[row, col_offset+1].imshow(plt.imread(img_path))
                axes[row, col_offset+1].set_axis_off()

                if self.plot_info:
                    mod_title = f"{modality}"
                    img_name_title = f"{Path(val_df.iloc[df_idx][modality]).name}"
                    if self.include_impacts:
                        impact_title = f"impact (raw / norm.): {impacts[m_idx][i]:.3E} / {normalized_impacts[m_idx][i]:.3f}"
                        subtitle = f"{mod_title}\n{img_name_title}\n{impact_title}"
                    else:
                        subtitle = f"{mod_title}\n{img_name_title}"
                    axes[row,col_offset+0].set_title(subtitle)

                df_dict[f"{modality}"] += [val_df.iloc[df_idx][modality]]
                df_dict[f"{modality}_impact_normalized"] += [normalized_impacts[m_idx][i].item()]
                df_dict[f"{modality}_impact_raw"] += [impacts[m_idx][i].item()]

            ggradcam_img_name = f"{df_idx}_{patient_id}.png"
            df_dict["ggradcam_img_name"] += [ggradcam_img_name]
            csv_df = pd.DataFrame(df_dict)
            csv_path = save_dir.parent / "ggradcam_data.csv"
            with csv_path.open(mode="a", newline="") as f:
                csv_df.to_csv(f, header=f.tell()==0, index=False)
            file_dir = save_dir/ggradcam_img_name
            fig.savefig(file_dir, bbox_inches="tight")
            plt.close(fig)


    def save_eval_df(self, val_df, preds, save_dir):
        """
        Writes a csv file detailing some of the evaluation results. It contains the following columns:
        - C-Index case:
            Patient_ID, pred_hazard *[labels], *[paths_to_modalities]
        - Classification case:
            Patient_ID, pred_label_idx, probabilities, real_label_idx, *[labels], *[paths_to_modalities]
        where *[labels] and *[paths_to_modalities] are placeholders for the appropriate data from the validation dataframe.
        """
        label_names = self.data_loader.labels
        label2idx = self.data_loader.class_dict
        mod_path_names = self.data_loader.valid_dataset.valid_columns
        if self.is_classification:
            df_dict = {
                "Patient_ID": val_df["Patient_ID"],
                "pred_label_idx": np.argmax(preds, axis=1),
                "probabilities": [np.array(x) for x in preds],
                "real_label_idx": val_df[label_names[0]].map(lambda x: label2idx[x]),
                **{label: val_df[label] for label in label_names},
                **{mod_path: val_df[mod_path] for mod_path in mod_path_names}
            }
        else:
            df_dict = {
                "Patient_ID": val_df["Patient_ID"],
                "pred_hazard": preds,
                **{label: val_df[label] for label in label_names},
                **{mod_path: val_df[mod_path] for mod_path in mod_path_names}
            }
        save_df = pd.DataFrame(df_dict)
        save_df.to_csv(save_dir/"tile_df_with_paths.csv", index=False)