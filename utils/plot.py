import numpy as np
import sklearn.metrics as metrics
from itertools import cycle, product
import matplotlib.pyplot as plt
from pathlib import Path
from numpy import interp


def plot_confusion_matrix(cm, target_names,
                          path=None,
                          title='ConfusionMatrix',
                          cmap=None,
                          normalize=False,
                          save_fig=False,
                          return_fig=False):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else (cm.max() + cm.min()) / 2

    if len(target_names) < 10:
        for i,j in product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j,i, f'{cm[i,j]:.4f}', horizontalalignment='center', color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j,i, f'{cm[i,j]}', horizontalalignment='center', color="white" if cm[i, j] > thresh else "black")

    plt.xlabel(f'Predicted Label\naccuracy={accuracy:.4f}; misclass={misclass:.4f}')
    plt.ylabel('True Label')
    plt.tight_layout(h_pad=2.5)
    if save_fig:
        name = Path(title + '_normalized.png') if normalize else Path(title + '.png')
        path = Path(path) if path is not None else Path('.')
        plt.savefig(path/Path(name), bbox_inches='tight')
    elif return_fig:
        # plt.savefig('./test.png')
        return fig
    else:
        plt.show()
    plt.close()


def plot_auroc(fpr, tpr, auc, path=None, title='AuRoc', save_fig=False):
    fig = plt.figure(figsize=(10,6))
    plt.plot(fpr, tpr, color='darkorange', lw=0.7, label=f'ROC curve (area = {auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.7, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    if save_fig:
        path = Path(path) if path is not None else Path('.')
        plt.savefig(path.joinpath(title + '.png'))
    else:
        plt.show()
    plt.close()


def plot_multi_auroc(class_dict, fpr, tpr, auc, path=None, title='AuRoc', save_fig=False):
    fig = plt.figure(figsize=(10,6))
    plt.plot(fpr['micro'], tpr['micro'], color='deeppink', lw=1.2, ls=':', label=f'micro-average ROC curve (area= {auc["micro"]:.2f})')
    try:
        plt.plot(fpr['macro'], tpr['macro'], color='navy', lw=1.2, ls=':', label=f'macro-average ROC curve (area= {auc["macro"]:.2f})')
    except:
        pass

    colors = cycle(['aqua', 'darkorange', 'greenyellow', 'cornflowerblue', 'firebrick', 'plum'])
    for (cls,i),color in zip(class_dict.items(), colors):
        if np.isnan(tpr[i]).sum() > 0:
            continue
        else:
            plt.plot(fpr[i], tpr[i], color=color, lw=1.2, label=f'ROC curve of class {cls} (area = {auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=0.7, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    if save_fig:
        path = Path(path) if path is not None else Path('.')
        plt.savefig(path.joinpath(title + '.png'))
    else:
        plt.show()
    plt.close()

def plot(class_dict, df, path:str=None, name:str='', save_fig:bool=False):
    tmp_df = df.groupby('Label')
    labels,classes = [],[]
    # Plot Distribution Histograms
    for cls,i in class_dict.items():
        labels.append(i)
        classes.append(cls)
        try:
            df_ = tmp_df.get_group(i)
        except:
            continue
        ax = df_.Probability.str[i].plot.hist(bins=20, title=f'Distribution {cls}')
        fig = ax.get_figure()
        if save_fig:
            path = Path(path) if path is not None else Path('.')
            fig.savefig(path.joinpath(f'prob_distribution_{name}_{cls}.png'))
        else:
            plt.show()
        plt.close()

    # Evaluation Metrics
    report = metrics.classification_report(df.Label, df.Prediction, labels=labels, target_names=classes, digits=4)
    cm     = metrics.confusion_matrix(df.Label, df.Prediction, labels=labels)
    plot_confusion_matrix(cm, class_dict, path, title='ConfusionMatrix'+name, save_fig=save_fig)

    if save_fig:
        path = Path(path) if path is not None else Path('.')
        with open(path.joinpath('Evaluation.txt'), 'a') as file:
            file.write(f'Classification-Report{name}:\n')
            file.write('-' * 30 + '\n')
            file.write(report)
            file.write('-' * 30 + '\n')
            file.write('-' * 30 + '\n')
    else:
        print(f'Classification-Report{name}:\n')
        print('-' * 30 + '\n')
        print(report)
        print('-' * 30 + '\n')
        print('-' * 30 + '\n')

    if len(classes) < 3:
        # Single-Class Classification
        fpr, tpr, threshold = metrics.roc_curve(df.Label, df.Probability.str[1], pos_label=1)
        auc = metrics.auc(fpr, tpr)
        plot_auroc(fpr, tpr, auc, path, 'AuRoc'+name, save_fig)
    else:
        # Multi-Class Classification
        fpr,tpr,auc = dict(),dict(),dict()
        labels_micro,pred_micro = [],[]
        for i in labels:
            tmp_labels = df.Label.apply(lambda x: 1 if x == i else 0)
            predictions = df.Probability.str[i]
            fpr[i],tpr[i],_ = metrics.roc_curve(tmp_labels, predictions, pos_label=1)
            auc[i] = metrics.auc(fpr[i], tpr[i])

            labels_micro.append(tmp_labels)
            pred_micro.append(predictions)

        fpr['micro'],tpr['micro'],_ = metrics.roc_curve(np.array(labels_micro).ravel(), np.array(pred_micro).ravel())
        auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in labels]))
        mean_tpr = np.zeros_like(all_fpr)
        errors = 0
        for i in labels:
            if np.isnan(tpr[i]).sum() > 0:  errors += 1
            else:                           mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= (len(classes) - errors)

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        plot_multi_auroc(class_dict, fpr, tpr, auc, path, 'AuRoc'+name, save_fig)


def plot_to_tensorboard():
    pass


def plot_cam(imgs, heatmaps, idx:int, path=None, impact=None, **kwargs):
    labels = kwargs.get('labels', [""])
    pred = kwargs.get('pred', "")

    fig, axes = plt.subplots(ncols=2, nrows=len(imgs), figsize=(12,(len(imgs)*6+2)))
    fig.suptitle(f'Labels: {*labels,} \nPred: {pred}', fontsize=14)
    max_impact = np.max(impact)
    for i in range(len(imgs)):
        img = imgs[i]
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Impact: {impact[i]:.5f}')

        axes[i, 1].imshow(img)
        axes[i, 1].imshow(heatmaps[i], alpha=0.4, extent=(0,img.size[0], img.size[1], 0), interpolation='bilinear', cmap='jet',)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f'Impact: {impact[i]/max_impact:.3f}')
    path = Path(path) if path is not None else Path('.')
    plt.savefig(path.joinpath(f'grad_cam_{idx}.png'))
    plt.close(fig)