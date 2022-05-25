import torch
import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


def accuracy(output, target, ignore_index=None):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        if ignore_index is not None:
            mask_arr = target.ne(ignore_index)
            target = target.masked_select(mask_arr)
            pred = pred.masked_select(mask_arr)
        correct = torch.sum(pred == target).item()
    return correct / target.numel()


def accuracy_epoch(output, target, ignore_index=None):
    target = target
    assert output.shape[0] == len(target)
    correct = torch.sum(output == target)
    return correct / len(target)


def cox_accuracy(hazardsdata, targets):
    target = targets[0]
    median = torch.quantile(hazardsdata, q=0.5)
    hazards_dichotomize = torch.zeros(len(hazardsdata), dtype=torch.long, device=hazardsdata.device)
    hazards_dichotomize[hazardsdata.squeeze() > median] = 1
    correct = torch.sum(torch.isclose(hazards_dichotomize, target)).item()
    return correct / target.numel()


def cox_accuracy_epoch(hazardsdata, targets):
    target = targets[0] # == censor data
    median = torch.quantile(hazardsdata, q=0.5) # should give the same result as np.median(hazardsdata)
    hazards_dichotomize = torch.zeros((len(hazardsdata),), dtype=torch.long)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = torch.sum(torch.isclose(hazards_dichotomize, target)) # I think having == rather than isclose might have caused this to always be 0.
    return correct / len(target)


def c_index(hazardsdata, targets, survtime_all):
    # target = targets[0]
    # survtime = targets[1]
    concord = 0.
    total = 0.
    N_test = targets.shape[0]
    for i in range(N_test):
        if targets[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total += 1
                    if hazardsdata[j] < hazardsdata[i]: concord += 1
                    elif hazardsdata[j] < hazardsdata[i]: concord += 0.5
    return concord/total if total > 0 else 0.0


def c_index_lifeline(hazarddata, targets):
    target_all = targets[0]
    survtime_all = targets[1]
    return(concordance_index(survtime_all, -hazarddata, target_all))


def c_index_lifeline_run(hazarddata, targets):
    target_all = targets[0].cpu().detach().numpy()
    survtime_all = targets[1].cpu().detach().numpy()
    hazarddata = hazarddata.cpu().detach().numpy().reshape(-1)
    return(concordance_index(survtime_all, -hazarddata, target_all))


def p_value(hazardsdata, targets):
    target_all = targets[0]
    survtime_all = targets[1]
    #median = np.median(hazardsdata)
    median = torch.quantile(hazardsdata, q=0.5)
    #hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize = torch.zeros((len(hazardsdata),), dtype=torch.int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = torch.isclose(hazards_dichotomize, torch.tensor([0], dtype=torch.int))
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = target_all[idx]
    E2 = target_all[~idx]
    result = logrank_test(T1, T2, E1, E2)
    return result.p_value


def p_value_run(hazardsdata, targets):
    target_all = targets[0].cpu().detach().numpy()
    survtime_all = targets[1].cpu().detach().numpy()
    hazardsdata = hazardsdata.cpu().detach().numpy().reshape(-1)

    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = target_all[idx]
    E2 = target_all[~idx]
    result = logrank_test(T1, T2, E1, E2)
    return result.p_value


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def precision(tp, tn, fp, fn):
    return tp / float(tp + fp)


def recall(tp, tn, fp, fn):
    return tp / float(tp + fn)


def f1_score(tp, tn, fp, fn):
    return 2. * tp / (2. * tp + fp + fn)


