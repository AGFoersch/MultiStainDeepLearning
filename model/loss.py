import torch


class CoxLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, censor: torch.tensor, survtime: torch.tensor, hazard_pred: torch.tensor, device):
        current_batch_len = len(survtime)
        R_mat = (survtime[None, :] >= survtime[:, None]).float() # this is R_mat[i,j] = survtime[j] >= survtime[i] for all i,j without the loops
        theta = torch.nn.Sigmoid()(hazard_pred).reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
        return loss_cox