import torch


def f1_score(y_pred, y_true, beta: float = 1, eps: float = 1e-9):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2

    y_pred = y_pred.argmax(dim=1).float()
    y_true = y_true.float()

    TP = (y_pred * y_true).sum()
    prec = TP / (y_pred.sum() + eps)
    rec = TP / (y_true.sum() + eps)
    res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
    return res.mean()


def accuracy(y_pred, y_true):
    y_pred = y_pred.argmax(dim=1).float()
    y_true = y_true.float()

    return (y_pred == y_true).float().sum() / (y_pred.size(0))


if __name__ == '__main__':
    y_pred = torch.Tensor([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6]])
    y_true = torch.Tensor([0, 1, 1])
    print(accuracy(y_pred, y_true))
