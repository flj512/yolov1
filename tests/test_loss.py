import torch

from config import LAMBDA_COORD, LAMBDA_NOOBJ
from utils.loss import YOLOLoss


def wh_error(v1, v2):
    return (v1**0.5 - v2**0.5) ** 2


def xy_error(v1, v2):
    return (v1 - v2) ** 2


def compare_loss(predict, target, result):
    loss_fn = YOLOLoss()
    loss, _ = loss_fn(predict, target)

    assert (
        abs(loss.item() - result) < 1.0e-5
    ), f"Expected loss: {result}, Got: {loss.item()}"


def test_loss_1():
    batch_size = 2
    predict = torch.zeros((batch_size, 7, 7, 30))
    target = torch.zeros((batch_size, 7, 7, 25))

    # batch 0
    target[0, 3, 5, 0:7] = torch.tensor([0.5, 0.5, 0.3, 0.4, 1.0, 0.88, 0.13])
    predict[0, 3, 5, 0:12] = torch.tensor(
        [0.45, 0.43, 0.2, 0.3, 0.77, 0.55, 0.6, 0.1, 0.1, 1.0, 0.33, 0.26]
    )
    predict[0, 6, 6, 0:11] = torch.tensor(
        [0.5, 0.5, 0.3, 0.4, 0.71, 0.5, 0.5, 0.3, 0.4, 0.63, 1.0]
    )
    predict[0, 4, 5, 25] = 1.0
    predict[0, 5, 5, 4] = 0.49
    predict[0, 6, 0, 9] = 0.38

    l01 = LAMBDA_COORD * (
        wh_error(0.2, 0.3)
        + wh_error(0.3, 0.4)
        + xy_error(0.5, 0.45)
        + xy_error(0.5, 0.43)
    )
    l02 = (1.0 - 0.77) ** 2
    l03 = LAMBDA_NOOBJ * (0.71**2 + 0.63**2 + 0.49**2 + 0.38**2)
    l04 = (0.88 - 0.33) ** 2 + (0.26 - 0.13) ** 2

    # batch 1
    target[1, 2, 4, 0:7] = torch.tensor([0.5, 0.5, 0.3, 0.4, 1.0, 0.75, 0.13])
    predict[1, 2, 4, 0:12] = torch.tensor(
        [0.45, 0.43, 0.2, 0.3, 0.68, 0.55, 0.6, 0.1, 0.1, 1.0, 0.33, 0.29]
    )
    target[1, 2, 4, 24] = 0.34
    predict[1, 2, 4, 29] = 0.28
    predict[1, 6, 6, 0:11] = torch.tensor(
        [0.5, 0.5, 0.3, 0.4, 0.71, 0.5, 0.5, 0.3, 0.4, 0.63, 1.0]
    )
    predict[1, 4, 5, 25] = 1.0
    predict[1, 5, 5, 4] = 0.49
    predict[1, 3, 0, 9] = 0.58
    predict[1, 3, 0, 4] = 0.19

    l11 = LAMBDA_COORD * (
        wh_error(0.2, 0.3)
        + wh_error(0.3, 0.4)
        + xy_error(0.5, 0.45)
        + xy_error(0.5, 0.43)
    )
    l12 = (1.0 - 0.68) ** 2
    l13 = LAMBDA_NOOBJ * (0.71**2 + 0.63**2 + 0.49**2 + 0.58**2 + 0.19**2)
    l14 = (0.75 - 0.33) ** 2 + (0.29 - 0.13) ** 2 + (0.34 - 0.28) ** 2

    compare_loss(
        predict, target, (l01 + l02 + l03 + l04 + l11 + l12 + l13 + l14) / batch_size
    )


if __name__ == "__main__":
    test_loss_1()
    print("All tests passed")
