import unittest

import torch

from haruna.models import UNet


class TestUNet(unittest.TestCase):

    def setUp(self):
        self.model = UNet()
        self.model.eval()

    def test_forward(self):
        x = torch.randn(1, 3, 250, 250)
        with torch.no_grad():
            y = self.model(x)

        self.assertListEqual(list(y.size()), [1, 3, 52, 52])
