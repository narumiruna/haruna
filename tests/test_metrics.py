import unittest
from haruna.metrics import Average
from haruna.metrics import ExponentialMovingAverage


class TestAverage(unittest.TestCase):

    def test_update(self):
        m = Average()
        m.update(1)
        m.update(2)
        m.update(3)
        m.update(4)
        self.assertEqual(m.value, 2.5)


class TestExponentialMovingAverage(unittest.TestCase):

    def test_update(self):
        m = ExponentialMovingAverage(bias_correct=False)
        m.update(1)
        m.update(2)
        self.assertEqual(m.value, 1.25)
