import numpy as np
import pandas as pd
import sklearn
#import unittest
from tensorflow import keras
import keras.src
from tensorflow.keras import layers
from twisted.trial import unittest
from assg_tasks import get_logistic_regression_model
from assg_tasks import get_neural_network_model
from assg_tasks import get_more_powerful_model


class test_get_logistic_regression_model(unittest.TestCase):

    def setUp(self):
        self.lr_model = get_logistic_regression_model()

    def test_model_layers(self):
        self.assertEqual(len(self.lr_model.layers), 2)
        self.assertIsInstance(self.lr_model.layers[0], keras.src.layers.core.input_layer.InputLayer)
        self.assertEqual(self.lr_model.layers[0].batch_shape, (None, 2))
        self.assertIsInstance(self.lr_model.layers[1], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.lr_model.layers[1].weights[0].shape, (2,1))
        self.assertEqual(self.lr_model.layers[1].activation, keras.src.activations.activations.sigmoid)

    def test_model_attributes(self):
        self.assertEqual(self.lr_model.loss, 'binary_crossentropy')
        self.assertIsInstance(self.lr_model.optimizer, keras.src.optimizers.rmsprop.RMSprop)
        self.assertEqual(self.lr_model.metrics_names[0], 'loss')
        # would like to check the other metric is accuracy, but seems hidden?...
        self.assertEqual(self.lr_model.metrics_names[1], 'compile_metrics')

    def test_model_names(self):
        self.assertEqual(self.lr_model.name, 'logistic_regression_model')
        self.assertEqual(self.lr_model.layers[0].name, 'lr_input')
        self.assertEqual(self.lr_model.layers[1].name, 'lr_output')


class test_get_neural_network_model(unittest.TestCase):

    def setUp(self):
        self.nn_model = get_neural_network_model()

    def test_model_layers(self):
        self.assertEqual(len(self.nn_model.layers), 3)
        self.assertIsInstance(self.nn_model.layers[0], keras.src.layers.core.input_layer.InputLayer)
        self.assertEqual(self.nn_model.layers[0].batch_shape, (None, 2))

        self.assertIsInstance(self.nn_model.layers[1], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.nn_model.layers[1].weights[0].shape, (2,64))
        self.assertEqual(self.nn_model.layers[1].activation, keras.src.activations.activations.relu)

        self.assertIsInstance(self.nn_model.layers[2], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.nn_model.layers[2].weights[0].shape, (64,1))
        self.assertEqual(self.nn_model.layers[2].activation, keras.src.activations.activations.sigmoid)

    def test_model_attributes(self):
        self.assertEqual(self.nn_model.loss, 'binary_crossentropy')
        self.assertIsInstance(self.nn_model.optimizer, keras.src.optimizers.rmsprop.RMSprop)
        self.assertEqual(self.nn_model.metrics_names[0], 'loss')
        # would like to check the other metric is accuracy, but seems hidden?...
        self.assertEqual(self.nn_model.metrics_names[1], 'compile_metrics')

    def test_model_names(self):
        self.assertEqual(self.nn_model.name, 'neural_network_model')
        self.assertEqual(self.nn_model.layers[0].name, 'nn_input')
        self.assertEqual(self.nn_model.layers[2].name, 'nn_output')


class test_get_more_powerful_model(unittest.TestCase):

    def setUp(self):
        self.deep_model = get_more_powerful_model()

    def test_model_layers(self):
        self.assertTrue(len(self.deep_model.layers) >= 4)
        self.assertIsInstance(self.deep_model.layers[0], keras.src.layers.core.input_layer.InputLayer)
        self.assertEqual(self.deep_model.layers[0].batch_shape, (None, 2))

        self.assertIsInstance(self.deep_model.layers[1], keras.src.layers.core.dense.Dense)
        #self.assertEqual(self.deep_model.layers[1].weights[0].shape, (2,64))
        self.assertEqual(self.deep_model.layers[1].activation, keras.src.activations.activations.relu)

        self.assertIsInstance(self.deep_model.layers[-1], keras.src.layers.core.dense.Dense)
        #self.assertEqual(self.deep_model.layers[-1].weights[0].shape, (64,1))
        self.assertEqual(self.deep_model.layers[-1].activation, keras.src.activations.activations.sigmoid)

    def test_model_attributes(self):
        self.assertEqual(self.deep_model.loss, 'binary_crossentropy')
        self.assertIsInstance(self.deep_model.optimizer, keras.src.optimizers.rmsprop.RMSprop)
        self.assertEqual(self.deep_model.metrics_names[0], 'loss')
        # would like to check the other metric is accuracy, but seems hidden?...
        self.assertEqual(self.deep_model.metrics_names[1], 'compile_metrics')

    def test_model_names(self):
        #self.assertEqual(self.deep_model.name, 'neural_network_model')
        #self.assertEqual(self.deep_model.layers[0].name, 'nn_input')
        #self.assertEqual(self.deep_model.layers[2].name, 'nn_output')
        pass
