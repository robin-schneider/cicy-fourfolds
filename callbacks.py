"""
Some custom callbacks for tensorflow models.
"""
import numpy as np
import tensorflow as tf
tfk = tf.keras


class RegressionSAccuracy(tfk.callbacks.Callback):
    """Compute accuracy for a single hodge number of a regression model
    """

    def __init__(self, validation_data, lowest, highest, name):
        super(RegressionSAccuracy, self).__init__()
        self.X_val, self.y_val = validation_data
        self.lowest = lowest
        self.highest = highest
        self.y_val = self.y_val*self.highest+self.lowest
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        # might have to batch this
        y_pred = self.highest*self.model.predict(self.X_val)+self.lowest
        h_acc = self.compute_acc(y_pred, self.y_val)
        logs[self.name+'_acc'] = h_acc.numpy().tolist()
        print('\n '+self.name+' acc: {}.'.format(h_acc))

    def compute_acc(self, y_pred, y_true):
        return tf.reduce_mean(tf.cast(tf.math.equal(
            tf.math.round(y_pred), tf.math.round(y_true)), 'float32'))


class ClassAcc(tfk.callbacks.Callback):
    def __init__(self, test_data, name='test'):
        self.test_data = test_data
        self.name = name

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        scores = self.model.evaluate(x, y, verbose=False)
        print('\nTesting loss: {}, accuracy: {}\n'.format(
            scores[0], scores[1]))
        logs[self.name+'_acc'] = float(scores[1])
        logs[self.name+'_loss'] = float(scores[0])


class RegressionAccuracy(tfk.callbacks.Callback):
    """Compute accuracy for all hodge numbers of a regression model
    """

    def __init__(self, validation_data, lowest, highest, name):
        super(RegressionAccuracy, self).__init__()
        self.X_val, self.y_val = validation_data
        self.lowest = lowest
        self.highest = highest
        self.y_val = self.y_val*self.highest+self.lowest
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        # might have to batch this
        y_pred = self.highest*self.model.predict(self.X_val)+self.lowest
        y_true = self.y_val

        h11_acc = self.compute_acc(y_pred[:, 0], y_true[:, 0])
        h21_acc = self.compute_acc(y_pred[:, 1], y_true[:, 1])
        h31_acc = self.compute_acc(y_pred[:, 2], y_true[:, 2])
        h22_acc = self.compute_acc(y_pred[:, 3], y_true[:, 3])

        logs['h11_acc_'+self.name] = h11_acc.numpy().tolist()
        logs['h21_acc'+self.name] = h21_acc.numpy().tolist()
        logs['h31_acc'+self.name] = h31_acc.numpy().tolist()
        logs['h22_acc'+self.name] = h22_acc.numpy().tolist()
        print('\n'+self.name+' h11acc: {}, h21acc: {}, h31acc: {}, \
            h22acc: {}.'.format(h11_acc, h21_acc, h31_acc, h22_acc))

    def compute_acc(self, y_pred, y_true):
        return tf.reduce_mean(tf.cast(tf.math.equal(
            tf.math.round(y_pred), tf.math.round(y_true)), 'float32'))
