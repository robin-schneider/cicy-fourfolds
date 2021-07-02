"""
Some custom metric function for tensorflow models.
"""
import tensorflow as tf


def hi_reg_acc(h):
    r"""Computes accuracy of a single hodge number for regression model,
    which predicts all four hodge numbers.

    NOTE:
        This increases computational time significantly.

    Args:
        h (int): hodge number to compute acc of

    Returns:
        fn: python function
    """
    # metrics are not allowed to have the same name
    def h_acc_11(y_true, y_pred):
        return tf.reduce_mean(tf.cast(tf.math.equal(tf.math.round(
            y_pred[:, 0]), y_true[:, 0]), 'float32'))

    def h_acc_21(y_true, y_pred):
        return tf.reduce_mean(tf.cast(tf.math.equal(tf.math.round(
            y_pred[:, 1]), y_true[:, 1]), 'float32'))

    def h_acc_31(y_true, y_pred):
        return tf.reduce_mean(tf.cast(tf.math.equal(tf.math.round(
            y_pred[:, 2]), y_true[:, 2]), 'float32'))

    def h_acc_22(y_true, y_pred):
        return tf.reduce_mean(tf.cast(tf.math.equal(tf.math.round(
            y_pred[:, 3]), y_true[:, 3]), 'float32'))
    if h == 0:
        return h_acc_11
    elif h == 1:
        return h_acc_21
    elif h == 2:
        return h_acc_31
    elif h == 3:
        return h_acc_22


def reg_acc_scaled(scaled):
    """Computes accuracy of a regression model with scaled hodge values.

    Args:
        scaled (dict): includes 'min' and 'max' hodge number.

    Returns:
        fn: python function
    """
    if scaled is not None:
        hmin = tf.cast(scaled['min'], dtype=tf.float32)
        hmax = tf.cast(scaled['max'], dtype=tf.float32)
    else:
        hmin = 0.
        hmax = 1.

    def accuracy(y_true, y_pred):
        # for single hodge number
        pred = hmax*y_pred+hmin
        true = hmax*y_true+hmin
        return tf.reduce_mean(tf.cast(tf.math.equal(
            tf.math.round(pred), tf.math.round(true)), 'float32'))

    return accuracy


def reg_acc(y_true, y_pred):
    # for single hodge number
    return tf.reduce_mean(tf.cast(tf.math.equal(
        tf.math.round(y_pred), tf.math.round(y_true)), 'float32'))
