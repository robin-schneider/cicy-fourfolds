"""
Here the tensorflow inception models for classification of a single
hodge number and regression of one or all hodge numbers are created.
"""

import tensorflow as tf
from tensorflow.python.keras.engine import training
from reg_acc import *
tfk = tf.keras

def lmse(alpha = 0.1, reg_scales = None):
    r"""Computes the 'linear mean squared error' that is
    mean squared + absolute value of the linear relationship
    relating the four hodge numbers.

    Args:
        alpha (float, optional): factor in front of linear term.
            Defaults to 0.1.
        reg_scales (dict, optional): dict containing min and max values
            for all hodge numbers. Defaults to None.

    Returns:
        function: loss function for tf model
    """
    if reg_scales is not None:
        hmin = tf.cast(reg_scales['min'], dtype=tf.float32)
        hmax = tf.cast(reg_scales['max'], dtype=tf.float32)
    else:
        hmin = 0
        hmax = 1
    alpha = tf.cast(alpha, dtype=tf.float32)
    def linear_mean_squared_error(y_true, y_pred):
        mse = tfk.losses.mean_squared_error(y_true, y_pred)
        linear = -4. * (hmax[0]*y_pred[:,0]+hmin[0]) + \
                    2. * (hmax[1]*y_pred[:,1]+hmin[1]) - \
                    4. * (hmax[2]*y_pred[:,2]+hmin[2]) + \
                    (hmax[3] * y_pred[:,3] +hmin[3]) - 44.
        # take mse of linear with mean value = 0.
        loss = mse + alpha * tf.square(tf.abs(linear))
        return loss

    return linear_mean_squared_error

def conv2d_bn(x, filters, num_row, num_col, act='relu', padding='same', 
                  strides=(1,1), name='None', l1 = 0.0001, l2 = 0.001):
    r"""convolutional block

    Args:
        x (tf.tensor): tensor parsing through the nn
        filters (int): number of filters
        num_row (int): number of rows
        num_col (int): number of cols
        act (str, optional): activation function. Defaults to 'relu'.
        padding (str, optional): padding. Defaults to 'same'.
        strides (tuple, optional): strides. Defaults to (1,1).
        name (str, optional): name. Defaults to 'None'.
        l1 (float, optional): l1 reg. Defaults to 0.0001.
        l2 (float, optional): l2 reg. Defaults to 0.001.

    Returns:
        tf.tensor: data
    """
    x = tfk.layers.Conv2D(filters, (num_row, num_col), strides=strides,
        padding=padding, use_bias=False, name=name+'_conv',
        kernel_regularizer=tfk.regularizers.l1_l2(l1=l1, l2=l2))(x)
    #x = tfk.layers.BatchNormalization(
    # axis=-1, scale=False, name=name+'_bn')(x)
    x = tfk.layers.Activation(act, name=name)(x)
    return x

def inception_model(input_shape, num_classes,
        classification, config, reg_scales=None):
    r"""Builds and returns tensorflow model for regression
    and classification task.

    Note:
        This is not the cleanest code and shoul probably be broken up
        into two modules.

    Args:
        input_shape (tuple): shape of configuration matrix, i.e

        num_classes (int): if classification number of classes
                           if regression predict 1 or 4 hodge numbers
        classification (bool): determines whether regression or 
            classification model
        config (dict): collection of hyperparameters in a dict
        reg_scales (dict, optional): contains min and max for when
            the hodge numbers got rescaled. Defaults to None.

    Returns:
        tfk.model: tensorflow model
    """
    inputs = tfk.layers.Input(shape = input_shape, name='input')
    proj, poly, _ = input_shape
    x = inputs
    for i in range(config['nBlock']):
        # add less units
        bunits = 16 if i == config['nBlock']-1 else config['bunits']
        branchPrx1 = conv2d_bn(x, bunits, proj, 1, l1=config['l1'],
            l2=config['l2'], name = 'b'+str(i)+'_'+str(proj)+'x1')
        branch1xPo = conv2d_bn(x, bunits, 1, poly, l1=config['l1'],
            l2=config['l2'], name = 'b'+str(i)+'_1x'+str(poly))
        x = tfk.layers.concatenate([branchPrx1, branch1xPo],
            axis=-1, name='block'+str(i))
        # moved bn to after concat to make consistent
        #  with CICY threefold Inception paper
        x = tfk.layers.BatchNormalization(
            axis=-1, scale=False, name=str(i)+'_bn')(x)
        x = tfk.layers.Dropout(
            rate=config['dropout'], name='dr'+str(i))(x)
        

    if config['nDense'] > 1 and config['auxloss']:
        fa = tfk.layers.Flatten(name='aflatten')(x)
        if classification:
            aux_out = tfk.layers.Dense(num_classes,
                activation='softmax', name='auxout',
                kernel_regularizer=tfk.regularizers.l1_l2(
                    l1=config['l1'], l2=config['l2']))(fa)
        else:
            aux_out = tfk.layers.Dense(num_classes, name='auxout',
                kernel_regularizer=tfk.regularizers.l1_l2(
                    l1=config['l1'], l2=config['l2']))(fa)

    x = tfk.layers.Flatten(name='flatten')(x)
    for i in range(config['nDense']):
        x = tfk.layers.Dense(config['dunits'],
            activation=config['act_d'], name='dense'+str(i),
            kernel_regularizer=tfk.regularizers.l1_l2(
                l1=config['l1'], l2=config['l2']))(x)
        x = tfk.layers.Dropout(rate=config['dropout'],
            name='dr'+str(config['nBlock']+i))(x)
    
    if classification:
        x = tfk.layers.Dense(num_classes,
            activation='softmax', name='prediction',
            kernel_regularizer=tfk.regularizers.l1_l2(
                l1=config['l1'], l2=config['l2']))(x)
    else:
        x = tfk.layers.Dense(num_classes, name='prediction',
            kernel_regularizer=tfk.regularizers.l1_l2(
                l1=config['l1'], l2=config['l2']))(x)

    if config['nDense'] > 1 and config['auxloss']:
        model = training.Model(inputs, [x, aux_out], name='inception')
    else:
        model = training.Model(inputs, x, name='inception')
    model.build(input_shape)
    if config['optimizer'] == 'Adam':
        optimizer = tfk.optimizers.Adam(learning_rate = config['lr'])
    else:
        optimizer = tfk.optimizers.SGD(learning_rate = config['lr'],
            momentum = config['SGDmoment'])
    
    #take appropiate loss
    if config['nDense'] > 1 and config['auxloss']:
        if classification:
            loss = ["categorical_crossentropy" for _ in range(2)]
        elif num_classes > 2:
            loss = [lmse(config['alpha'], reg_scales) for _ in range(2)]
        else:
            loss = ["mean_squared_error" for _ in range(2)]
    else:
        if classification:
            loss = "categorical_crossentropy"
        elif num_classes > 2 and config['linear']:
            loss = lmse(config['alpha'], reg_scales)
        else:
            loss = "mean_squared_error"

    if classification:
        model.compile(loss=loss,
             optimizer=optimizer, metrics=["accuracy"])
    else:
        if num_classes == 1:
            # single hodge
            model.compile(loss=loss, optimizer=optimizer,
                 metrics=[reg_acc_scaled(reg_scales)])
        elif num_classes == 2:
            # all hodge three folds
            model.compile(loss=loss, optimizer=optimizer)
            #, metrics=["mean_squared_error"] 
            # + [hi_reg_acc(i) for i in range(2)])
        else:
            # all hodge four folds
            model.compile(loss=loss, optimizer=optimizer)
            # using the hi_acc metric slows training too much
            #, metrics=["mean_squared_error"] 
            # + [hi_reg_acc(i) for i in range(4)])
    return model