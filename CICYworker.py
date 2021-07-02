"""
CICYworker to use with BOHB to learn all kind of hodge number
combinations.
"""
import os as os
from reg_acc import reg_acc
from callbacks import *
import logging
import tensorflow as tf
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from inception import inception_model
from hpbandster.core.worker import Worker
import json
tfk = tf.keras
logging.basicConfig(level=logging.INFO)


class CICYWorker(Worker):

    def __init__(self, cname, hname, dname, classification,
        train_ratio=[80, 10, 10], h=2, tails='', **kwargs):
        """CICYworker.

        Args:
            cname (str): path to config matrices
            hname (str): path to hodge numbers
            dname (str): path to direct products
            classification (int): 0-regression, 1-classification
            train_ratio (list, optional): train:val:test split.
                 Defaults to [80, 10, 10].
            h (int, optional): hodge number to learn.
                4-folds: -1-all, 0-h11, 1-h21, 2-h31, 3-h22.
                3-folds: -1-all, 0-h11, 1-h21.
                 Defaults to 2.
            tails (str, optional): path to tails. Defaults to ''.
        """
        super().__init__(**kwargs)

        configs = np.load(cname)
        hodge = np.load(hname)
        self.nfold = 3 if hodge.shape[1] < 3 else 4
        self.direct = np.load(dname)
        self.configs = configs[~self.direct]
        self.h = h
        if self.h < 0:
            self.hodge = hodge[~self.direct]
        else:
            self.hodge = np.expand_dims(hodge[~self.direct, h], -1)
        input_shape = tuple([*np.shape(self.configs), 1])
        self.configs = self.configs.reshape(input_shape)
        self.input_shape = input_shape[1:]
        # subtract hodge numbers to min hodge = 0
        self.min = np.min(self.hodge, axis=0)
        self.hodge -= self.min
        self.max = np.max(self.hodge, axis=0)

        if classification and self.h > -1:
            assert self.h > -1, "all hodge only works with reg currently"
            self.classification = classification
            self.num_classes = self.max[0]-self.min[0]
            y_values = tf.one_hot(self.hodge.reshape(
                (-1,)), depth=self.num_classes).numpy()
            print('Predicting h {} with classification and {} classes.'\
                .format(h, self.num_classes))
        else:
            # rescale numbers to interval [0,1]
            self.hodge = self.hodge/self.max
            self.num_classes = self.max
            self.classification = False
            self.num_hodges = np.shape(self.hodge)[1]
            print('Predicting {} hodge numbers with regression.'.format(
                self.num_hodges))
            y_values = self.hodge

        if np.sum(train_ratio) != 1:
            train_ratio = np.array(train_ratio)/np.sum(train_ratio)
            itrain = int(train_ratio[0]*len(self.hodge))
            ival = int(np.sum(train_ratio[0:2]*len(self.hodge)))

        # shuffle
        if tails:
            tail_mask = np.load(tails)
            tail_indices = np.where(tail_mask)[0]
            other_indices = np.where(~tail_mask)[0]
            np.random.shuffle(other_indices)
            indices = np.concatenate((other_indices, tail_indices))
        else:
            indices = np.arange(len(self.hodge))
            np.random.shuffle(indices)
        self.configs = self.configs[indices]
        y_values = y_values[indices]
        # make train val test split
        self.x_train, self.y_train = self.configs[:itrain], y_values[:itrain]
        self.x_val, self.y_val = self.configs[itrain:ival], y_values[itrain:ival]
        self.x_test, self.y_test = self.configs[ival:], y_values[ival:]

        print('There are {} train confs, {} val confs and {} \
            test confs.'.format(
            np.shape(self.x_train)[0], np.shape(
                self.x_val)[0], np.shape(self.x_test)[0]
        ))

        # verbose variable 0 - silent,  1 - tf output,
        # 2 - saves history to workdir
        self.verbose = 0

        # rescale for computing accuracy in reg
        if not self.classification:
            self.hodge_test = self.max * self.y_test + self.min
            self.hodge_val = self.max * self.y_val + self.min
            self.hodge_train = self.max * self.y_train + self.min

    def compute(self, config, budget, working_directory,
         *args, **kwargs):
        if self.classification:
            num_classes = self.num_classes
        else:
            num_classes = self.num_hodges
        model = inception_model(
            self.input_shape, num_classes, self.classification, config,
            reg_scales={'min': self.min, 'max': self.max})

        # make callback list
        patience = int(0.2*budget)
        reduce_lr = tfk.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy' if self.classification else 'accuracy',
            factor=0.5, patience=patience,
            verbose=1, mode='auto', min_delta=0.01,
            cooldown=0, min_lr=1e-6
        )
        callback_list = [reduce_lr]
        if self.verbose > 1:
            model.summary()
            if not self.classification:
                if self.num_hodges > 1:
                    reg_call_back = RegressionAccuracy
                else:
                    reg_call_back = RegressionSAccuracy
                #reg_acc_cb = [reg_call_back((self.x_val, self.y_val),
                #  self.min, self.max, 'val')]
                callback_list += [reg_call_back((self.x_test,
                    self.y_test), self.min, self.max, 'test')]
                callback_list += [reg_call_back((self.x_val,
                    self.y_val), self.min, self.max, 'val')]
            else:
                callback_list += [ClassAcc((self.x_test, self.y_test),
                     'test')]

        # fit the model with budget
        history = model.fit(
            self.x_train, self.y_train,
            validation_data=(self.x_val,
                self.y_val) if self.classification else None,
            epochs=int(budget), shuffle=True,
            verbose=1 if self.verbose else 0,
            batch_size=config['bSize'],
            callbacks=callback_list
        )

        # dump history if verbose = 2
        if self.verbose > 1:
            with open(os.path.join(working_directory,
                 'history.json'), 'w') as f:
                json.dump(str(history.history), f)
            model.save(os.path.join(
                working_directory, 'inception_model'))

        if self.classification:
            val_score = model.evaluate(
                self.x_val, self.y_val, verbose=0)
            test_score = model.evaluate(
                self.x_test, self.y_test, verbose=0)
            # loss, acc
            return ({
                    'loss': 1-val_score[1],
                    'info': {'validation accuracy': val_score[1],
                        'test accuracy': test_score[1],
                        'test loss': test_score[0],
                        'validation loss': val_score[0],
                        'number of parameters': model.count_params(),
                        }
                    })
        else:
            val_pred = model.predict(self.x_val)
            test_pred = model.predict(self.x_test)
            train_pred = model.predict(self.x_train)
            val_pred = self.max * val_pred + self.min
            test_pred = self.max * test_pred + self.min
            train_pred = self.max * train_pred + self.min
            val_acc = [float(reg_acc(
                self.hodge_val[:, i], val_pred[:, i]))
                    for i in range(self.num_hodges)]
            test_acc = [float(reg_acc(
                self.hodge_test[:, i], test_pred[:, i]))
                    for i in range(self.num_hodges)]
            train_acc = [float(reg_acc(
                self.hodge_train[:, i], train_pred[:, i]))
                    for i in range(self.num_hodges)]
            val_mean = float(np.mean(val_acc))
            test_mean = float(np.mean(test_acc))
            train_mean = float(np.mean(train_acc))
            return ({
                    'loss': 1-val_mean,
                    'info': {'test accuracy': test_acc,
                        'validation accuracy': val_acc,
                        'test mean': test_mean,
                        'train accuracy': train_acc,
                        'train mean': train_mean,
                        'validation mean': val_mean,
                        'number of parameters': model.count_params(),
                        }
                    })

    @staticmethod
    def get_configspace():

        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter('lr',
            lower=1e-6, upper=1e-2, default_value='1e-3', log=True)
        optimizer = CSH.CategoricalHyperparameter('optimizer',
                                                  ['Adam', 'SGD'])
        act_d = CSH.CategoricalHyperparameter('act_d',
            ['relu', 'tanh', 'sigmoid', 'elu', 'selu'])
        sgd_momentum = CSH.UniformFloatHyperparameter('SGDmoment',
            lower=0.0, upper=0.99, default_value=0.9, log=False)
        hunits = CSH.UniformIntegerHyperparameter('bunits',
            lower=2**4, upper=2**7, default_value=32)
        nunits = CSH.UniformIntegerHyperparameter('dunits',
            lower=2**4, upper=2**7, default_value=32)
        nBlock = CSH.UniformIntegerHyperparameter('nBlock',
            lower=1, upper=5, default_value=3, log=False)
        nDense = CSH.UniformIntegerHyperparameter('nDense',
            lower=0, upper=3, default_value=1, log=False)
        # nDense2 = CSH.UniformIntegerHyperparameter('nDense2',
        #                lower=0, upper=1, default_value=1, log=False)
        bSize = CSH.UniformIntegerHyperparameter('bSize',
            lower=2**5, upper=2**10, default_value=128)
        l1 = CSH.UniformFloatHyperparameter('l1',
            lower=0.000001, upper=0.01, default_value=0.0001, log=True)
        l2 = CSH.UniformFloatHyperparameter('l2',
            lower=0.000001, upper=0.01, default_value=0.0001, log=True)
        dropout = CSH.UniformFloatHyperparameter('dropout',
            lower=0., upper=0.5, default_value=0.2, log=False)
        auxloss = CSH.CategoricalHyperparameter('auxloss',
            [False])  # , True
        linear = CSH.CategoricalHyperparameter('linear',
                                               [False, True])
        alpha = CSH.UniformFloatHyperparameter('alpha',
            lower=0.000001, upper=0.001, default_value=0.0001, log=True)
        cs.add_hyperparameters([lr, optimizer, sgd_momentum, auxloss,
                                hunits, nunits, nBlock, nDense, bSize,
                                act_d, l1, l2, dropout, linear, alpha])

        # The hyperparameter sgd_momentum will be used, 
        # if the configuration contains 'SGD' as optimizer.
        cond1 = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond1)
        cond2 = CS.EqualsCondition(alpha, linear, True)
        cs.add_condition(cond2)
        cond3 = CS.GreaterThanCondition(auxloss, nDense, 1)
        cs.add_condition(cond3)
        return cs


if __name__ == "__main__":
    # this is 4fold
    worker = CICYWorker('data/conf.npy', 'data/hodge.npy',
                        'data/direct.npy', True, run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, working_directory='.')
    print(res)
