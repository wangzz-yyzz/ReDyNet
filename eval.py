from logging import getLogger

import pandas as pd

import loss


class TrafficStateEvaluator:

    def __init__(self):
        super().__init__()
        self.metrics = ['MAE', 'RMSE', 'MAPE',
                        'masked_MAE', 'masked_RMSE', 'masked_MAPE']
        self.allowed_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE',
                                'masked_MAE', 'masked_MSE', 'masked_RMSE', 'masked_MAPE']
        self.mode = 'average'
        self.len_timeslots = 0
        self.result = {}
        self.intermediate_result = {}
        self._check_config()
        self._logger = getLogger()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise ValueError('the metric {} is not allowed in TrafficStateEvaluator'.format(str(metric)))

    def collect(self, batch):
        print('starting collect...')
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        y_true_origin = batch['y_true']  # tensor
        y_pred_origin = batch['y_pred']  # tensor
        features = y_true_origin.shape[-1]
        if y_true_origin.shape != y_pred_origin.shape:
            raise ValueError("batch['y_true'].shape is not equal to batch['y_pred'].shape")

        print("features: ", features)
        for feature in range(features):
            print("evaluating feature: ", feature)
            y_true = y_true_origin[..., feature]
            y_pred = y_pred_origin[..., feature]

            self.len_timeslots = y_true.shape[1]
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    self.intermediate_result[metric + '@' + str(i)] = []

            if self.mode.lower() == 'average':
                for i in range(1, self.len_timeslots + 1):
                    for metric in self.metrics:
                        self.intermediate_result[metric + '@' + str(i)].append(
                            self.eval_with_loss_func(metric)(y_pred[:, :i], y_true[:, :i]).item())

            elif self.mode.lower() == 'single':
                for i in range(1, self.len_timeslots + 1):
                    for metric in self.metrics:
                        self.intermediate_result[metric + '@' + str(i)].append(
                            self.eval_with_loss_func(metric)(y_pred[:, i - 1], y_true[:, i - 1]).item())
            else:
                raise ValueError(
                    'Error parameter evaluator_mode={}, please set `single` or `average`.'.format(self.mode))

            self.print_logs()

        print("eval avg")
        for i in range(1, self.len_timeslots + 1):
            for metric in self.metrics:
                self.intermediate_result[metric + '@' + str(i)] = []
        if self.mode.lower() == 'average':
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    self.intermediate_result[metric + '@' + str(i)].append(
                        self.eval_with_loss_func(metric)(y_pred_origin[:, :i], y_true_origin[:, :i]).item())
        elif self.mode.lower() == 'single':
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    self.intermediate_result[metric + '@' + str(i)].append(
                        self.eval_with_loss_func(metric)(y_pred_origin[:, i - 1], y_true_origin[:, i - 1]).item())
        else:
            raise ValueError(
                'Error parameter evaluator_mode={}, please set `single` or `average`.'.format(self.mode))
        self.print_logs()

    def evaluate(self):
        for i in range(1, self.len_timeslots + 1):
            for metric in self.metrics:
                self.result[metric + '@' + str(i)] = sum(self.intermediate_result[metric + '@' + str(i)]) / \
                                                     len(self.intermediate_result[metric + '@' + str(i)])
        return self.result

    def print_logs(self):
        self.evaluate()
        dataframe = {}
        for metric in self.metrics:
            dataframe[metric] = []
        for i in range(1, self.len_timeslots + 1):
            for metric in self.metrics:
                dataframe[metric].append(self.result[metric + '@' + str(i)])
        dataframe = pd.DataFrame(dataframe, index=range(1, self.len_timeslots + 1))
        print("\n" + str(dataframe))
        return dataframe

    def eval_with_loss_func(self, loss_func):
        if loss_func == 'masked_MAE':
            return loss.masked_mae_torch
        elif loss_func == 'masked_MSE':
            return loss.masked_mse_torch
        elif loss_func == 'masked_RMSE':
            return loss.masked_rmse_torch
        elif loss_func == 'masked_MAPE':
            return loss.masked_mape_torch
        elif loss_func == 'MAE':
            return loss.mae_torch
        elif loss_func == 'MSE':
            return loss.mse_torch
        elif loss_func == 'RMSE':
            return loss.rmse_torch
        elif loss_func == 'MAPE':
            return loss.mape_torch

    def clear(self):
        self.result = {}
        self.intermediate_result = {}
