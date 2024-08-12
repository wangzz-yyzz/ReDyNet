import argparse
import json
import time

import nni
import torch.backends.cudnn
import torch.optim as optim

from eval import TrafficStateEvaluator
from loss import mae_torch, rmse_torch, masked_mape_torch
from model.model_selector import ModelSelector
from raw_data.dataset import *


class Trainer:
    def __init__(self, args, model, data_feature):
        self.saved = args.saved
        self.cache_dir = args.saved_model_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = args.num_epochs
        self.log_every = args.log_every

        self.model = model
        self.data_feature = data_feature

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, eps=args.lr_epsilon,
                                    betas=(args.lr_beta1, args.lr_beta2), weight_decay=args.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)

        # early stop
        self.patience = args.num_epochs
        self.use_early_stop = True

        # test
        self.output_dim = args.output_dim
        self._scaler = data_feature['scaler']
        self.evaluator = TrafficStateEvaluator()

    def train(self, train_dataloader, eval_dataloader):
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []

        for epoch_idx in range(self.num_epochs):
            start_time = time.time()
            train_losses = []
            self.model.train()
            for batch in train_dataloader:
                self.optimizer.zero_grad()
                batch.to_tensor(self.device)
                loss = self.model.calculate_loss_train(batch)
                train_losses.append(loss.item())

                loss.backward()
                self.optimizer.step()
            train_loss = np.mean(train_losses)
            t1 = time.time()
            train_time.append(t1 - start_time)

            # eval
            t2 = time.time()
            with torch.no_grad():
                self.model.eval()
                y_preds = []
                y_truths = []
                for batch in eval_dataloader:
                    batch.to_tensor(self.device)

                    output = self.model.predict(batch)
                    y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
                    y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
                    y_truths.append(y_true.cpu())
                    y_preds.append(y_pred.cpu())

                y_preds = torch.Tensor(np.concatenate(y_preds, axis=0))
                y_truths = torch.Tensor(np.concatenate(y_truths, axis=0))

                mae_loss = mae_torch(y_preds, y_truths).item()
                rmse_loss = rmse_torch(y_preds, y_truths).item()
                mape_loss = masked_mape_torch(y_preds, y_truths).item()
                val_loss = mae_loss + rmse_loss + mape_loss
            end_time = time.time()
            eval_time.append(end_time - t2)

            self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = ('Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, mae: {:.4f}, rmse: {:.4f}, '
                           'mape: {:.4f}, lr: {:.6f}, {:.2f}s'). \
                    format(epoch_idx, self.num_epochs, train_loss, val_loss, mae_loss, rmse_loss, mape_loss, log_lr,
                           (end_time - start_time))
                print(message)
                if args.nni:
                    nni.report_intermediate_result({'default': mae_loss,
                                                    'rmse': rmse_loss, 'mape': mape_loss})

            if val_loss < min_val_loss and epoch_idx % self.log_every == 0:
                wait = 0
                if self.saved:
                    self.save_model_with_epoch(epoch_idx)
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    print('Early stopping at epoch: %d' % epoch_idx)
                    break

        time_message = 'Average train time is {:.3f}s, average eval time is {:.3f}s'. \
            format(sum(train_time) / len(train_time), sum(eval_time) / len(eval_time))
        print(time_message)
        if args.nni:
            nni.report_final_result({'default': min_val_loss, 'rmse': min_val_loss, 'mape': min_val_loss})

        return min_val_loss, best_epoch

    def save_model_with_epoch(self, epoch):
        config = dict()
        config['model_state_dict'] = self.model.state_dict()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + args.dataset + '/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        save_model_name = model_path + args.dataset + '_epoch%d.tar' % epoch
        torch.save(config, save_model_name)
        print("Saved model at Epoch {}".format(epoch))

    def evaluate(self, test_dataloader, best_epoch=None):
        if best_epoch is not None:
            print('Load model from epoch %d' % best_epoch)
            model_path = self.cache_dir + '/' + args.dataset + '/'
            save_model_name = model_path + args.dataset + '_epoch%d.tar' % best_epoch
            checkpoint = torch.load(save_model_name, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        with torch.no_grad():
            self.model.eval()
            y_truths = []
            y_preds = []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                output = self.model.predict(batch)
                y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
                y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())

            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)
            outputs = {'prediction': y_preds, 'truth': y_truths}

            self.evaluator.clear()
            self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--dataset", type=str, default='BJMETRO', help="Dataset.", )
    parser.add_argument("--model_name", type=str, default='ReDyNet', help="Model name.", )
    parser.add_argument("--input_window", type=int, default=4, help="Input length.", )
    parser.add_argument("--output_window", type=int, default=4, help="Output length.", )
    parser.add_argument("--train_rate", type=int, default=0.7, help="Train rate.", )
    parser.add_argument("--valid_rate", type=int, default=0.1, help="Validation rate.", )
    parser.add_argument("--test_rate", type=int, default=0.2, help="Test rate.", )
    parser.add_argument("--nni", type=bool, default=False, help="Whether use nni or not.", )

    parser.add_argument("--scaler_type", type=str, default='standard', help="Scaler type.", )
    parser.add_argument("--output_dim", type=int, default=2, help="Output dim.", )  # ?

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.", )
    parser.add_argument("--pad_with_last_sample", type=bool, default=False, help=".", )
    parser.add_argument("--num_workers", type=int, default=0, help="Num workers.", )
    # adj_mx
    parser.add_argument("--bidir_adj_mx", type=bool, default=True)
    parser.add_argument("--set_weight_link_or_dist", type=str, default='link')
    parser.add_argument("--init_weight_inf_or_zero", type=str, default='zero')
    parser.add_argument("--calculate_weight_adj", type=bool, default=False)
    parser.add_argument("--weight_adj_epsilon", type=float, default=0.1, help="weight_adj_epsilon.", )
    # external
    parser.add_argument("--load_external", type=bool, default=True, help="Whether load external or not.", )
    parser.add_argument("--ext_dim", type=int, default=0, help="External dim.", )
    parser.add_argument("--add_day_in_week", type=bool, default=True, help="Whether add day in week or not.", )
    parser.add_argument("--add_time_in_day", type=bool, default=True, help="Whether add time in day or not.", )
    parser.add_argument("--add_weather", type=bool, default=True, help="Whether add weather or not.", )
    parser.add_argument("--normal_external", type=bool, default=True, help="Whether normal external or not.", )
    parser.add_argument("--ext_scaler_type", type=str, default='standard', help="External scaler type.", )

    # train
    parser.add_argument("--saved", type=bool, default=True, help="Whether save model or not.", )
    parser.add_argument("--saved_model_dir", type=str, default='saved/', help="Saved model dir.", )

    parser.add_argument("--num_epochs", type=int, default=300, help="Train epochs.", )
    parser.add_argument("--log_every", type=int, default=1, help="Log every x epochs.", )
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Initial learning rate.", )
    parser.add_argument("--lr_epsilon", type=float, default=1e-8, help="Learning rate epsilon.", )
    parser.add_argument("--lr_beta1", type=float, default=0.9, help="Learning rate beta1.", )
    parser.add_argument("--lr_beta2", type=float, default=0.999, help="Learning rate beta2.", )
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay rate.", )

    parser.add_argument("--step_size", type=int, default=20, help="Learning rate schedule step.", )
    parser.add_argument("--gamma", type=float, default=0.8, help="Learning rate scheduler gamma.", )
    args = parser.parse_args()

    set_seeds(0)

    if args.model_name == 'ReDyNet':
        config = {}
        config_path = 'model/MyModel/' + args.dataset + '.json'
        with open(config_path, 'r') as f:
            x = json.load(f)
            for key in x:
                if key not in config:
                    config[key] = x[key]
        if "lr" in config:
            args.learning_rate = config["lr"]
            print("Learning rate is set to %.4f" % args.learning_rate)
        if "batch_size" in config:
            args.batch_size = config["batch_size"]
            print("Batch size is set to %d" % args.batch_size)

    dataset = Dataset(args)

    train_dataloader, eval_dataloader, test_dataloader = dataset.get_data()
    data_feature = dataset.get_data_feature()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ModelSelector(args.model_name, data_feature, args.dataset).get_model()
    model = model.to(device)
    print("Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    trainer = Trainer(args, model, data_feature)
    _, best_epoch = trainer.train(train_dataloader, eval_dataloader)
    print('Best epoch is %d' % best_epoch)
    trainer.evaluate(test_dataloader, best_epoch)
