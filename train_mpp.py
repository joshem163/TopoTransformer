
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
# from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from module import *
from models import DualTransformerWithMLPClassifier

from sklearn.model_selection import train_test_split
from data_loader_MPP import load_dataset,stat

cls_criterion = torch.nn.BCEWithLogitsLoss()



def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # Unpack the batch and move each tensor to the device
        X1_batch, X2_batch, X3_batch, y_batch = [data.to(device) for data in batch]

        # Check for any conditions you want to handle
        if X1_batch.shape[0] == 1:
            continue

        pred = model(X1_batch, X2_batch, X3_batch)  # Pass all input tensors to the model
        optimizer.zero_grad()

        # Compute loss and ignore NaN targets
        is_labeled = y_batch == y_batch  # Avoid NaNs
        loss = cls_criterion(pred.to(torch.float32)[is_labeled], y_batch.to(torch.float32)[is_labeled])

        loss.backward()
        optimizer.step()


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # Unpack the batch and move each tensor to the device
        X1_batch, X2_batch, X3_batch, y_batch = [data.to(device) for data in batch]

        if X1_batch.shape[0] == 1:
            continue

        with torch.no_grad():
            pred = model(X1_batch, X2_batch, X3_batch)

        y_true.append(y_batch.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)


def main():
    parser = argparse.ArgumentParser(description='experiment')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='sider')#BBBP, bace,tox21,toxcast,HIV,clintox,sider
    parser.add_argument('--filename', type=str, default='resultBBBP')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--gap_pmeter', type=int, default=2)
    parser.add_argument('--head', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    f1, f2, f3, Label = load_dataset(args.dataset)

    # Convert to PyTorch tensors
    X1 = torch.tensor(f1, dtype=torch.float32)
    X2 = torch.tensor(f2, dtype=torch.float32)
    X3 = torch.tensor(f3, dtype=torch.float32)
    y = torch.tensor(Label, dtype=torch.long)
    if args.dataset in ['toxcast', 'tox21']:
        y[y == -1] = 0

    # Extract dataset details
    num_samples = len(X1)
    num_timesteps_1 = len(X1[0])
    num_timesteps_2 = len(X2[0])
    num_features1 = len(X1[0][0])
    num_features2 = len(X2[0][0])

    num_features3 = len(X3[0])
    num_classes = len(np.unique(y))
    num_task = len(y[0])

    X1 = min_max_scaling(X1, feature_range=(0, 1))
    X2 = min_max_scaling(X2, feature_range=(0, 1))
    X3 = X3

    # Define input and output dimensions
    hidden_dim = args.hidden_channels
    output_dim = num_classes
    n_heads = args.head
    n_layers = args.num_layers

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    indices = np.arange(num_samples)
    auc = []
    new_seed = 23
    for run in range(args.runs):
        if args.dataset in ['toxcast']:
        # Split into train (80%) and temporary (20%)
            train_idx, temp_idx = train_test_split(indices, test_size=0.2,random_state=42+run)

            # Split temporary set into validation (10%) and test (10%)
            valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5,random_state=42+run)
            y_train, y_test = y[train_idx], y[test_idx]

            if not validate_test_set(y_test):
                print("Test set is invalid, reshuffling and retrying...")
                # Repeat the splitting until the test set is valid
                while not validate_test_set(y_test):
                    new_seed += 1
                    train_idx, temp_idx = train_test_split(indices, test_size=0.2,random_state=new_seed)

                    # Split temporary set into validation (10%) and test (10%)
                    valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5,random_state=new_seed)
                    y_train, y_test = y[train_idx], y[test_idx]
        elif args.dataset in ['BBBP','bace']:
            train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42 + run,stratify=y)

            # Split temporary set into validation (10%) and test (10%)
            valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42 + run,stratify=y[temp_idx])
        elif args.dataset in ['HIV','tox21','sider','clintox']:
            train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42 + run)

            # Split temporary set into validation (10%) and test (10%)
            valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42 + run)
        else:
            print('train test spliting is not available for this dataset')


        # Split data based on fixed indices
        X1_train, X1_val, X1_test = X1[train_idx], X1[valid_idx], X1[test_idx]
        X2_train, X2_val, X2_test = X2[train_idx], X2[valid_idx], X2[test_idx]
        X3_train, X3_val, X3_test = X3[train_idx], X3[valid_idx], X3[test_idx]
        y_train, y_val, y_test = y[train_idx], y[valid_idx], y[test_idx]

        # Create DataLoaders
        train_data = TensorDataset(X1_train, X2_train, X3_train, y_train)
        val_data = TensorDataset(X1_val, X2_val, X3_val, y_val)
        test_data = TensorDataset(X1_test, X2_test, X3_test, y_test)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        valid_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # Initialize model, criterion, and optimizer
        model = DualTransformerWithMLPClassifier(num_features1, num_features2, num_features3, hidden_dim, num_task,
                                                 n_heads, n_layers, num_timesteps_1, num_timesteps_2).to(device)

        ### automatic evaluator. takes dataset name as input
        evaluator = AUCEvaluator()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # optimizer = optim.Adadelta(model.parameters(), weight_decay=1e-2)

        valid_curve = []
        test_curve = []
        train_curve = []

        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}".format(epoch))
            print('Training...')
            train(model, device, train_loader, optimizer)

            print('Evaluating...')
            train_perf = eval(model, device, train_loader, evaluator)
            valid_perf = eval(model, device, valid_loader, evaluator)
            test_perf = eval(model, device, test_loader, evaluator)

            #print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

            train_curve.append(train_perf["auc"])
            valid_curve.append(valid_perf["auc"])
            test_curve.append(test_perf["auc"])

        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
        test_acc = max(test_curve)
        auc.append(test_acc)

        print('Finished training! run', run)
        print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
        print('Test score: {}'.format(test_acc))

        if not args.filename == '':
            torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                        'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)

    stat(auc, 'AUC')


if __name__ == "__main__":
    main()
