import time
import torch
import torch.utils.data as Data
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from data_loader import load_data, snp_to_matrix
from model import PCAGP
from evaluate import metrics_sklearn
from config import BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model():
    X, Y = load_data()
    kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
    fold_results = []
    loss_func = torch.nn.HuberLoss()
    start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        print(f'Fold {fold + 1}')
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        X_train_t = snp_to_matrix(torch.from_numpy(X_train.astype(np.float32)))
        X_test_t = snp_to_matrix(torch.from_numpy(X_test.astype(np.float32)))

        y_train_t = torch.from_numpy(Y_train.astype(np.float32)).unsqueeze(1)
        y_test_t = torch.from_numpy(Y_test.astype(np.float32)).unsqueeze(1)

        train_data = Data.TensorDataset(X_train_t, y_train_t)
        train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        model = PCAGP(X_train_t.shape[2], X_train_t.shape[3]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_r = -1
        best_model_path = f"PCAGP_fold{fold + 1}_TKW_best.pt"

        for epoch in range(EPOCHS):
            model.train()
            avg_loss = []
            for b_x, b_y in train_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                output = model(b_x)
                loss = loss_func(output.squeeze(), b_y.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.append(loss.item())

            model.eval()
            with torch.no_grad():
                y_pred = model(X_test_t.to(device)).cpu().numpy()
                r, mse = metrics_sklearn(y_test_t.numpy(), y_pred)

                if r > best_r:
                    torch.save(model.state_dict(), best_model_path)
                    best_r = r

            print(f"Epoch {epoch + 1}, Train Loss: {np.mean(avg_loss):.8f}, R: {r:.4f}, MSE: {mse:.8f}")

        model.load_state_dict(torch.load(best_model_path,weights_only=False))
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_t.to(device)).cpu().numpy()
        r, mse = metrics_sklearn(y_test_t.numpy(), y_pred)
        fold_results.append((fold + 1, r, mse))
        print(f"Fold {fold + 1} - Best R: {r:.4f}, MSE: {mse:.4f}")

    results_df = pd.DataFrame(fold_results, columns=['Fold', 'R', 'MSE'])
    avg_r = results_df['R'].mean()
    avg_mse = results_df['MSE'].mean()
    results_df.to_csv('results_GL.csv', index=False)
    print(f'Average R: {avg_r:.4f}, Average MSE: {avg_mse:.4f}')
    total_time = time.time() - start_time
    print(f'Total time: {total_time:.2f} s')

if __name__ == "__main__":
    train_model()