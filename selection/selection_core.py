#!/usr/bin/env python3
"""TOPMAX selection module: learn one weight per trait with mini-batch AdamW.

Weights are learned from the predicted and observed phenotypes of the training samples, as in TOP, but each
update uses a mini-batch of samples instead of the full sample-by-sample matrix.
"""

import time

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Helpers
def set_seed(seed=42, has_gpu=False):
    np.random.seed(seed)
    if has_gpu:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def prePro(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    value = (data - mean) / (std + 1e-10)
    return {'Mean': mean, 'SD': std, 'value': value}

def scale_z(x, mean_vec, sd_vec):
    x = np.asarray(x, dtype=float)
    mean_vec = np.asarray(mean_vec, dtype=float)
    sd_vec = np.asarray(sd_vec, dtype=float)
    sd_vec = np.where((sd_vec == 0) | np.isnan(sd_vec), 1.0, sd_vec)
    return (x - mean_vec) / (sd_vec + 1e-10)

# ------------------------------------------------------------
# AdamW optimizer, GPU (PyTorch)
def optimize_weights_gpu(Pre_train, Obs_train, batch_size, n_epochs, lr_init,
                         weight_decay, beta1, beta2, tol, patience, use_adaptive_stop,
                         device, verbose=True):
    import torch
    n_samples, n_features = Pre_train.shape
    Pre_t = torch.from_numpy(Pre_train).float().to(device)
    Obs_t = torch.from_numpy(Obs_train).float().to(device)
    Dis_t = torch.abs(Pre_t - Obs_t)

    w = torch.zeros(n_features, dtype=torch.float32, device=device)
    m = torch.zeros(n_features, device=device)
    v = torch.zeros(n_features, device=device)
    t = 0
    loss_history = []
    w_old = w.clone()
    converged_count = 0

    for epoch in range(n_epochs):
        lr = lr_init * (0.01 + 0.99 * (1 + np.cos(np.pi * epoch / n_epochs)) / 2)
        indices = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            bs = len(batch_indices)
            Obs_batch = Obs_t[batch_indices]
            Dis_batch = Dis_t[batch_indices]
            Pre_batch = Pre_t[batch_indices]

            a_M = torch.abs(Pre_batch.unsqueeze(0) - Obs_batch.unsqueeze(1))
            pi_M = torch.einsum('ijk,k->ij', a_M, w)
            pi_M = pi_M - pi_M.max(dim=1, keepdim=True)[0]
            proi = torch.exp(pi_M)
            proi = proi / proi.sum(dim=1, keepdim=True)

            diag_idx = torch.arange(bs, device=device)
            diagonal_probs = proi[diag_idx, diag_idx]
            loss = -torch.mean(torch.log(diagonal_probs + 1e-10))

            expected_dist = torch.einsum('ij,ijk->ik', proi, a_M)
            grad = torch.mean(Dis_batch - expected_dist, axis=0)

            t += 1
            g_t = grad
            m = beta1 * m + (1 - beta1) * g_t
            v = beta2 * v + (1 - beta2) * (g_t ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            w = w + lr * (m_hat / (torch.sqrt(v_hat) + 1e-8) + weight_decay * w)

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if use_adaptive_stop:
            w_diff_inf = torch.max(torch.abs(w - w_old)).item()
            if w_diff_inf < tol:
                converged_count += 1
                if verbose and (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}, Loss: {avg_loss:.6f}, ||w||: {torch.norm(w).item():.4f}, "
                          f"Δw∞: {w_diff_inf:.2e} [{converged_count}/{patience}]")
                if converged_count >= patience:
                    print(f"Converged at epoch {epoch+1}, max |dw| = {w_diff_inf:.2e} < {tol:.2e}")
                    break
            else:
                converged_count = 0
                if verbose and (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}, Loss: {avg_loss:.6f}, ||w||: {torch.norm(w).item():.4f}, "
                          f"Δw∞: {w_diff_inf:.2e}")
            w_old = w.clone()
        else:
            if verbose and (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}, Loss: {avg_loss:.6f}, ||w||: {torch.norm(w).item():.4f}")

    return -w.detach().cpu().numpy(), loss_history

# ------------------------------------------------------------
# AdamW optimizer, CPU (NumPy)
def optimize_weights_cpu(Pre_train, Obs_train, batch_size, n_epochs, lr_init,
                         weight_decay, beta1, beta2, tol, patience, use_adaptive_stop,
                         verbose=True):
    n_samples, n_features = Pre_train.shape
    Dis = np.abs(Pre_train - Obs_train)
    w = np.zeros(n_features)
    m = np.zeros(n_features)
    v = np.zeros(n_features)
    t = 0
    loss_history = []
    w_old = w.copy()
    converged_count = 0

    for epoch in range(n_epochs):
        lr = lr_init * (0.01 + 0.99 * (1 + np.cos(np.pi * epoch / n_epochs)) / 2)
        indices = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            bs = len(batch_indices)
            Obs_batch = Obs_train[batch_indices]
            Dis_batch = Dis[batch_indices]
            Pre_batch = Pre_train[batch_indices]

            a_M = np.abs(Pre_batch[np.newaxis, :, :] - Obs_batch[:, np.newaxis, :])
            pi_M = np.einsum('ijk,k->ij', a_M, w)
            pi_M = pi_M - pi_M.max(axis=1, keepdims=True)
            proi = np.exp(pi_M)
            proi = proi / proi.sum(axis=1, keepdims=True)

            diag_idx = np.arange(bs)
            diagonal_probs = proi[diag_idx, diag_idx]
            loss = -np.mean(np.log(diagonal_probs + 1e-10))

            expected_dist = np.einsum('ij,ijk->ik', proi, a_M)
            grad = np.mean(Dis_batch - expected_dist, axis=0)

            t += 1
            g_t = grad
            m = beta1 * m + (1 - beta1) * g_t
            v = beta2 * v + (1 - beta2) * (g_t ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            w = w + lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + weight_decay * w)

            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if use_adaptive_stop:
            w_diff_inf = np.max(np.abs(w - w_old))
            if w_diff_inf < tol:
                converged_count += 1
                if verbose and (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}, Loss: {avg_loss:.6f}, ||w||: {np.linalg.norm(w):.4f}, "
                          f"Δw∞: {w_diff_inf:.2e} [{converged_count}/{patience}]")
                if converged_count >= patience:
                    print(f"Converged at epoch {epoch+1}, max |dw| = {w_diff_inf:.2e} < {tol:.2e}")
                    break
            else:
                converged_count = 0
                if verbose and (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}, Loss: {avg_loss:.6f}, ||w||: {np.linalg.norm(w):.4f}, "
                          f"Δw∞: {w_diff_inf:.2e}")
            w_old = w.copy()
        else:
            if verbose and (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}, Loss: {avg_loss:.6f}, ||w||: {np.linalg.norm(w):.4f}")

    return -w, loss_history

# ------------------------------------------------------------
# Weight learning
def train_weights(args, device_info):
    set_seed(args.seed, has_gpu=device_info['use_gpu'])

    # Predicted and observed phenotypes of the training samples
    print(f"Training predictions: {args.train_predictions}")
    print(f"Training observations: {args.train_observations}")
    pred_df = pd.read_csv(args.train_predictions)
    obs_df = pd.read_csv(args.train_observations)

    # No missing values allowed
    if pred_df.isna().any().any():
        raise ValueError("The prediction table has missing values")
    if obs_df.isna().any().any():
        raise ValueError("The observation table has missing values")

    id_col = pred_df.columns[0]
    if id_col not in obs_df.columns:
        raise ValueError(f"The observation table has no ID column '{id_col}'")

    # Samples present in both tables, in the order of the prediction table (keeps runs reproducible)
    obs_ids = set(obs_df[id_col])
    common_ids = [i for i in pred_df[id_col] if i in obs_ids]
    if len(common_ids) == 0:
        raise ValueError("The prediction and observation tables share no sample IDs")
    print(f"Samples in both tables: {len(common_ids)}")

    # Align rows by ID
    pred_df = pred_df.set_index(id_col).loc[common_ids].reset_index()
    obs_df = obs_df.set_index(id_col).loc[common_ids].reset_index()

    # Keep trait columns only
    pred_df = pred_df.drop(columns=[id_col])
    obs_df = obs_df.drop(columns=[id_col])

    # Align trait columns
    pred_cols = list(pred_df.columns)
    obs_cols = list(obs_df.columns)
    if set(pred_cols) != set(obs_cols):
        common_cols = [c for c in pred_cols if c in obs_cols]
        if len(common_cols) == 0:
            raise ValueError("The prediction and observation tables share no trait columns")
        print(f"Warning: trait columns differ; using the {len(common_cols)} shared traits")
        pred_df = pred_df[common_cols]
        obs_df = obs_df[common_cols]
    else:
        # same traits, possibly in another order
        obs_df = obs_df[pred_cols]

    prediction_all = pred_df.values.astype(np.float64)
    observation_all = obs_df.values.astype(np.float64)

    # Optionally use only the first samples
    if args.num_train_samples > 0 and args.num_train_samples < len(prediction_all):
        print(f"Using the first {args.num_train_samples} of {len(prediction_all)} samples")
        prediction_train = prediction_all[:args.num_train_samples]
        observation_train = observation_all[:args.num_train_samples]
    else:
        prediction_train = prediction_all
        observation_train = observation_all

    # Keep traits whose prediction accuracy (PCC) reaches the threshold
    n_traits = prediction_train.shape[1]
    pcc_list = []
    for i in range(n_traits):
        pcc = np.corrcoef(prediction_train[:, i], observation_train[:, i])[0, 1]
        pcc_list.append(pcc)
    Ib = np.where(np.array(pcc_list) >= args.pcc_threshold)[0]
    print(f"Traits used: {len(Ib)} of {n_traits} (prediction PCC >= {args.pcc_threshold})")

    # Standardize
    pre_Pro = prePro(prediction_train[:, Ib])
    obs_Pro = prePro(observation_train[:, Ib])

    # Learn the weights
    print(f"Learning weights from {observation_train.shape[0]} samples and {len(Ib)} traits")
    start = time.time()

    if device_info['use_gpu']:
        print("Device: GPU (PyTorch)")
        import torch
        device = torch.device("cuda")
        weights, loss_history = optimize_weights_gpu(
            pre_Pro['value'], obs_Pro['value'],
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            lr_init=args.lr,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            tol=args.tol,
            patience=args.patience,
            use_adaptive_stop=not args.no_adaptive_stop,
            device=device,
            verbose=True
        )
    else:
        print("Device: CPU (NumPy)")
        weights, loss_history = optimize_weights_cpu(
            pre_Pro['value'], obs_Pro['value'],
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            lr_init=args.lr,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            tol=args.tol,
            patience=args.patience,
            use_adaptive_stop=not args.no_adaptive_stop,
            verbose=True
        )

    elapsed = time.time() - start
    print(f"Weight learning finished in {elapsed:.2f} s, final loss {loss_history[-1]:.6f}")

    all_trait_names = pred_df.columns.tolist()
    names_trait = [all_trait_names[i] for i in Ib]

    # Means and SDs of the observed traits, keyed by trait name
    obs_mean_dict = dict(zip(names_trait, obs_Pro['Mean']))
    obs_sd_dict = dict(zip(names_trait, obs_Pro['SD']))

    return weights, names_trait, obs_mean_dict, obs_sd_dict, all_trait_names
