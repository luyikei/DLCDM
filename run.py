import argparse
import models.model as model

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# ------------------
# Args
# ------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["PISA", "PIAAC", "PIAAC_FLAT", "PISA12R", "PISA12RLong", "PISA12M", "PISA12MLong"])
parser.add_argument("--model", default=None,
                    choices=["LLM", "DINA", "DINO", "NeuralCD"],
                    help="CDM type. Defaults: PISA=LLM, PIAAC/PIAAC_FLAT=DINA")
parser.add_argument("--arch", default="LSTM", choices=["LSTM", "Attn", "None", "FlatV2", "LSTMV2", "FlatV3", "LSTMV3"],
                    help="Process encoder: LSTM, Attn, FlatV2, LSTMV2, FlatV3, LSTMV3, or None (vanilla CDM)")
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=None,
                    help="Learning rate. Defaults: PISA=0.001, PIAAC/PIAAC_FLAT=0.01")
parser.add_argument("--fold", type=int, default=None,
                    help="Run a single fold (1-based). Omit to run all folds sequentially.")
parser.add_argument("--full", action="store_true",
                    help="Train on the entire dataset; proc_w tuned using fold 1 as validation.")
args = parser.parse_args()

DATASET_NAME = args.dataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH_N = 150
BATCH_SIZE = args.batch_size
WEIGHTS_CAND = [1, 0.1, 0.01, 0.001]  # 0 removed: use --arch None for vanilla CDM
PROC_DIM = 8  # used by PIAAC (LSTM hidden size)

# ------------------
# Dataset-specific config
# ------------------

if DATASET_NAME == "PISA":
    from datasets.pisa import PISA
    full_ds = PISA()
    folds = pd.read_csv('datafiles/PISA_folds.csv')
    MAX_FOLD = 10
    MODEL_NAME = args.model or "LLM"

elif DATASET_NAME == "PIAAC_FLAT":
    from datasets.piaac_flat import PIAACFlat
    full_ds = PIAACFlat()
    folds = pd.read_csv('datafiles/PIAAC/loo_cv_fold_PIAAC.csv')
    MAX_FOLD = 7
    MODEL_NAME = args.model or "DINA"

elif DATASET_NAME == "PISA12R":
    from datasets.PISA12R import PISA12R
    full_ds = PISA12R()
    folds = pd.read_csv('datafiles/PISA12R/loo_cv_fold_PISA12R.csv')
    MAX_FOLD = 8
    MODEL_NAME = args.model or "DINA"

elif DATASET_NAME == "PISA12RLong":
    from datasets.PISA12RLong import PISA12RLong
    full_ds = PISA12RLong()
    folds = pd.read_csv('datafiles/PISA12R/loo_cv_fold_PISA12R.csv')
    MAX_FOLD = 8
    MODEL_NAME = args.model or "DINA"

elif DATASET_NAME == "PISA12M":
    from datasets.PISA12M import PISA12M
    full_ds = PISA12M()
    folds = pd.read_csv('datafiles/PISA12M/loo_cv_fold_PISA12M.csv')
    MAX_FOLD = 9
    MODEL_NAME = args.model or "DINA"

elif DATASET_NAME == "PISA12MLong":
    from datasets.PISA12MLong import PISA12MLong
    full_ds = PISA12MLong()
    folds = pd.read_csv('datafiles/PISA12M/loo_cv_fold_PISA12M.csv')
    MAX_FOLD = 9
    MODEL_NAME = args.model or "DINA"

else:  # PIAAC
    from datasets.piaac import PIAAC
    full_ds = PIAAC()
    folds = pd.read_csv('datafiles/PIAAC/loo_cv_fold_PIAAC.csv')
    MAX_FOLD = 7
    MODEL_NAME = args.model or "DINA"

LR = args.lr or 0.01
n_total = len(full_ds)


# ------------------
# Data helpers
# ------------------

def gen_idx(fold):
    if fold == MAX_FOLD + 1:
        fold = 1
    rows, cols = np.where(folds == fold)
    return list(map(lambda x: full_ds.findIndex(x[0], x[1]), zip(rows, cols)))


def collate_fn_piaac(batch):
    us, is_, logs, event_idxs, ys = zip(*batch)
    lengths = [max(log.size(0), 1) for log in logs]
    logs = [log if log.size(0) > 0 else torch.zeros(1, 5) for log in logs]
    event_idxs = [ei if ei.size(0) > 0 else torch.zeros(1, dtype=torch.long) for ei in event_idxs]
    logs_padded = nn.utils.rnn.pad_sequence(logs, batch_first=True)          # (B, T_max, 5)
    ei_padded = nn.utils.rnn.pad_sequence(event_idxs, batch_first=True, padding_value=-1)
    log_packed = nn.utils.rnn.pack_padded_sequence(
        logs_padded, torch.tensor(lengths), batch_first=True, enforce_sorted=False
    )
    return torch.stack(us), torch.stack(is_), log_packed, ei_padded, torch.stack(ys)


def make_loader(indices, shuffle=False):
    kwargs = dict(batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=False)
    if DATASET_NAME in ("PIAAC", "PISA12RLong", "PISA12MLong"):
        kwargs["collate_fn"] = collate_fn_piaac
    # PISA and PIAAC_FLAT use the default collate_fn
    return DataLoader(Subset(full_ds, indices), **kwargs)


def make_model():
    Q = torch.as_tensor(full_ds.Q, dtype=torch.bool).to(DEVICE)
    if args.arch == "None":
        return model.VanillaCDM(
            full_ds.n_users, full_ds.n_items, full_ds.n_skills, Q, MODEL_NAME
        ).to(DEVICE)
    elif args.arch == "FlatV2":
        return model.ProcCDMFlatV2(
            full_ds.n_users, full_ds.n_items, full_ds.n_skills,
            full_ds.n_proc, Q, MODEL_NAME
        ).to(DEVICE)
    elif args.arch == "FlatV3":
        return model.ProcCDMFlatV3(
            full_ds.n_users, full_ds.n_items, full_ds.n_skills,
            full_ds.n_proc, Q, MODEL_NAME
        ).to(DEVICE)
    elif args.arch == "LSTMV3":
        return model.ProcCDMLSTMV3(
            full_ds.n_users, full_ds.n_items, full_ds.n_skills,
            PROC_DIM, Q, MODEL_NAME
        ).to(DEVICE)
    elif DATASET_NAME in ("PISA", "PIAAC_FLAT", "PISA12R", "PISA12M"):
        return model.ProcCDMFlat(
            full_ds.n_users, full_ds.n_items, full_ds.n_skills,
            full_ds.n_proc, Q, MODEL_NAME
        ).to(DEVICE)
    elif args.arch == "LSTMV2":
        return model.ProcCDMLSTMV2(
            full_ds.n_users, full_ds.n_items, full_ds.n_skills,
            PROC_DIM, Q, MODEL_NAME
        ).to(DEVICE)
    elif args.arch == "Attn":
        return model.ProcCDMAttn(
            full_ds.n_users, full_ds.n_items, full_ds.n_skills,
            PROC_DIM, Q, MODEL_NAME
        ).to(DEVICE)
    else:
        return model.ProcCDMLSTM(
            full_ds.n_users, full_ds.n_items, full_ds.n_skills,
            PROC_DIM, Q, full_ds.n_event_types, MODEL_NAME
        ).to(DEVICE)



# ------------------
# Evaluation
# ------------------

def evaluate(loader, net):
    net.eval()
    all_u, all_i, all_probs, all_labels = [], [], [], []
    total_seq_loss = 0.0

    with torch.no_grad():
        if DATASET_NAME in ("PISA", "PIAAC_FLAT", "PISA12R", "PISA12M"):
            for u_ids, i_ids, procs, labels in loader:
                all_u.append(u_ids.numpy())
                all_i.append(i_ids.numpy())
                if args.arch in ("FlatV2", "FlatV3"):
                    preds, _ = net(u_ids.to(DEVICE), i_ids.to(DEVICE), procs.to(DEVICE))
                else:
                    preds, _ = net(u_ids.to(DEVICE), i_ids.to(DEVICE))
                all_probs.append(preds.detach().cpu().numpy())
                all_labels.append(labels.numpy().astype(int))
        else:
            for u_ids, i_ids, log_seq, event_idxs, labels in loader:
                all_u.append(u_ids.numpy())
                all_i.append(i_ids.numpy())
                log_seq = log_seq.to(DEVICE)
                event_idxs = event_idxs.to(DEVICE)
                preds, proc_out = net(u_ids.to(DEVICE), i_ids.to(DEVICE), log_seq)
                all_probs.append(preds.detach().cpu().numpy())
                all_labels.append(labels.numpy().astype(int))
                if proc_out is not None and args.arch not in ("LSTMV2", "Attn"):
                    T = proc_out.size(1)
                    if T > 1:
                        h = proc_out[:, :-1, :]
                        logits = net.proc_fc(h)
                        total_seq_loss += nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            event_idxs[:, 1:T].reshape(-1),
                            ignore_index=-1, reduction="mean",
                        ).item()

    u = np.concatenate(all_u)
    i = np.concatenate(all_i)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds_binary = (probs >= 0.5).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float('nan')

    return {
        "auc": auc,
        "acc": (preds_binary == labels).mean(),
        "f1": f1_score(labels, preds_binary, zero_division=0),
        "precision": precision_score(labels, preds_binary, zero_division=0),
        "recall": recall_score(labels, preds_binary, zero_division=0),
        "seq_loss": total_seq_loss,
        "predictions": pd.DataFrame({"u": u, "i": i, "probs": probs, "labels": labels}),
    }


# ------------------
# Training
# ------------------

def train_epoch(net, loader, proc_w, optimizer):
    net.train()
    running_loss = 0.0

    if DATASET_NAME in ("PISA", "PIAAC_FLAT", "PISA12R", "PISA12M"):
        for u_ids, i_ids, procs, labels in loader:
            u_ids, i_ids, procs, labels = (
                u_ids.to(DEVICE), i_ids.to(DEVICE), procs.to(DEVICE), labels.to(DEVICE)
            )
            optimizer.zero_grad()
            if args.arch == "FlatV2":
                preds, theta_proc = net(u_ids, i_ids, procs)
                loss1 = nn.functional.binary_cross_entropy(preds, labels, reduction="mean")
                if theta_proc is not None:
                    theta = torch.sigmoid(net.cdm.student_emb[u_ids])
                    loss2 = nn.functional.mse_loss(theta_proc, theta, reduction="mean")
                else:
                    loss2 = torch.tensor(0.0, device=DEVICE)
                (loss1 + proc_w * loss2).backward()
            elif args.arch == "FlatV3":
                preds, _ = net(u_ids, i_ids, procs)
                loss1 = nn.functional.binary_cross_entropy(preds, labels, reduction="mean")
                loss1.backward()
            else:
                preds, preds2 = net(u_ids, i_ids)
                loss1 = nn.functional.binary_cross_entropy(preds, labels, reduction="mean")
                if preds2 is not None:
                    loss2 = nn.functional.mse_loss(preds2.unsqueeze(1), procs.unsqueeze(1), reduction="mean")
                else:
                    loss2 = torch.tensor(0.0, device=DEVICE)
                (loss1 + proc_w * loss2).backward()
            optimizer.step()
            net.apply_clipper()
            running_loss += loss1.item()
    else:
        for u_ids, i_ids, log_seq, event_idxs, labels in loader:
            u_ids, i_ids, labels = u_ids.to(DEVICE), i_ids.to(DEVICE), labels.to(DEVICE)
            log_seq = log_seq.to(DEVICE)
            event_idxs = event_idxs.to(DEVICE)
            optimizer.zero_grad()
            preds, proc_out = net(u_ids, i_ids, log_seq)
            loss1 = nn.functional.binary_cross_entropy(preds, labels, reduction="mean")
            if args.arch in ("LSTMV2", "Attn"):
                theta_proc = proc_out  # (B, K) process-informed θ estimate
                if theta_proc is not None:
                    theta = torch.sigmoid(net.cdm.student_emb[u_ids])
                    loss2 = nn.functional.mse_loss(theta_proc, theta, reduction="mean")
                else:
                    loss2 = torch.tensor(0.0, device=DEVICE)
                (loss1 + proc_w * loss2).backward()
            elif proc_out is not None:
                T = proc_out.size(1)
                if T > 1:
                    h = proc_out[:, :-1, :]                          # (B, T-1, proc_dim)
                    logits = net.proc_fc(h)
                    loss_type = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        event_idxs[:, 1:T].reshape(-1),
                        ignore_index=-1, reduction="mean",
                    )
                else:
                    loss_type = torch.tensor(0.0, device=DEVICE)
                (loss1 + proc_w * loss_type).backward()
            else:
                loss1.backward()
            optimizer.step()
            net.apply_clipper()
            running_loss += loss1.item()

    return running_loss


def validation(train_loader, val_loader, proc_w):
    net = make_model()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    best_auc, epoch_max = 0, 0
    log_prefix = f"{prefix}_{proc_w}"

    header = ('Epoch,TrainLoss,ValAUC,ValACC,ValSeqLoss\n'
              if DATASET_NAME in ("PIAAC", "PISA12RLong", "PISA12MLong") else
              'Epoch,TrainLoss,ValAUC,ValACC\n')

    with open(f"output/log_{log_prefix}.csv", "w") as f:
        f.write(header)
        no_improve = 0
        for epoch in range(EPOCH_N):
            running_loss = train_epoch(net, train_loader, proc_w, optimizer)
            metrics = evaluate(val_loader, net)
            if DATASET_NAME in ("PIAAC", "PISA12RLong", "PISA12MLong"):
                f.write(f'{epoch+1:03d},{running_loss:.4f},{metrics["auc"]:.4f},'
                        f'{metrics["acc"]:.4f},{metrics["seq_loss"]:.4f}\n')
            else:
                f.write(f'{epoch+1:03d},{running_loss:.4f},{metrics["auc"]:.4f},{metrics["acc"]:.4f}\n')
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                epoch_max = epoch + 1
                no_improve = 0
                print(f'Epoch {epoch+1:03d}  proc_w: {proc_w}  train_loss: {running_loss:.4f} '
                      f'Val AUC: {metrics["auc"]:.4f}  Val Acc: {metrics["acc"]:.4f}')
            else:
                no_improve += 1
                if no_improve >= 10:
                    break

    return {"best_auc": best_auc, "proc_w": proc_w, "epoch_max": epoch_max}


def train(train_val_loader, test_loader, epoch_max, proc_w, fold):
    import time
    net = make_model()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    t0 = time.perf_counter()
    for _ in range(epoch_max):
        train_epoch(net, train_val_loader, proc_w, optimizer)
    fit_time = time.perf_counter() - t0
    print(f"Fold {fold} training time: {fit_time:.2f}s")

    metrics = evaluate(test_loader, net)
    file_prefix = f"{prefix}_{proc_w}_{fold}"
    for path, fn in [
        (f'output/pred_{file_prefix}.csv',    lambda p: metrics["predictions"].to_csv(p)),
        (f'output/model_{file_prefix}.pth',   lambda p: torch.save(net.state_dict(), p)),
        (f'output/profile_{file_prefix}.csv', lambda p: np.savetxt(p, net.profile().cpu().detach().numpy(), delimiter=",")),
        (f'output/params_{file_prefix}.csv',  lambda p: np.savetxt(p, net.get_exer_params().cpu().detach().numpy(), delimiter=",")),
    ]:
        fn(path)
        print(f"Wrote {path}")

    return {
        "fit_time_s": fit_time,
        "AUC": metrics["auc"],
        "ACC": metrics["acc"],
        "F1": metrics["f1"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
    }


# ------------------
# Full-dataset training
# ------------------

def run_full():
    # Use fold 1 as validation set for proc_w selection; all other indices as training
    val_idx   = gen_idx(1)
    train_idx = list(set(range(n_total)) - set(val_idx))

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx)
    full_loader  = make_loader(list(range(n_total)), shuffle=True)

    if args.arch in ("None", "FlatV3", "LSTMV3"):
        val_result = validation(train_loader, val_loader, 0)
        proc_w    = 0
        epoch_max = int(val_result["epoch_max"])
    else:
        val_results = [validation(train_loader, val_loader, w) for w in WEIGHTS_CAND]
        df_val = pd.DataFrame(val_results)
        df_val.to_csv(f'output/results_val_{prefix}_full.csv', index=False)
        print(f"Wrote output/results_val_{prefix}_full.csv")
        best_cond = df_val.loc[df_val.best_auc.idxmax()]
        proc_w    = best_cond.proc_w
        epoch_max = int(best_cond.epoch_max)
    print(f"Full training: selected proc_w={proc_w}, epoch_max={epoch_max}")

    # Retrain on the entire dataset with the selected proc_w
    import time
    net = make_model()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    file_prefix = f"{prefix}_full_{proc_w}"
    t0 = time.perf_counter()
    for _ in range(epoch_max):
        train_epoch(net, full_loader, proc_w, optimizer)
    fit_time = time.perf_counter() - t0
    print(f"Full training time: {fit_time:.2f}s")

    metrics = evaluate(full_loader, net)
    for path, fn in [
        (f'output/pred_{file_prefix}.csv',     lambda p: metrics["predictions"].to_csv(p)),
        (f'output/model_{file_prefix}.pth',    lambda p: torch.save(net.state_dict(), p)),
        (f'output/profile_{file_prefix}.csv',  lambda p: np.savetxt(p, net.profile().cpu().detach().numpy(), delimiter=",")),
        (f'output/params_{file_prefix}.csv',   lambda p: np.savetxt(p, net.get_exer_params().cpu().detach().numpy(), delimiter=",")),
        (f'output/metrics_{file_prefix}.csv',  lambda p: pd.DataFrame([{
            "proc_w": proc_w, "fit_time_s": fit_time, "AUC": metrics["auc"], "ACC": metrics["acc"],
            "F1": metrics["f1"], "Precision": metrics["precision"], "Recall": metrics["recall"],
        }]).to_csv(p, index=False)),
    ]:
        fn(path)
        print(f"Wrote {path}")


# ------------------
# Cross-validation loop
# ------------------

_arch_suffix = "" if args.arch == "LSTM" else f"_{args.arch}"
prefix = f"{DATASET_NAME}_{MODEL_NAME}{_arch_suffix}"
folds_to_run = [args.fold] if args.fold is not None else list(range(1, MAX_FOLD + 1))


def run_fold(fold):
    test_idx = gen_idx(fold)
    val_idx = gen_idx(fold + 1)
    train_idx = list(set(range(n_total)) - set(val_idx) - set(test_idx))
    train_val_idx = list(set(val_idx) | set(train_idx))

    train_loader = make_loader(train_idx, shuffle=True)
    test_loader = make_loader(test_idx)
    val_loader = make_loader(val_idx)
    train_val_loader = make_loader(train_val_idx)

    if args.arch in ("None", "FlatV3", "LSTMV3"):
        val_result = validation(train_loader, val_loader, 0)
        res = train(train_val_loader, test_loader, int(val_result["epoch_max"]), 0, fold)
    else:
        val_results = [validation(train_loader, val_loader, w) for w in WEIGHTS_CAND]
        df = pd.DataFrame(val_results)
        df.to_csv(f'output/results_val_{prefix}_{fold}.csv', index=False)
        print(f"Wrote output/results_val_{prefix}_{fold}.csv")
        print(val_results)
        best_cond = df.loc[df.best_auc.idxmax()]
        res = train(train_val_loader, test_loader, int(best_cond.epoch_max), best_cond.proc_w, fold)
        res["proc_w"] = best_cond.proc_w

    path = f'output/results_{prefix}_fold{fold}.csv'
    pd.DataFrame([{"Fold": fold, **res}]).to_csv(path, index=False)
    print(f"Wrote {path}")


if args.full:
    run_full()
else:
    for fold in folds_to_run:
        run_fold(fold)

    # Aggregate per-fold CSVs when running all folds sequentially
    if args.fold is None:
        all_rows = [pd.read_csv(f'output/results_{prefix}_fold{f}.csv') for f in range(1, MAX_FOLD + 1)]
        agg_path = f'output/results_{prefix}.csv'
        pd.concat(all_rows).to_csv(agg_path, index=False)
        print(f"Wrote {agg_path}")
