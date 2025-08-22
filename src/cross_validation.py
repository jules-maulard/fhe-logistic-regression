import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold

from config import setup_logging
logger = setup_logging("INFO")

from data.data_utils import load_data, num_to_data_path, stack_and_scale
from data.preprocessing import bin_to_pm1, add_bias, init_wv
from data.scalers import MaxAbsScaler, MinMaxScaler, StandardScaler
from sigmoid import get_sigmoid_approx
from nag_momentum import NAGUpdater
from models.logreg_pt_sigmoid import LogisticRegressionIDASH
from metrics import print_classification_metrics, calculate_acc_auc_idash_style


def run_cv(args):
    # PARAMETERS
    scalers = {
        "maxabs": MaxAbsScaler(),
        "minmax": MinMaxScaler(),
        "standard": StandardScaler()
    }
    scaler = scalers[args.scaler]
    degree = args.deg
    sigmoid = get_sigmoid_approx("least_squares", degree=degree)
    nag_updater = NAGUpdater('idash')

    # LOAD DATA
    X, y_bin = load_data(num_to_data_path(args.data_num))

    # FOLD
    n_splits = 10 if args.data_num == 1 else 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_bin), 1):
        logger.info(f"Fold {fold}/{n_splits}")

        # PREPROCESSING
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_bin[train_idx], y_bin[test_idx]

        Z_train_ = bin_to_pm1(y_train)[:, np.newaxis] * add_bias(X_train)
        Z_test_ = add_bias(X_test) * bin_to_pm1(y_test)[:, np.newaxis]

        Z_train, Z_test = stack_and_scale(scaler, Z_train_, Z_test_)
        w, v = init_wv(Z_train, method=args.beta_init)

        # MODELE
        model = LogisticRegressionIDASH(
            sigmoid=sigmoid,
            use_NAG=args.use_nag,
            nag_momentum_update=nag_updater
        )

        model.fit(
            Z_train, Z_train.shape,
            w,
            n_epoch=args.epochs,
            gamma_up=args.gu,
            gamma_down=args.gd,
            pt_v=v
        )

        # EVALUATION
        w_trained = model.pt_beta
        proba = model.predict_proba(Z_test, w_trained, sigmoid=sigmoid)

        acc, auc = print_classification_metrics(
            y_test, proba,
            show_acc_auc=False,
            show_report=False,
            show_matrix=False,
            show_mse_nmse=False
        )

        acc_idash, auc_idash = calculate_acc_auc_idash_style(Z_test, w_trained)

        results.append((acc, auc, acc_idash, auc_idash))

    # Résumé global
    accs, aucs, accs_idash, aucs_idash = zip(*results)

    print("\n=== Résultats globaux ===")
    print(f"Normal Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Normal AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"iDASH Accuracy:  {np.mean(accs_idash):.4f} ± {np.std(accs_idash):.4f}")
    print(f"iDASH AUC:       {np.mean(aucs_idash):.4f} ± {np.std(aucs_idash):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_num", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--gu", type=float, default=10)
    parser.add_argument("--gd", type=float, default=-1)
    parser.add_argument("--use_nag", action="store_true")
    parser.add_argument("--scaler", choices=["maxabs", "minmax", "standard"], default="maxabs")
    parser.add_argument("--beta_init", choices=["mean", "mean-idash", "zero", "random"], default="mean-idash")
    parser.add_argument("--deg", type=int, choices=[3, 5, 7], default=5)

    args = parser.parse_args()
    run_cv(args)
