"""Running MixHop or N-GCN."""

import torch
from pathlib import Path
from param_parser import parameter_parser
from trainer_and_networks import Trainer
from utils import (
    tab_printer,
    graph_reader,
    feature_reader,
    target_reader,
    load_mimic_propagator,
    load_mimic_train_test,
    load_mimic_features_from_npz,
    load_mimic_target_full,
)


def main():
    """
    Parsing command line parameters, reading data.
    Fitting an NGCN / MixHop and scoring the model.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)

    if args.dataset in ("mimic", "eicu"):
        data_root = Path(args.data_dir)
        data_name = "eICU" if args.dataset == "eicu" else "mimicIV"
        # KGAT-style: prefer data_dir/data_name; still allow passing a full dataset directory.
        data_dir = data_root if (data_root / "train2.txt").exists() else (data_root / data_name)
        print(f"Using data directory: {data_dir}")

        print("Loading train/test splits ...")
        train_items, test_set, n_users, n_items = load_mimic_train_test(
            str(data_dir / "train2.txt"), str(data_dir / "test2.txt")
        )
        print(f"  n_users={n_users}  n_items={n_items}")

        print("Loading normalized adjacency matrix ...")
        propagation_matrix = load_mimic_propagator(str(data_dir / "s_norm_adj_mat2.npz"))

        print("Loading node features ...")
        features = load_mimic_features_from_npz(
            str(data_dir / "feature.npz"), n_users, n_items
        )

        target = load_mimic_target_full(str(data_dir / "mimic_target.csv"), n_users, n_items)

        # Pass MIMIC-specific data through args so Trainer.setup_mimic_features can pick them up
        args.n_users = n_users
        args.n_items = n_items
        args.propagation_matrix = propagation_matrix
        args.train_items = train_items
        args.test_set = test_set

        if args.mimic_hidden is not None:
            h = args.mimic_hidden
            args.layers_1 = [h, h, h]
            args.layers_2 = [h, h, h]

        tab_printer(args)
        trainer = Trainer(args, None, features, target, True)
        trainer.fit()
        trainer.evaluate_mimic_ranking()
        return

    tab_printer(args)
    graph = graph_reader(args.edge_path)
    features = feature_reader(args.features_path)
    target = target_reader(args.target_path)
    trainer = Trainer(args, graph, features, target, True)
    trainer.fit()
    if args.model == "mixhop":
        trainer.evaluate_architecture()
        args = trainer.reset_architecture()
        trainer = Trainer(args, graph, features, target, False)
        trainer.fit()


if __name__ == "__main__":
    main()
