"""Parameter parsing."""

import argparse
from pathlib import Path

def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on the Cora dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="Run MixHop/N-GCN.")

    _repo_data_root = Path(__file__).resolve().parents[2] / "Data"

    parser.add_argument("--dataset",
                        nargs="?",
                        default="mimic",
                        choices=["cora", "mimic", "eicu"],
                        help="cora: node classification. mimic/eicu: patient–disease recommendation (same file layout).")

    parser.add_argument("--data-dir",
                        nargs="?",
                        default=str(_repo_data_root),
                        help="Dataset root or full dataset dir. KGAT-style resolution is used: data-dir + data_name (mimicIV/eICU).")

    parser.add_argument("--ks",
                        nargs="+",
                        type=int,
                        default=[3, 5, 10, 20],
                        help="Top-K list for Precision, Recall, NDCG, Hit Rate, MRR. Default: 3 5 10 20.")

    parser.add_argument("--metric-every",
                        type=int,
                        default=0,
                        help="MIMIC: print test ranking metrics every N epochs (default 10). Set 0 to disable.")

    parser.add_argument("--mimic-hidden",
                        type=int,
                        default=None,
                        help="MIMIC only: set layers_1 and layers_2 to [H,H,H] to reduce CPU/GPU peak memory (e.g. 96 or 128).")

    parser.add_argument("--force-gpu",
                        action="store_true",
                        help="For MIMIC: use CUDA anyway (large graphs often OOM on consumer GPUs).")

    parser.add_argument("--edge-path",
                        nargs="?",
                        default="./input/cora_edges.csv",
	                help="Edge list csv.")

    parser.add_argument("--features-path",
                        nargs="?",
                        default="./input/cora_features.json",
	                help="Features json.")

    parser.add_argument("--target-path",
                        nargs="?",
                        default="./input/mimic_target.csv",
	                help="Target classes csv.")

    parser.add_argument("--model",
                        nargs="?",
                        default="mixhop",
	                help="Target classes csv.")

    parser.add_argument("--epochs",
                        type=int,
                        default=100,
	                help="Number of training epochs. Default is 2000.")

    parser.add_argument("--seed",
                        type=int,
                        default=456,
	                help="Random seed for train-test split. Default is 42.")

    parser.add_argument("--early-stopping",
                        type=int,
                        default=10,
	                help="Number of early stopping rounds. Default is 10.")

    parser.add_argument("--training-size",
                        type=int,
                        default=1500,
	                help="Training set size. Default is 1500.")

    parser.add_argument("--validation-size",
                        type=int,
                        default=500,
	                help="Validation set size. Default is 500.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
	                help="Dropout parameter. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
	                help="Learning rate. Default is 0.01.")

    parser.add_argument("--cut-off",
                        type=float,
                        default=0.1,
	                help="Weight cut-off. Default is 0.1.")

    parser.add_argument("--lambd",
                        type=float,
                        default=0.0005,
	                help="L2 regularization coefficient. Default is 0.0005.")

    parser.add_argument("--layers-1",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space (top). E.g. 200 20.")

    parser.add_argument("--layers-2",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space (bottom). E.g. 200 200.")

    parser.add_argument("--budget",
                        type=int,
                        default=60,
                        help="Architecture neuron allocation budget. Default is 60.")

    parser.set_defaults(layers_1=[200, 200, 200])
    parser.set_defaults(layers_2=[200, 200, 200])
    
    return parser.parse_args()
