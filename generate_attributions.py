# generate_attributions.py
import os
import copy
import warnings
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import tsl
from tsl import config
from tsl.data import SpatioTemporalDataModule, ImputationDataset
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from tsl.imputers import Imputer
from tsl.nn.utils import casting
from tsl.ops.imputation import add_missing_values, sample_mask
from tsl.utils import ArgParser, parser_utils, numpy_metrics
from tsl.utils.python_utils import ensure_list

from spin.baselines import SAITS, TransformerModel, BRITS
from spin.imputers import SPINImputer, SAITSImputer, BRITSImputer
from spin.models import SPINModel, SPINHierarchicalModel

from run_imputation import get_model_classes

warnings.filterwarnings("ignore")
tsl.logger.disabled = True


# ============================================================
# 1. Argument Parser
# ============================================================
def parse_args():
    parser = ArgParser()

    parser.add_argument("--model-name", type=str, default="spin",
                        choices=["spin", "saits", "brits", "grin", "transformer"])
    parser.add_argument("--dataset-name", type=str, default="air")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--root", type=str, default="log")

    # Data sparsity parameters
    parser.add_argument('--p-fault', type=float, default=0.0)
    parser.add_argument('--p-noise', type=float, default=0.75)
    parser.add_argument('--test-mask-seed', type=int, default=None)

    # Dataset splitting settings
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=16)

    # Graph connectivity threshold
    parser.add_argument("--adj-threshold", type=float, default=0.1)

    # MFABA attack parameters
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.001)

    # Output directory
    parser.add_argument("--out-dir", type=str, default="attributions")

    args = parser.parse_args()

    # If config file is provided, load additional arguments
    if args.config is not None:
        cfg_path = os.path.join(config.config_dir, args.config)
    else:
        root = os.path.join(tsl.config.curr_dir, args.root)
        exp_dir = os.path.join(root, args.dataset_name, args.model_name, args.exp_name)
        cfg_path = os.path.join(exp_dir, "config.yaml")

    with open(cfg_path, 'r') as fp:
        config_args = yaml.load(fp, Loader=yaml.FullLoader)
    for k, v in config_args.items():
        setattr(args, k, v)

    return args


# ============================================================
# 2. Model Loading
# ============================================================
def load_model(exp_dir, exp_config, dm, model_name: str):
    """Load the trained imputation model (Imputer wrapper)."""
    model_cls, imputer_class = get_model_classes(exp_config['model_name'])

    additional_model_hparams = dict(
        n_nodes=dm.n_nodes,
        input_size=dm.n_channels,
        u_size=4,
        output_size=dm.n_channels,
        window_size=dm.window,
    )

    # Extract model kwargs
    model_kwargs = parser_utils.filter_args(
        args={**exp_config, **additional_model_hparams},
        target_cls=model_cls,
        return_dict=True
    )

    # Extract imputer kwargs
    imputer_kwargs = parser_utils.filter_argparse_args(
        exp_config,
        imputer_class,
        return_dict=True
    )

    # Build Imputer
    imputer = imputer_class(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={},
        loss_fn=None,
        **imputer_kwargs
    )

    # Locate checkpoint file
    model_path = None
    for file in os.listdir(exp_dir):
        if file.endswith(".ckpt"):
            model_path = os.path.join(exp_dir, file)
            break
    if model_path is None:
        raise ValueError(f"No .ckpt found in {exp_dir}")

    imputer.load_model(model_path)
    imputer.freeze()

    return imputer


# ============================================================
# 3. Build DataModule for Imputation
# ============================================================
def build_datamodule(args, exp_config):
    """Load dataset, apply preprocessing, and prepare DataModule."""
    if args.dataset_name.lower().startswith("air"):
        dataset = AirQuality(impute_nans=True, small=False)
    elif args.dataset_name.lower() == "metrla":
        dataset = MetrLA(impute_nans=True)
    elif args.dataset_name.lower() == "pemsbay":
        dataset = PemsBay(impute_nans=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    # Temporal encoding (day of week, etc.)
    time_emb = dataset.datetime_encoded(['day', 'week']).values
    exog_map = {'global_temporal_encoding': time_emb}

    # Map dataset keys to model input names
    input_map = {
        'u': 'temporal_encoding',
        'x': 'data'
    }

    # Graph adjacency matrix
    adj = dataset.get_connectivity(
        threshold=args.adj_threshold,
        include_self=False,
        force_symmetric=True
    )

    # Build PyTorch dataset
    torch_dataset = ImputationDataset(
        *dataset.numpy(return_idx=True),
        training_mask=dataset.training_mask,
        eval_mask=dataset.eval_mask,
        connectivity=adj,
        exogenous=exog_map,
        input_map=input_map,
        window=exp_config['window'],
        stride=exp_config['stride']
    )

    splitter = dataset.get_splitter(
        val_len=args.val_len,
        test_len=args.test_len
    )

    scalers = {'data': StandardScaler(axis=(0, 1))}

    dm = SpatioTemporalDataModule(
        torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=args.batch_size
    )
    dm.setup()

    return dm


# ============================================================
# 4. FGSM + MFABA Implementation
# ============================================================
class FGSMGrad:
    """Performs iterative FGSM-like gradient perturbations on time-series inputs."""

    def __init__(self, epsilon=0.001):
        self.epsilon = epsilon

    def __call__(self, model, batch, num_steps=10, alpha=0.001):
        device = next(model.parameters()).device

        # Move all tensors inside the batch to the model's device
        data = {k: v.to(device) for k, v in batch.items()
                if isinstance(v, torch.Tensor)}

        dt = data['x'].clone().detach().requires_grad_(True)
        ori_dt = dt.clone().detach()

        # Store intermediate perturbed samples and gradients for each sequence
        hats = [[dt[i:i + 1].clone()] for i in range(dt.shape[0])]
        grads = [[] for _ in range(dt.shape[0])]

        for _ in range(num_steps):
            output = model(x=dt, mask=data['mask'])
            y_hat = output[0]

            loss = F.l1_loss(y_hat[data['eval_mask']], data['y'][data['eval_mask']])
            data_grad = torch.autograd.grad(loss, dt, retain_graph=False)[0]

            # FGSM update
            adv_data = dt - alpha * data_grad.sign()
            total_grad = torch.clamp(adv_data - ori_dt, -self.epsilon, self.epsilon)

            dt = (ori_dt + total_grad).detach().requires_grad_(True)

            # Save trajectory
            for idx in range(dt.shape[0]):
                hats[idx].append(dt[idx:idx + 1].clone())
                grads[idx].append(total_grad[idx:idx + 1].clone())

        # Final gradient computation
        dt_last = torch.cat([h[-1] for h in hats], dim=0).detach().requires_grad_(True)
        output = model(x=dt_last, mask=data['mask'])
        y_hat = output[0]
        loss = F.l1_loss(y_hat[data['eval_mask']], data['y'][data['eval_mask']])
        grad_last = torch.autograd.grad(loss, dt_last)[0]

        for i in range(grad_last.shape[0]):
            grads[i].append(grad_last[i:i + 1].clone())

        hats = [torch.cat(h, dim=0) for h in hats]
        grads = [torch.cat(g, dim=0) for g in grads]

        return hats, grads


class MFABA:
    """Implements MFABA attribution computation."""
    def __call__(self, hats, grads):
        t_list = hats[1:] - hats[:-1]
        grads = grads[:-1]
        total_grads = -torch.sum(t_list * grads, dim=0)
        attribution_map = total_grads.unsqueeze(0)
        return attribution_map.detach().cpu().numpy()


def mfaba(model, batch, epsilon=0.01, num_steps=10, alpha=0.001):
    """Compute MFABA attribution maps for a batch."""
    attacker = FGSMGrad(epsilon=epsilon)
    mfaba_core = MFABA()
    hats, grads = attacker(model, batch, num_steps=num_steps, alpha=alpha)

    attribution_maps = []
    for i in range(len(hats)):
        attribution_maps.append(mfaba_core(hats[i], grads[i]))
    return np.concatenate(attribution_maps, axis=0)


# ============================================================
# 5. Main Script: Generate and Save Attributions
# ============================================================
def main():
    args = parse_args()

    root = os.path.join(tsl.config.curr_dir, args.root)
    exp_dir = os.path.join(root, args.dataset_name, args.model_name, args.exp_name)

    # Load experiment configuration
    with open(os.path.join(exp_dir, 'config.yaml'), 'r') as fp:
        exp_config = yaml.load(fp, Loader=yaml.FullLoader)

    # Build datamodule and load model
    dm = build_datamodule(args, exp_config)
    imputer = load_model(exp_dir, exp_config, dm, args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imputer = imputer.to(device)
    imputer.eval()

    # Storage containers
    all_attr, all_y, all_mask, all_eval_mask = [], [], [], []

    # Attribution generation loop
    for batch in dm.test_dataloader():
        all_y.append(batch['y'].numpy())
        all_mask.append(batch['mask'].numpy())
        all_eval_mask.append(batch['eval_mask'].numpy())

        attr_batch = mfaba(
            imputer,
            batch,
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            alpha=args.alpha
        )
        all_attr.append(attr_batch)

    # Concatenate all batches
    attributions = np.concatenate(all_attr, axis=0)
    ys = np.concatenate(all_y, axis=0)
    masks = np.concatenate(all_mask, axis=0)
    eval_masks = np.concatenate(all_eval_mask, axis=0)

    # Save results
    out_dir = os.path.join(exp_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    file_name = f"mfaba_eps{args.epsilon}_steps{args.num_steps}_alpha{args.alpha}.npz"
    save_path = os.path.join(out_dir, file_name)

    np.savez_compressed(
        save_path,
        attribution=attributions,
        y=ys,
        mask=masks,
        eval_mask=eval_masks
    )

    print("Attributions saved to:", save_path)
    print("Attribution shape:", attributions.shape)


if __name__ == "__main__":
    main()
