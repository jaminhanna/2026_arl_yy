import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib import animation

from tennlab_slayer import *
from tennlab_slayer.datasets import FredDataset

from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)

train_set = FredDataset(
    train=True,
    train='/lustre/isaac24/scratch/repos/tennlab-slayer/data',
    repeats=25,
)

test_set = FredDataset(
    train=False,
    train='/lustre/isaac24/scratch/repos/tennlab-slayer/data',
    repeats=25
)

class loss_func:

    def __init__(self, true_rate:float, false_rate:float) -> None:
        self.spike_rate_loss = lava_slayer.loss.SpikeRate(true_rate, false_rate)

    def __call__(self, spikes:torch.tensor, voltage:torch.tensor, target:torch.tensor) -> torch.tensor:
        if target.dtype not in [torch.int64, torch.int32]:
            target = target.to(torch.long)
        return self.spike_rate_loss(spikes, target)

def pred_func(spikes:torch.Tensor, voltage:torch.Tensor, target:torch.Tensor) -> int:
    """Prediction function for training. Returns number of correct predictions."""
    # spike_count = torch.sum(spikes.flatten(start_dim=1, end_dim=3), dim=-1)
    spike_count = torch.sum(spikes, dim=-1)
    pred_class = torch.argmax(spike_count, dim=-1)
    target_class = target.flatten()
    return torch.sum(pred_class == target_class).item()

time_steps = 1

encoder = FeatureEncoderArray([
    SpikeEncoder(dmin=0, dmax=1, time_steps=time_steps) for _ in range(1*31*39)
])
decoder = VoteDecoder([0, 1])

network_params = {
    "risp": {
        "threshold":                ("float", (0.1, 2.0)),
        "decay":                    ("categorical", ("none", "all")),
        "min_potential":            ("float", (-0.1, 0.0)),
        "discrete":                 ("categorical", (True, False)),
        "delay_bounds":             ("float", (1.0, 20.0)),
        "encoder":                  None,
        "decoder":                  ("categorical", {"vote_decoder": decoder}),
        "tau_grad":                 ("float", (0.1, 10.0)),
        "scale_grad":               ("float", (0.1, 10.0)),
    },
}

arch_params = {
    "input_shape": (1, 31, 39), # (C, H, W)
    "output_shape": (2, 1, 1), # (C, H, W)

    "hidden_layers": ("int", 2),

    "blocks": {
        "block_1": {
            "block_type": "conv",
            "conv_params": {
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1
            },
        },
        "block_2": {
            "block_type": "conv",
            "conv_params": {
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1
            },
        },
    },
}

training_params = {
    "train_set":            train_set,
    "test_set":             test_set,
    "num_workers":          3,
    "epochs":               3,
    "batch_size":           300,
    "lr":                   ("float", (1e-4, 1e-2)),
    "optimizer":            torch.optim.Adam,
    "loss_func":            ("categorical", {
                                "spike_rate_loss": {
                                    "class_obj": loss_func,
                                    "params": {
                                        "true_rate":    ("float", (0.5, 1.0)),
                                        "false_rate":   ("float", (0.0, 0.3)),
                                    },
                                },
                            }),
    "pred_func":            pred_func,
    "verbose":              True,
    "checkpoint_interval":  None,
    "sim_time": 50
}

tuner = HPO(
    network_params=network_params,
    arch_params=arch_params,
    training_params=training_params,
    train_set=train_set,
    test_set=test_set,
    opt_metric="accuracy",
    direction="maximize",
)

study = tuner.tune(n_trials=100)
