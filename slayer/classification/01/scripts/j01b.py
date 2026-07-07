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
    root='/lustre/isaac24/scratch/jhanna8/repos/tennlab-slayer/data',
    repeats=25,
)

test_set = FredDataset(
    train=False,
    root='/lustre/isaac24/scratch/jhanna8/repos/tennlab-slayer/data',
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

slayer_net = SlayerRispNetwork(
    threshold=1.480225712417752,
    decay="none",
    discrete=True,
    min_potential=-0.07617387433603225,
    threshold_inclusive=False,
    encoder=None,
    label_encoder=None,
    decoder=None,
    delay_bounds=(2.371068657241743, 3.869772905406638),
    persistent_state=False,
    pre_hook_fx=quantize_8bit,
)

slayer_net.add_conv_layer(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1)  # 39x31x1 -> 20x16x4
slayer_net.add_conv_layer(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1)  # 20x16x4 -> 10x8x4
slayer_net.add_dense_layer(10*8*4, 2)

loss = loss_func(true_rate=0.48108997255143454, false_rate=0.0039048308261501585)

stats = train_net(
    net=slayer_net,
    train_set=train_set,
    test_set=test_set,
    epochs=100,
    batch_size=300,
    num_workers=2,
    optimizer=torch.optim.Adam(params=slayer_net.parameters(), lr=0.0004955011120842235),
    loss_func=loss,
    pred_func=pred_func,
    verbose=True,
    sim_time=50
)

tennlab_net = slayer_net.to_tennlab_net()
print("Number of Nodes:", tennlab_net.tennlab_net.num_nodes())
print("Number of Edges:", tennlab_net.tennlab_net.num_edges())

net_json = tennlab_net.tennlab_net.as_json()

with open("networks/network.json", 'w') as f:
    f.write(str(json.dumps(net_json)))
    f.close()
