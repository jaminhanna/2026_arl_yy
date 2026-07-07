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

time_steps = 1

encoder = FeatureEncoderArray([
    SpikeEncoder(dmin=0, dmax=1, time_steps=time_steps) for _ in range(1*31*39)
])
decoder = VoteDecoder([0, 1])

slayer_net = SlayerRispNetwork(
    threshold=0.5588597458321564,
    decay="none",
    discrete=True,
    min_potential=-0.058267733127339516,
    threshold_inclusive=False,
    encoder=None,
    label_encoder=None,
    decoder=decoder,
    delay_bounds=(9.379981289693491, 13.052381029533251),
    persistent_state=False,
    pre_hook_fx=quantize_8bit,
    tau_grad=2.5132162619142497,
    scale_grad=0.12563346000213893
)

slayer_net.add_conv_layer(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1)  # 39x31x1 -> 20x16x4
slayer_net.add_conv_layer(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1)  # 20x16x4 -> 10x8x4
slayer_net.add_dense_layer(10*8*4, 2)

loss = loss_func(true_rate=0.5009522816873605, false_rate=0.046668129255549036)

stats = train_net(
    net=slayer_net,
    train_set=train_set,
    test_set=test_set,
    epochs=100,
    batch_size=300,
    num_workers=2,
    optimizer=torch.optim.Adam(params=slayer_net.parameters(), lr=0.00916974800240844),
    loss_func=loss,
    pred_func=pred_func,
    verbose=True,
    sim_time=50
)

tennlab_net = slayer_net.to_tennlab_net()
print("Number of Nodes:", tennlab_net.tennlab_net.num_nodes())
print("Number of Edges:", tennlab_net.tennlab_net.num_edges())

net_json = tennlab_net.tennlab_net.as_json()

with open("fred_regular.json", 'w') as f:
    f.write(str(json.dumps(net_json)))
    f.close()

x, y = train_set[163072]
C, H, W, T = x.shape

with torch.no_grad():
    slayer_pred, _, slayer_spike_raster, slayer_voltage_traces = slayer_net.pred(x.unsqueeze(0), enable_traces=True, sim_time=50)
tennlab_pred, tennlab_spike_raster, tennlab_voltage_traces = tennlab_net.pred(x.reshape(C*H*W, T), enable_traces=True, sim_time=50)

spike_count = torch.sum(slayer_pred, dim=-1)
slayer_pred = torch.argmax(spike_count, dim=-1)

spike_count = torch.sum(tennlab_pred, dim=-1)
tennlab_pred = torch.argmax(spike_count, dim=-1)

print("Label:", y)
print("SLAYER Prediction:", slayer_pred)
print("TENNLAB Prediction:", tennlab_pred)

print(slayer_spike_raster.shape)

print(f"SLAYER and RAVENS spike raster match  : {np.array_equal(slayer_spike_raster, tennlab_spike_raster)}")
print(f"SLAYER and RAVENS voltage traces match: {np.array_equal(slayer_voltage_traces, tennlab_voltage_traces)}")

for i, _ in enumerate(slayer_voltage_traces):
    if not np.array_equal(slayer_voltage_traces[i], tennlab_voltage_traces[i]):
        print(f"Voltage trace {i} does not match:")
        for j in range(len(slayer_voltage_traces[i])):
            if slayer_voltage_traces[i][j] != tennlab_voltage_traces[i][j]:
                print(f"  Mismatch at index {j}: SLAYER={slayer_voltage_traces[i][j]}, TENNLAB={tennlab_voltage_traces[i][j]}")
        print()
