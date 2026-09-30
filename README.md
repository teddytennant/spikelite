JAX kernel of SpikeLite (arXiv:2609.35097): a frequency-selective LIF encoder and sparse spiking channel attention for multivariate forecasting.

pip install -e ".[tests]"
python -m pytest
python train.py

Does not reproduce the SeqSNN or SpikF tables. Those need the authors' data splits and training configs. This checks the FSSE telescoping identity, the SSCA mask and softmax-free attention, and that a few steps of Adam reduce loss on a synthetic series.

The paper names RevIN, component normalization, horizon scaling, the spike-replication neuron, and the surrogate family without formulas. RevIN is the standard time-axis instance norm. Component norm is per-component time standardization before the projection. The spike-replication LIF is a scalar-tau neuron driven by repeated Z. Q, K, and V use LayerNorm instead of the figure's batch norm. Horizon scale is a learnable vector of length P. Branch count defaults to 7 to match the L0-L6 figure. Spikes use a sigmoid surrogate with scale 4. Decay factors start on linspace from 0.9 to 0.3, mapped by a logit, with the same value on every channel. The paper only says that start is descending. Thresholds start at 1.0, which the paper does not give. Successive differences follow branch index order and are not re-sorted.
