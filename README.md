CodeToAGI — Deep Learning Series

Episode 16: Batch Normalization Explained — Why It Speeds Up Training

Module 4 — Training Deep Networks
Presenter: Mahaz Abbasi · AI Engineer
Series: 72 episodes · From neurons to GPT



What this episode covers





Internal covariate shift — why activations drift during training



BatchNorm forward pass (4 steps): mean → variance → normalize → scale & shift



Learnable parameters γ (gamma) and β (beta)



Train mode vs Eval mode (the critical difference)



Running mean & variance (how inference works)



Layer Normalization — why Transformers use it instead of BatchNorm



Convergence comparison (with vs without BatchNorm)



PyTorch code: nn.BatchNorm1d / nn.BatchNorm2d



Challenge

See BatchNorm Speed Up Training Yourself





Take your EP15 10-layer MLP with He initialization.



Train it on MNIST for 30 epochs. Record loss curve and final accuracy.



Add nn.BatchNorm1d after every Linear layer (before activation).



Train again — same epochs, same learning rate.



Compare loss curves: how many epochs to reach the same accuracy?



Try removing model.eval() at test time — see what breaks.



Post your epoch-to-accuracy comparison in the comments.

Solution file: ep16_batchnorm_comparison.py

python ep16_batchnorm_comparison.py

The script:





Builds a 10-layer MLP (He init)



Trains without BatchNorm and with BatchNorm1d



Plots test loss & accuracy side-by-side



Demonstrates the silent bug when you forget model.eval()



Reports how many epochs each version needs to reach 97% accuracy



Key takeaways







Concept



Takeaway





Internal Covariate Shift



Every weight update changes the input distribution of the next layer





BatchNorm 4 steps



μ_B → σ²_B → normalize → γ·x̂ + β





γ & β



Let the network re-scale if the task needs a non-standard distribution (init γ=1, β=0)





Train vs Eval



model.train() → batch stats · model.eval() → running stats





Running statistics



Exponential moving average (momentum=0.1 in PyTorch)





LayerNorm



Normalize across features (per sample) — used by Transformers





Placement



After Linear / Conv, before activation



PyTorch quick reference

import torch.nn as nn

# MLP
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),   # ← after Linear, before activation
    nn.ReLU(),
    nn.Linear(256, 10),
)

# CNN
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.BatchNorm2d(32),    # num_channels
    nn.ReLU(),
    ...
)

model.train()   # training loop
model.eval()    # validation / inference  ← NEVER forget this



Resources





Original paper: Ioffe & Szegedy, 2015 — Batch Normalization



Full series code: github.com/CodeToAGI/deep-learning-series



Next episode: EP17 — Dropout & Regularization Explained



License

Code and materials for educational use. Feel free to fork, experiment, and share your results.
