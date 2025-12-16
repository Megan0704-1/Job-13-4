This example uses `linalg.conv_2d_nhwc_hwcf` to implement a 1×1 (pointwise) convolution.
A per-pixel linear projection from the input channel dimension to the output channel dimension (NHWC x HWCF -> NHWC).
With a 1×1 filter, each spatial location (h, w) multiplies its $C_{in}$ length vector by a $C_{in}xC_{out}$ weight matrix, yielding a $C_{out}$ length output.

Formula

Let \(X \in \mathbb{R}^{N \times H \times W \times C_{\text{in}}}\) and
\(W \in \mathbb{R}^{1 \times 1 \times C_{\text{in}} \times C_{\text{out}}}\).
With stride \(=1\) and no padding, the output
\(Y \in \mathbb{R}^{N \times H \times W \times C_{\text{out}}}\) is

\[
Y[n,h,w,o] \;=\; \sum_{c=0}^{C_{\text{in}}-1} X[n,h,w,c]\; W[0,0,c,o].
\]

Equivalently, per spatial location \((n,h,w)\),
if \(\mathbf{x}_{n,h,w} \in \mathbb{R}^{C_{\text{in}}}\) and
\(\mathbf{W} \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}}}\),

\[
\mathbf{y}_{n,h,w} \;=\; \mathbf{x}_{n,h,w}\, \mathbf{W}.
\]


Why it matters: 1×1 convs are the standard for channel mixing.
