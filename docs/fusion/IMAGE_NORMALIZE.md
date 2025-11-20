# Image Normalisation Fusion

This fusion covers any workflow that whitens Batch × Height × Width tensors, applies gain/bias, and optionally raises the result to a gamma—all inside one GPU dispatch.

## What Qualifies

- **Pattern:** `((X - mean) ./ sqrt(var + eps)) .* gain + bias`, optionally followed by `.^gamma`.
- **Statistics inputs.** `mean` and `var` must be computed over the same input tensor `X`. The planner allows constants or scalar tensors for gain/bias/gamma/epsilon.
- **Tensor shape.** Inputs must be 3-D `[batch, height, width]` in column-major order; channel dimension is folded into the batch axis in current pipelines.
- **Constant scalars.** `epsilon`, `gain`, `bias`, and `gamma` must resolve to scalars (either literals or scalar tensors) so the executor can materialise them.

## Why Image Normalisation?

Image normalisation is a common pattern in imaging and sensor pipelines. It shows up in image preprocessing, feature extraction, and other imaging operations. By fusing this pattern, we can keep the normalised tensor resident on the GPU, avoiding the overhead of uploading and downloading it to the CPU, and allows us to compute it without launching a second kernel.

It is also generally a pattern that shows up in pipelines outside of imaging, such as signal processing and solver workloads. For example, in the context of a Kalman filter, image normalisation can be used to preprocess the state vector before it is passed to the Kalman gain computation, or in a control loop, image normalisation can be used to preprocess the state vector before it is passed to the control law computation.

## Benefits

- **Zero host traffic.** Neither the input tensor nor the normalised tensor round-trips through the CPU.
- **Fewer GPU launches.** We can consolidate multiple computations into a single kernel for a common pattern, reducing the number of GPU launches.

## Limitations

- Assumes a channel-last `[batch, height, width]` layout. Channel-first tensors require reshaping before this fusion can trigger.
- Per-pixel or spatially varying gain/bias/gamma tensors are not supported; they must be broadcast scalars.
- At present only the canonical builtin path recognises this pattern; other custom normalisers must be lowered to the same graph shape to benefit.

If a workload should fuse but does not, enable `RUNMAT_DEBUG_FUSION=1` to have the planner print why a node was rejected, then compare against the criteria above.