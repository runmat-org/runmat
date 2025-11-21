## GPU Residency & Precision Guide

RunMat keeps tensors on the GPU whenever that offers a measurable win, but it
never changes numeric precision behind the user’s back. This document walks
through the rules the runtime follows so you can predict when data is uploaded,
downloaded, or kept on the host.

### Providers and Supported Precisions

Every acceleration provider advertises a single logical precision:

- `ProviderPrecision::F64` – the provider executes double-precision kernels and
  can safely consume both `double` and `single` inputs.
- `ProviderPrecision::F32` – the provider can only execute single-precision
  kernels. Uploading an f64 tensor to a provider with this capability is an
  accuracy loss, so RunMat refuses to do it.

When a provider reports `F32`, any request to run in f64 falls back to the host
implementation. You can still opt into a manual `single(...)` cast or use the
`single` variants of your data structures if you want the GPU performance at
reduced precision.

### Dtype Tracking on the Host

RunMat host tensors carry their logical dtype (`NumericDType::F64` or
`NumericDType::F32`). Scalars (`Value::Num`, integers, logicals, chars) are
treated as f64 unless you explicitly wrap them in `single`.

### Upload Decisions

The runtime gateways every call to `provider.upload(...)` through a dtype check:

1. Determine the logical dtype of the value about to be uploaded.
2. Ask the provider whether that dtype is supported.
3. If the provider advertises support, perform the upload.
4. If not, abort the upload and execute the operation on the CPU instead.

This guard is applied consistently in:

- **Auto-offload promotions** (elementwise, reductions, matmul).
- **Fusion execution** (both elementwise and reduction kernels).
- **Manual “convert to GPU” helpers** that the builtins use to materialise
  outputs on the device (for example, `times(..., 'like', gpuArray)`).
- **Calibration and profiling code** – the planner never tries to benchmark an
  f64 kernel on an f32-only device.

### Fusion Behaviour

When a fusion group is accepted for GPU execution we still re-check every input
at runtime. If *any* operand would need an implicit f64→f32 conversion, the
fusion execution aborts with an informative error and the VM reruns the group
on the CPU. This keeps fused kernels fast when they are legal and avoids
accidental precision loss when they are not.

The same guard applies to fusion constants. Scalars injected by the VM are
treated as f64 unless they were explicitly generated as single precision.

### Downloads

Downloads are symmetric: calling `gather` or any built-in that returns a host
result simply copies the device buffer into a host tensor with the same dtype
the provider used. There is no implicit cast on the way back to the host.

### Opting Into Lower Precision

Some workflows intentionally trade accuracy for throughput. In that case,
explicitly cast to `single` (or use single-precision allocation helpers) so the
runtime knows the request is deliberate.

If you really do want the old behaviour—implicitly downcasting doubles to the
provider’s native precision—set `RUNMAT_ALLOW_PRECISION_DOWNCAST=1` in your
environment. When the flag is present the precision guard allows f64 uploads to
f32-only providers and emits a one-time warning. Use this sparingly; the flag is
global and bypasses the accuracy protection described above.

### Summary Checklist

- ✅ Provider reports `F64`: both `double` and `single` inputs run on the GPU.
- ✅ Provider reports `F32`: `single` inputs run on the GPU.
- ❌ Provider reports `F32`: `double` inputs stay on the CPU unless you cast.
- ⚠️ Fusion groups obey the same rules; a single incompatible operand forces a
  CPU fallback for the whole group.
- 📓 Calibration and profiling benchmark GPU kernels in the provider’s native
  precision; if a dtype isn’t supported, the run simply falls back to the CPU
  path for those samples.

If you notice a workflow where these rules don’t line up with what you expect,
file an issue and include the minimal script along with the provider telemetry
(`runmat accel-info --json`). Keeping the policy simple and documented makes it
easier for us to extend RunMat to new hardware without surprising anyone.*** End Patch
