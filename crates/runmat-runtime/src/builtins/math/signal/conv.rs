//! MATLAB-compatible `conv` builtin with GPU-aware semantics for RunMat.

use num_complex::Complex;
use runmat_accelerate_api::{ProviderConv1dOptions, ProviderConvMode, ProviderConvOrientation};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "conv",
        wasm_path = "crate::builtins::math::signal::conv"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "conv"
category: "math/signal"
keywords: ["conv", "convolution", "linear convolution", "signal processing", "polynomial multiplication", "gpu"]
summary: "One-dimensional linear convolution with MATLAB-compatible padding modes."
references:
  - title: "MATLAB conv documentation"
    url: "https://www.mathworks.com/help/matlab/ref/conv.html"
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "The in-process provider and WGPU backend expose conv1d; other providers fall back to the host implementation when this hook is unavailable."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::signal::conv::tests"
  integration: "builtins::math::signal::conv::tests::conv_gpu_roundtrip_matches_cpu"
---

# What does the `conv` function do in MATLAB / RunMat?
`conv(a, b)` computes the one-dimensional linear convolution of the vectors `a` and `b`. The default
padding mode returns the full convolution (`length(a) + length(b) - 1`). Optional shape arguments
(`'same'`, `'valid'`) select MATLAB-compatible padding behaviour.

## How does the `conv` function behave in MATLAB / RunMat?
- Accepts real or complex scalars, vectors, or tensors that can be flattened column-major into a vector.
- Keeps the orientation of the first input when returning row or column vectors (`'same'` and `'valid'` honour this rule).
- Supports the three MATLAB shape modes: `'full'` (default), `'same'`, and `'valid'`.
- Returns empty outputs when either input is empty or when `'valid'` is requested with insufficient overlap.
- Logical inputs are promoted to double precision before the convolution.
- GPU inputs are gathered automatically when the active provider lacks a native 1-D convolution kernel.

## Examples of using the `conv` function in MATLAB / RunMat

### Computing the full convolution of two row vectors

```matlab
a = [1 2 3];
b = [1 1 1];
c = conv(a, b);
```

Expected output:

```matlab
c = [1 3 6 5 3];
```

### Keeping the same length as the first input

```matlab
kernel = [1 0 -1];
signal = [3 4 5 6 7];
edge = conv(signal, kernel, 'same');
```

Expected output:

```matlab
edge = [4 2 2 2 -6];
```

### Valid convolution without zero padding

```matlab
weights = [1 2 3 4];
window = [1 1 1];
valid = conv(weights, window, 'valid');
```

Expected output:

```matlab
valid = [6 9];
```

### Convolving column vectors

```matlab
a = (1:3)';
b = [2; 0; -2];
c = conv(a, b);
```

Expected output:

```matlab
c =
     2
     4
     4
     -4
     -6
```

### Multiplying polynomials using convolution

```matlab
p = [1 3 3 1];    % (x + 1)^3 coefficients
q = [1 -1];       % (x - 1)
coeff = conv(p, q);
```

Expected output:

```matlab
coeff = [1 2 0 -2 -1];
```

### Complex-valued convolution

```matlab
t = 0:3;
sig = exp(1i * pi/4 * t);
filt = [1 2i];
resp = conv(sig, filt);
```

Expected output (RunMat prints complex numbers as `a + bi`):

```matlab
resp =
   1.0000 + 1.0000i   1.4142 + 2.4142i   0.0000 + 3.4142i  -1.4142 + 2.4142i  -1.0000 + 1.0000i
```

### Scaling a signal by a scalar

```matlab
s = [4 5 6];
y = conv(2, s);
```

Expected output:

```matlab
y = [8 10 12];
```

### Using gpuArray inputs with automatic host fallback

```matlab
g = gpuArray([1 2 3 4]);
h = gpuArray([1 0 -1]);
edge = conv(g, h, 'same');
result = gather(edge);
```

When the provider implements the `conv1d` hook (the in-process provider and the WGPU backend both do),
the convolution executes without leaving the device and the result remains GPU-resident. When the
hook is unavailable, RunMat gathers both arguments, performs the computation on the CPU, and returns a
host tensor that matches MATLAB semantics.

## FAQ

### Does `conv` require row vectors?
No. `conv` accepts row vectors, column vectors, scalars, and tensors that can be flattened into a vector. The result preserves the orientation of the first input when that orientation is unambiguous.

### What happens when one of the inputs is empty?
The result is an empty vector with an orientation derived from the non-empty input (or a `0×1` column vector when both inputs are empty), matching MATLAB's behaviour.

### How is `'same'` computed?
`'same'` returns the central portion of the full convolution whose length matches the first input. Internally RunMat performs a full convolution and slices the appropriate window.

### When should I use `'valid'`?
Use `'valid'` when you only want results that do not rely on zero padding. This is common when sliding windows should fit completely inside the input without extending past the boundaries.

### Does `conv` support single precision?
The host path computes in double precision. Provider implementations choose their native precision: the in-process provider mirrors double results, while the WGPU backend emits either `f32` or `f64` kernels depending on device support. Providers without a `conv1d` hook gather inputs back to the CPU to maintain MATLAB-compatible answers.

### Will the result stay on the GPU?
Yes—provided the active provider exposes the `conv1d` hook (the in-process provider and WGPU backend do). Without that hook, RunMat gathers inputs, computes on the host, and returns a CPU tensor with the correct orientation.

### Can I convolve matrices or higher-dimensional arrays?
`conv` treats inputs as vectors using MATLAB column-major order. For multi-dimensional convolution use the dedicated `conv2` or `convn` builtins (planned).

### How do I convolve with an impulse (delta) kernel?
Include a `1` followed by zeros in your kernel. The input will be preserved (with appropriate padding) because convolution with a delta function is an identity operation.

### What about circular convolution?
Use `cconv` for circular convolution, or compute the FFT manually and multiply in the frequency domain before performing an inverse FFT.

### Are there GPU-specific tuning knobs?
Not yet. Current providers choose kernel launch parameters automatically; user-facing tuning switches will arrive alongside future backend updates.

## See Also
[fft](../fft/fft), [ifft](../fft/ifft), [fftshift](../fft/fftshift), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- Full source: `crates/runmat-runtime/src/builtins/math/signal/conv.rs`
- Found an issue? [Open a ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::math::signal::conv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "conv",
    op_kind: GpuOpKind::Custom("conv1d"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("conv1d")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may implement `conv1d` to keep results on the device; when unavailable the runtime gathers inputs and runs on the CPU.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::math::signal::conv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "conv",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Convolution boundaries terminate fusion plans; intermediate expressions run on the host today.",
};

#[runtime_builtin(
    name = "conv",
    category = "math/signal",
    summary = "One-dimensional linear convolution with MATLAB-compatible padding.",
    keywords = "conv,convolution,signal processing,gpu",
    accel = "custom",
    wasm_path = "crate::builtins::math::signal::conv"
)]
fn conv_builtin(a: Value, b: Value, rest: Vec<Value>) -> Result<Value, String> {
    let mode = parse_mode(&rest)?;
    if let Some(device_value) = try_conv_gpu(&a, &b, mode)? {
        return Ok(device_value);
    }
    let lhs = normalize_input(a)?;
    let rhs = normalize_input(b)?;
    let orientation = output_orientation(&lhs, &rhs);

    if lhs.len == 0 || rhs.len == 0 {
        return convert_output(Vec::new(), orientation);
    }

    let full = convolve(&lhs.data, &rhs.data);
    let shaped = apply_mode(full, mode, lhs.len, rhs.len);
    convert_output(shaped, orientation)
}

const EPS: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConvMode {
    Full,
    Same,
    Valid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrientationHint {
    Row,
    Column,
    Scalar,
    General,
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Orientation {
    Row,
    Column,
}

#[derive(Clone, Copy, Debug)]
struct ConvInputMeta {
    len: usize,
    hint: OrientationHint,
}

#[derive(Clone)]
struct ConvInput {
    data: Vec<Complex<f64>>,
    len: usize,
    hint: OrientationHint,
}

fn parse_mode(args: &[Value]) -> Result<ConvMode, String> {
    match args.len() {
        0 => Ok(ConvMode::Full),
        1 => {
            let Some(text) = tensor::value_to_string(&args[0]) else {
                return Err(
                    "conv: third argument must be the string 'full', 'same', or 'valid'"
                        .to_string(),
                );
            };
            let lowered = text.trim().to_ascii_lowercase();
            match lowered.as_str() {
                "full" => Ok(ConvMode::Full),
                "same" => Ok(ConvMode::Same),
                "valid" => Ok(ConvMode::Valid),
                _ => Err(
                    "conv: third argument must be the string 'full', 'same', or 'valid'"
                        .to_string(),
                ),
            }
        }
        _ => Err("conv: expected at most three input arguments".to_string()),
    }
}

fn try_conv_gpu(a: &Value, b: &Value, mode: ConvMode) -> Result<Option<Value>, String> {
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    let (lhs_handle, rhs_handle) = match (a, b) {
        (Value::GpuTensor(lhs), Value::GpuTensor(rhs)) => (lhs, rhs),
        _ => return Ok(None),
    };

    #[cfg(all(test, feature = "wgpu"))]
    {
        if lhs_handle.device_id != 0 || rhs_handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }

    let lhs_meta = conv_meta_from_shape(&lhs_handle.shape);
    let rhs_meta = conv_meta_from_shape(&rhs_handle.shape);

    if lhs_meta.len == 0 || rhs_meta.len == 0 {
        return Ok(None);
    }

    let supported = |meta: &ConvInputMeta| {
        matches!(
            meta.hint,
            OrientationHint::Row | OrientationHint::Column | OrientationHint::Scalar
        )
    };

    if !supported(&lhs_meta) || !supported(&rhs_meta) {
        return Ok(None);
    }

    if matches!(mode, ConvMode::Valid) && lhs_meta.len < rhs_meta.len {
        return Ok(None);
    }

    let orientation = orientation_from_hints(lhs_meta.hint, rhs_meta.hint);
    let provider_orientation = match orientation {
        Orientation::Row => ProviderConvOrientation::Row,
        Orientation::Column => ProviderConvOrientation::Column,
    };
    let provider_mode = match mode {
        ConvMode::Full => ProviderConvMode::Full,
        ConvMode::Same => ProviderConvMode::Same,
        ConvMode::Valid => ProviderConvMode::Valid,
    };

    let options = ProviderConv1dOptions {
        mode: provider_mode,
        orientation: provider_orientation,
    };

    match provider.conv1d(lhs_handle, rhs_handle, options) {
        Ok(handle) => Ok(Some(Value::GpuTensor(handle))),
        Err(err) => {
            log::trace!("conv: provider conv1d unavailable, falling back to host: {err}");
            Ok(None)
        }
    }
}

fn normalize_input(value: Value) -> Result<ConvInput, String> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            convert_tensor(tensor)
        }
        Value::Tensor(tensor) => convert_tensor(tensor),
        Value::ComplexTensor(tensor) => convert_complex_tensor(tensor),
        Value::LogicalArray(logical) => {
            tensor::logical_to_tensor(&logical).and_then(convert_tensor)
        }
        Value::Num(n) => Ok(ConvInput {
            data: vec![Complex::new(n, 0.0)],
            len: 1,
            hint: OrientationHint::Scalar,
        }),
        Value::Int(i) => Ok(ConvInput {
            data: vec![Complex::new(i.to_f64(), 0.0)],
            len: 1,
            hint: OrientationHint::Scalar,
        }),
        Value::Bool(b) => Ok(ConvInput {
            data: vec![Complex::new(if b { 1.0 } else { 0.0 }, 0.0)],
            len: 1,
            hint: OrientationHint::Scalar,
        }),
        Value::Complex(re, im) => Ok(ConvInput {
            data: vec![Complex::new(re, im)],
            len: 1,
            hint: OrientationHint::Scalar,
        }),
        other => Err(format!(
            "conv: unsupported input type {:?}; expected numeric or logical values",
            other
        )),
    }
}

fn convert_tensor(tensor: Tensor) -> Result<ConvInput, String> {
    let Tensor {
        data,
        shape: _,
        rows,
        cols,
        ..
    } = tensor;
    let len = data.len();
    let hint = classify_orientation(rows, cols, len);
    let data = data.into_iter().map(|re| Complex::new(re, 0.0)).collect();
    Ok(ConvInput { data, len, hint })
}

fn convert_complex_tensor(tensor: ComplexTensor) -> Result<ConvInput, String> {
    let ComplexTensor {
        data,
        shape: _,
        rows,
        cols,
    } = tensor;
    let len = data.len();
    let hint = classify_orientation(rows, cols, len);
    let data = data
        .into_iter()
        .map(|(re, im)| Complex::new(re, im))
        .collect();
    Ok(ConvInput { data, len, hint })
}

fn classify_orientation(rows: usize, cols: usize, len: usize) -> OrientationHint {
    if len == 0 {
        OrientationHint::Empty
    } else if rows == 1 && cols > 1 {
        OrientationHint::Row
    } else if cols == 1 && rows > 1 {
        OrientationHint::Column
    } else if rows == 1 && cols == 1 {
        OrientationHint::Scalar
    } else {
        OrientationHint::General
    }
}

fn output_orientation(lhs: &ConvInput, rhs: &ConvInput) -> Orientation {
    orientation_from_hints(lhs.hint, rhs.hint)
}

fn orientation_from_hints(lhs: OrientationHint, rhs: OrientationHint) -> Orientation {
    match lhs {
        OrientationHint::Row => Orientation::Row,
        OrientationHint::Column => Orientation::Column,
        OrientationHint::General => Orientation::Column,
        OrientationHint::Scalar | OrientationHint::Empty => match rhs {
            OrientationHint::Column | OrientationHint::General => Orientation::Column,
            OrientationHint::Row => Orientation::Row,
            OrientationHint::Scalar | OrientationHint::Empty => Orientation::Row,
        },
    }
}

fn conv_meta_from_shape(shape: &[usize]) -> ConvInputMeta {
    let len = tensor::element_count(shape);
    let (rows, cols) = shape_rows_cols(shape);
    let hint = classify_orientation(rows, cols, len);
    ConvInputMeta { len, hint }
}

fn shape_rows_cols(shape: &[usize]) -> (usize, usize) {
    if shape.is_empty() {
        (0, 0)
    } else if shape.len() == 1 {
        (1, shape[0])
    } else {
        (shape[0], shape[1])
    }
}

fn convolve(a: &[Complex<f64>], b: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut out = vec![Complex::new(0.0, 0.0); a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

fn apply_mode(
    full: Vec<Complex<f64>>,
    mode: ConvMode,
    len_a: usize,
    len_b: usize,
) -> Vec<Complex<f64>> {
    match mode {
        ConvMode::Full => full,
        ConvMode::Same => {
            if len_a == 0 {
                return Vec::new();
            }
            let start = (len_b - 1) / 2;
            let end = (start + len_a).min(full.len());
            full[start..end].to_vec()
        }
        ConvMode::Valid => {
            if len_a == 0 || len_b == 0 || len_a < len_b {
                return Vec::new();
            }
            let start = len_b - 1;
            let valid_len = len_a - len_b + 1;
            let end = (start + valid_len).min(full.len());
            full[start..end].to_vec()
        }
    }
}

fn convert_output(data: Vec<Complex<f64>>, orientation: Orientation) -> Result<Value, String> {
    let len = data.len();
    let shape = match (orientation, len) {
        (Orientation::Row, 0) => vec![1, 0],
        (Orientation::Column, 0) => vec![0, 1],
        (Orientation::Row, _) => vec![1, len],
        (Orientation::Column, _) => vec![len, 1],
    };

    let all_real = data.iter().all(|c| c.im.abs() <= EPS);
    if all_real {
        let real_data: Vec<f64> = data.into_iter().map(|c| c.re).collect();
        let tensor = Tensor::new(real_data, shape)
            .map_err(|e| format!("conv: failed to build tensor: {e}"))?;
        return Ok(tensor::tensor_into_value(tensor));
    }

    let complex_data: Vec<(f64, f64)> = data.into_iter().map(|c| (c.re, c.im)).collect();
    let tensor = ComplexTensor::new(complex_data, shape)
        .map_err(|e| format!("conv: failed to build complex tensor: {e}"))?;
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        if im.abs() <= EPS {
            return Ok(Value::Num(re));
        }
        return Ok(Value::Complex(re, im));
    }
    Ok(Value::ComplexTensor(tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, LogicalArray, Tensor};

    #[test]
    fn conv_full_row_vectors() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let b = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 3]).unwrap();
        let result = conv_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.data, vec![1.0, 3.0, 6.0, 5.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn conv_same_matches_length() {
        let a = Tensor::new(vec![3.0, 4.0, 5.0, 6.0, 7.0], vec![1, 5]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, -1.0], vec![1, 3]).unwrap();
        let result = conv_builtin(
            Value::Tensor(a),
            Value::Tensor(b.clone()),
            vec![Value::from("same")],
        )
        .expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.data, vec![4.0, 2.0, 2.0, 2.0, -6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let result_column = conv_builtin(
            Value::Tensor(Tensor::new(vec![3.0, 4.0, 5.0, 6.0, 7.0], vec![5, 1]).unwrap()),
            Value::Tensor(b),
            vec![Value::from("same")],
        )
        .expect("conv");
        match result_column {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![5, 1]);
                assert_eq!(t.data, vec![4.0, 2.0, 2.0, 2.0, -6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn conv_valid_empty_when_kernel_longer() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 3]).unwrap();
        let result = conv_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("valid")],
        )
        .unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn conv_complex_inputs() {
        let a = Value::Complex(1.0, 2.0);
        let b = Value::Complex(3.0, -1.0);
        let result = conv_builtin(a, b, Vec::new()).expect("conv");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 5.0).abs() < 1e-12);
                assert!((im - 5.0).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[test]
    fn conv_scalar_times_vector_follows_other_orientation() {
        let scalar = Value::Num(2.0);
        let vec = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
        let result = conv_builtin(scalar, Value::Tensor(vec), Vec::new()).expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![8.0, 10.0, 12.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn conv_handles_empty_inputs_with_row_orientation() {
        let empty_row = Tensor::new(Vec::<f64>::new(), vec![1, 0]).unwrap();
        let kernel = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let result =
            conv_builtin(Value::Tensor(empty_row), Value::Tensor(kernel), Vec::new()).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    fn conv_handles_empty_inputs_with_column_orientation() {
        let empty_col = Tensor::new(Vec::<f64>::new(), vec![0, 1]).unwrap();
        let kernel = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result =
            conv_builtin(Value::Tensor(empty_col), Value::Tensor(kernel), Vec::new()).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 1]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    fn conv_promotes_logical_inputs_to_double() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let kernel = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let result = conv_builtin(
            Value::LogicalArray(logical.clone()),
            Value::Tensor(kernel),
            Vec::new(),
        )
        .expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert_eq!(t.data, vec![1.0, 1.0, 1.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        // ensure logical inputs on RHS follow the same promotion path
        let lhs = Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap();
        let result = conv_builtin(Value::Tensor(lhs), Value::LogicalArray(logical), Vec::new())
            .expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert_eq!(t.data, vec![2.0, 2.0, 2.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn conv_rejects_invalid_shape_keyword() {
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let b = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = conv_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("diagonal")],
        )
        .unwrap_err();
        assert!(err.contains("third argument"));
    }

    #[test]
    fn conv_rejects_non_numeric_input() {
        let err = conv_builtin(Value::from("hi"), Value::Num(1.0), Vec::new()).unwrap_err();
        assert!(err.contains("unsupported input type"));
    }

    #[test]
    fn conv_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let signal = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
            let kernel = Tensor::new(vec![1.0, 0.0, -1.0], vec![1, 3]).unwrap();
            let host_expected = conv_builtin(
                Value::Tensor(signal.clone()),
                Value::Tensor(kernel.clone()),
                Vec::new(),
            )
            .expect("host conv");

            let sig_view = HostTensorView {
                data: &signal.data,
                shape: &signal.shape,
            };
            let ker_view = HostTensorView {
                data: &kernel.data,
                shape: &kernel.shape,
            };
            let sig_handle = provider.upload(&sig_view).expect("upload signal");
            let ker_handle = provider.upload(&ker_view).expect("upload kernel");
            let gpu_result = conv_builtin(
                Value::GpuTensor(sig_handle),
                Value::GpuTensor(ker_handle),
                Vec::new(),
            )
            .expect("gpu conv");

            let gathered = test_support::gather(gpu_result).expect("gather gpu");
            let expected = test_support::gather(host_expected).expect("gather host");
            assert_eq!(gathered.shape, expected.shape);
            assert_eq!(gathered.data, expected.data);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn conv_wgpu_matches_cpu_same_mode() {
        register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu provider");
        let provider = runmat_accelerate_api::provider().expect("provider registry");
        let signal = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let kernel = Tensor::new(vec![0.5, 0.25], vec![1, 2]).unwrap();

        let host_expected = conv_builtin(
            Value::Tensor(signal.clone()),
            Value::Tensor(kernel.clone()),
            vec![Value::from("same")],
        )
        .expect("host conv");

        let sig_view = HostTensorView {
            data: &signal.data,
            shape: &signal.shape,
        };
        let ker_view = HostTensorView {
            data: &kernel.data,
            shape: &kernel.shape,
        };
        let sig_handle = provider.upload(&sig_view).expect("upload signal");
        let ker_handle = provider.upload(&ker_view).expect("upload kernel");

        let gpu_value = conv_builtin(
            Value::GpuTensor(sig_handle),
            Value::GpuTensor(ker_handle),
            vec![Value::from("same")],
        )
        .expect("gpu conv");

        let gathered_gpu = test_support::gather(gpu_value).expect("gather gpu");
        let gathered_host = test_support::gather(host_expected).expect("gather host");
        assert_eq!(gathered_gpu.shape, gathered_host.shape);
        assert_eq!(gathered_gpu.data, gathered_host.data);
    }

    #[test]
    fn conv_same_with_integer_dimension_argument() {
        let signal = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let kernel = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = conv_builtin(
            Value::Tensor(signal),
            Value::Tensor(kernel),
            vec![Value::Int(IntValue::I32(1))],
        );
        assert!(result.is_err());
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
