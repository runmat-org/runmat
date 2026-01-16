//! MATLAB-compatible `bandwidth` builtin with GPU-aware semantics for RunMat.

use log::debug;
use runmat_accelerate_api::{self, GpuTensorHandle};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "bandwidth",
        builtin_path = "crate::builtins::math::linalg::structure::bandwidth"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "bandwidth"
category: "math/linalg/structure"
keywords: ["bandwidth", "lower bandwidth", "upper bandwidth", "banded matrix", "structure", "gpu"]
summary: "Compute the lower and upper bandwidth of a matrix, optionally returning a single side."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Runs a device-side reduction when the acceleration provider implements the `bandwidth` hook; falls back to gathering on providers without support."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::structure::bandwidth::tests"
  integration: "builtins::math::linalg::structure::bandwidth::tests::bandwidth_gpu_roundtrip"
---

# What does the `bandwidth` function do in MATLAB / RunMat?
`bandwidth(A)` inspects the nonzero pattern of a matrix and reports two numbers:
the lower bandwidth (how many subdiagonals contain nonzeros) and the upper bandwidth
(how many superdiagonals contain nonzeros). These metrics help determine whether a
matrix is banded, which is essential for selecting the most efficient solver or factorisation.

## How does the `bandwidth` function behave in MATLAB / RunMat?
- `bandwidth(A)` returns a row vector `[lower upper]`. A diagonal matrix has `[0 0]`,
  a strictly upper-triangular matrix has `[0 k]`, and a strictly lower-triangular
  matrix has `[k 0]`, where `k` counts the furthest nonzero from the main diagonal.
- `bandwidth(A, 'lower')` returns only the lower bandwidth, while
  `bandwidth(A, 'upper')` returns only the upper bandwidth.
- Nonzero detection treats any value that is not numerically equal to zero (including
  `NaN` or `Inf`) as nonzero, matching MATLAB semantics.
- Empty matrices and all-zero matrices report `[0 0]`.
- Inputs must be numeric or logical and two-dimensional. Higher-dimensional arrays
  (with any dimension beyond the second larger than one) raise an error.
- Complex matrices are supported. A complex entry counts as nonzero if either the real
  or imaginary part is nonzero.

## `bandwidth` Function GPU Execution Behaviour
`bandwidth` leverages the active acceleration provider when available. The WGPU backend
launches a lightweight compute kernel that scans the matrix on-device and returns the
lower and upper bandwidths without transferring the entire tensor. Providers that do not
implement the `bandwidth` hook trigger a graceful fallback that gathers the matrix and
executes the CPU implementation instead. Either way, the result is returned as a small
host-side double tensor.

## Examples of using the `bandwidth` function in MATLAB / RunMat

### Checking the bandwidth of a diagonal matrix

```matlab
A = eye(4);
bw = bandwidth(A);
```

Expected output:

```matlab
bw = [0 0];
```

### Requesting only the lower bandwidth

```matlab
A = [-1 0 0; 2 3 0; 4 5 6];
lower_bw = bandwidth(A, 'lower');
```

Expected output:

```matlab
lower_bw = 2;
```

### Requesting only the upper bandwidth

```matlab
B = [1 2 0 0; 0 3 4 0; 0 0 5 6];
upper_bw = bandwidth(B, 'upper');
```

Expected output:

```matlab
upper_bw = 1;
```

### Analysing a rectangular matrix

```matlab
C = [0 0 7; 8 0 0; 0 9 0; 0 0 10];
bw = bandwidth(C);
```

Expected output:

```matlab
bw = [1 2];
```

### Working with complex-valued matrices

```matlab
Z = [1+2i 0; 0 3-4i; 5i 0];
bw = bandwidth(Z);
```

Expected output:

```matlab
bw = [2 0];
```

### Inspecting a GPU-resident matrix

```matlab
G = gpuArray([0 1 0; 2 0 3; 0 0 0]);
bw = bandwidth(G);
```

Expected output:

```matlab
bw = [1 1];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do NOT need to move data manually. When the active provider exposes the
`bandwidth` hook (the WGPU backend does), RunMat launches a device-side kernel and only
reads back the two bandwidth values. Providers without support seamlessly gather the
tensor and reuse the CPU implementation, so explicit `gpuArray` / `gather` calls remain
optional.

## FAQ

### What does a bandwidth of `[0 0]` mean?
It indicates the matrix is diagonal: all nonzero elements are on the main diagonal.

### How are rows or columns full of zeros handled?
Zero rows or columns do not increase the bandwidth; only nonzero entries affect the result.

### Does `bandwidth` treat `NaN` values as nonzero?
Yes. Any value that is not exactly zero—including `NaN` or `Inf`—counts as nonzero.

### Can I request only the lower or upper bandwidth?
Yes. Pass `'lower'` or `'upper'` as the second argument to obtain a scalar result.

### Why does `bandwidth` error on higher-dimensional arrays?
The builtin matches MATLAB and only operates on two-dimensional matrices. Use `reshape`
to collapse trailing singleton dimensions before calling `bandwidth`.

### Does `bandwidth` work with sparse matrices?
RunMat currently stores inputs as dense tensors but mirrors MATLAB's numerical semantics.
Future releases will preserve sparsity metadata while returning the same bandwidth values.

### What precision does the result use?
The result is always returned as double precision (`double`), matching MATLAB.

### Will this function keep my data on the GPU?
Yes, when the provider implements the hook: the matrix stays resident on the GPU and only
the two bandwidth values are copied back. If the provider lacks support, RunMat gathers
the tensor and computes the bandwidth on the CPU instead.

### Can I call `bandwidth` inside fused expressions?
Yes. The builtin returns a small host tensor, so it behaves like any other metadata query.

### What happens if I pass logical matrices?
Logical values are promoted to doubles (0 or 1) internally. `true` entries count as nonzero
and contribute to the bandwidth calculation.

## See Also
[diag](./diag), [tril](./tril), [triu](./triu), [spdiags](./spdiags), [issymmetric](./issymmetric), [find](./find), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- View the source: [`crates/runmat-runtime/src/builtins/math/linalg/structure/bandwidth.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/structure/bandwidth.rs)
- Found a bug or behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::linalg::structure::bandwidth"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "bandwidth",
    op_kind: GpuOpKind::Custom("structure_analysis"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("bandwidth")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "WGPU providers compute bandwidth on-device when available; runtimes gather to the host as a fallback when providers lack the hook.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::structure::bandwidth"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "bandwidth",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Structure query that returns a small host tensor; fusion treats it as a metadata operation.",
};

const BUILTIN_NAME: &str = "bandwidth";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandSelector {
    Both,
    Lower,
    Upper,
}

#[runtime_builtin(
    name = "bandwidth",
    category = "math/linalg/structure",
    summary = "Compute the lower and upper bandwidth of a matrix.",
    keywords = "bandwidth,lower bandwidth,upper bandwidth,structure,gpu",
    accel = "structure",
    builtin_path = "crate::builtins::math::linalg::structure::bandwidth"
)]
fn bandwidth_builtin(matrix: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let selector = parse_selector(&rest)?;
    let data = MatrixData::from_value(matrix)?;
    let (lower, upper) = data.bandwidth()?;
    match selector {
        BandSelector::Both => {
            let tensor = Tensor::new(vec![lower as f64, upper as f64], vec![1, 2]).map_err(|e| {
                build_runtime_error(format!("{BUILTIN_NAME}: {e}"))
                    .with_builtin(BUILTIN_NAME)
                    .build()
                    .into()
            })?;
            Ok(Value::Tensor(tensor))
        }
        BandSelector::Lower => Ok(Value::Num(lower as f64)),
        BandSelector::Upper => Ok(Value::Num(upper as f64)),
    }
}

fn parse_selector(args: &[Value]) -> BuiltinResult<BandSelector> {
    match args.len() {
        0 => Ok(BandSelector::Both),
        1 => {
            let text = tensor::value_to_string(&args[0]).ok_or_else(|| {
                build_runtime_error("bandwidth: selector must be a character vector or string scalar")
                    .with_builtin(BUILTIN_NAME)
                    .build()
                    .into()
            })?;
            let trimmed = text.trim();
            let lowered = trimmed.to_ascii_lowercase();
            match lowered.as_str() {
                "lower" => Ok(BandSelector::Lower),
                "upper" => Ok(BandSelector::Upper),
                other => Err(build_runtime_error(format!(
                    "bandwidth: unrecognized selector '{other}'; expected 'lower' or 'upper'"
                ))
                .with_builtin(BUILTIN_NAME)
                .build()
                .into()),
            }
        }
        _ => Err(build_runtime_error("bandwidth: too many input arguments")
            .with_builtin(BUILTIN_NAME)
            .build()
            .into()),
    }
}

fn value_into_tensor_for(name: &str, value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(t) => Ok(t),
        Value::LogicalArray(logical) => logical_to_tensor(name, &logical),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|e| {
            build_runtime_error(format!("{name}: {e}"))
                .with_builtin(name)
                .build()
                .into()
        }),
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| {
            build_runtime_error(format!("{name}: {e}"))
                .with_builtin(name)
                .build()
                .into()
        }),
        Value::Bool(b) => Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1]).map_err(|e| {
            build_runtime_error(format!("{name}: {e}"))
                .with_builtin(name)
                .build()
                .into()
        }),
        other => Err(build_runtime_error(format!(
            "{name}: unsupported input type {:?}; expected numeric or logical values",
            other
        ))
        .with_builtin(name)
        .build()
        .into()),
    }
}

fn logical_to_tensor(name: &str, logical: &LogicalArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = logical
        .data
        .iter()
        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
        .collect();
    Tensor::new(data, logical.shape.clone()).map_err(|e| {
        build_runtime_error(format!("{name}: {e}"))
            .with_builtin(name)
            .build()
            .into()
    })
}

enum MatrixData {
    Real(Tensor),
    Complex(ComplexTensor),
    Gpu(GpuTensorHandle),
}

impl MatrixData {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::ComplexTensor(ct) => Ok(Self::Complex(ct)),
            Value::Complex(re, im) => {
                let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| {
                    build_runtime_error(format!("{BUILTIN_NAME}: {e}"))
                        .with_builtin(BUILTIN_NAME)
                        .build()
                        .into()
                })?;
                Ok(Self::Complex(tensor))
            }
            Value::GpuTensor(handle) => Ok(Self::Gpu(handle)),
            other => {
                let tensor = value_into_tensor_for(BUILTIN_NAME, other)?;
                Ok(Self::Real(tensor))
            }
        }
    }

    fn bandwidth(&self) -> BuiltinResult<(usize, usize)> {
        match self {
            MatrixData::Real(tensor) => bandwidth_host_real_tensor(tensor),
            MatrixData::Complex(tensor) => bandwidth_host_complex_tensor(tensor),
            MatrixData::Gpu(handle) => bandwidth_gpu(handle),
        }
    }
}

fn bandwidth_gpu(handle: &GpuTensorHandle) -> BuiltinResult<(usize, usize)> {
    let (rows, cols) = ensure_matrix_shape(&handle.shape)?;
    if rows == 0 || cols == 0 {
        return Ok((0, 0));
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        match provider.bandwidth(handle) {
            Ok(result) => {
                let lower = result.lower as usize;
                let upper = result.upper as usize;
                return Ok((lower, upper));
            }
            Err(err) => {
                debug!("bandwidth: provider bandwidth fallback: {err}");
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor(handle)?;
    bandwidth_host_real_tensor(&tensor)
}

pub fn ensure_matrix_shape(shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    match shape.len() {
        0 => Ok((1, 1)),
        1 => Ok((1, shape[0])),
        _ => {
            if shape[2..].iter().any(|&dim| dim > 1) {
                Err(build_runtime_error("bandwidth: input must be a 2-D matrix")
                    .with_builtin(BUILTIN_NAME)
                    .build()
                    .into())
            } else {
                Ok((shape[0], shape[1]))
            }
        }
    }
}

pub fn bandwidth_host_real_data(shape: &[usize], data: &[f64]) -> BuiltinResult<(usize, usize)> {
    let (rows, cols) = ensure_matrix_shape(shape)?;
    Ok(compute_real_bandwidth(rows, cols, data))
}

pub fn bandwidth_host_complex_data(
    shape: &[usize],
    data: &[(f64, f64)],
) -> BuiltinResult<(usize, usize)> {
    let (rows, cols) = ensure_matrix_shape(shape)?;
    Ok(compute_complex_bandwidth(rows, cols, data))
}

pub fn bandwidth_host_real_tensor(tensor: &Tensor) -> BuiltinResult<(usize, usize)> {
    bandwidth_host_real_data(&tensor.shape, &tensor.data)
}

pub fn bandwidth_host_complex_tensor(tensor: &ComplexTensor) -> BuiltinResult<(usize, usize)> {
    bandwidth_host_complex_data(&tensor.shape, &tensor.data)
}

fn compute_real_bandwidth(rows: usize, cols: usize, data: &[f64]) -> (usize, usize) {
    if rows == 0 || cols == 0 {
        return (0, 0);
    }
    let mut lower = 0usize;
    let mut upper = 0usize;
    let stride = rows;
    for col in 0..cols {
        for row in 0..rows {
            let idx = row + col * stride;
            if idx >= data.len() {
                break;
            }
            let value = data[idx];
            if value != 0.0 || value.is_nan() {
                if row >= col {
                    lower = lower.max(row - col);
                } else {
                    upper = upper.max(col - row);
                }
            }
        }
    }
    (lower, upper)
}

fn compute_complex_bandwidth(rows: usize, cols: usize, data: &[(f64, f64)]) -> (usize, usize) {
    if rows == 0 || cols == 0 {
        return (0, 0);
    }
    let mut lower = 0usize;
    let mut upper = 0usize;
    let stride = rows;
    for col in 0..cols {
        for row in 0..rows {
            let idx = row + col * stride;
            if idx >= data.len() {
                break;
            }
            let (re, im) = data[idx];
            if !(re == 0.0 && im == 0.0) {
                if row >= col {
                    lower = lower.max(row - col);
                } else {
                    upper = upper.max(col - row);
                }
            }
        }
    }
    (lower, upper)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::LogicalArray;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_diagonal_matrix() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let value = Value::Tensor(tensor);
        let result = bandwidth_builtin(value, Vec::new()).expect("bandwidth");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![0.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_lower_selector() {
        let tensor = Tensor::new(
            vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 0.0, 0.0, 1.0],
            vec![3, 3],
        )
        .unwrap();
        let args = vec![Value::from("lower")];
        let result = bandwidth_builtin(Value::Tensor(tensor), args).expect("bandwidth");
        match result {
            Value::Num(n) => assert_eq!(n, 2.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_upper_selector() {
        let tensor = Tensor::new(
            vec![1.0, 0.0, 0.0, 2.0, 4.0, 0.0, 3.0, 5.0, 6.0],
            vec![3, 3],
        )
        .unwrap();
        let args = vec![Value::from("upper")];
        let result = bandwidth_builtin(Value::Tensor(tensor), args).expect("bandwidth");
        match result {
            Value::Num(n) => assert_eq!(n, 2.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_complex_matrix() {
        let data = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (0.0, 0.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result =
            bandwidth_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("bandwidth");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_rectangular_matrix() {
        let tensor = Tensor::new(
            vec![0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 7.0, 0.0, 0.0, 10.0],
            vec![4, 3],
        )
        .unwrap();
        let result = bandwidth_builtin(Value::Tensor(tensor), Vec::new()).expect("bandwidth");
        match result {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 2.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_empty_matrix_returns_zero() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result = bandwidth_builtin(Value::Tensor(tensor), Vec::new()).expect("bandwidth");
        match result {
            Value::Tensor(t) => assert_eq!(t.data, vec![0.0, 0.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_nan_counts_as_nonzero() {
        let tensor =
            Tensor::new(vec![0.0, f64::NAN, 0.0, 0.0], vec![2, 2]).expect("tensor construction");
        let result = bandwidth_builtin(Value::Tensor(tensor), Vec::new()).expect("bandwidth");
        match result {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 0.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_logical_input_supported() {
        let logical = LogicalArray::new(vec![1, 1, 1, 0], vec![2, 2]).expect("logical array");
        let result =
            bandwidth_builtin(Value::LogicalArray(logical), Vec::new()).expect("bandwidth");
        match result {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_selector_validation() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err =
            bandwidth_builtin(Value::Tensor(tensor), vec![Value::from("middle")]).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("lower") && message.contains("upper"),
            "unexpected error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_rejects_higher_dimensions() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 1, 2]).unwrap();
        let err = bandwidth_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("2-D"), "unexpected error message: {message}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 2.0, 0.0, 0.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                bandwidth_builtin(Value::GpuTensor(handle), Vec::new()).expect("bandwidth");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 2]);
            assert_eq!(gathered.data, vec![1.0, 0.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn bandwidth_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let tensor = Tensor::new(
            vec![0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 6.0],
            vec![3, 3],
        )
        .unwrap();
        let cpu = super::bandwidth_host_real_tensor(&tensor).expect("cpu bandwidth");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_meta = provider.bandwidth(&handle).expect("provider bandwidth");
        assert_eq!(gpu_meta.lower as usize, cpu.0);
        assert_eq!(gpu_meta.upper as usize, cpu.1);

        let result =
            bandwidth_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("bandwidth");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 2]);
        assert_eq!(gathered.data, vec![cpu.0 as f64, cpu.1 as f64]);
        let _ = provider.free(&handle);
    }
}
