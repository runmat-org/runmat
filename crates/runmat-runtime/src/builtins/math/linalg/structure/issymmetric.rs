//! MATLAB-compatible `issymmetric` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, ProviderSymmetryKind};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::linalg::{matrix_dimensions_for, parse_tolerance_arg};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "issymmetric")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "issymmetric"
category: "math/linalg/structure"
keywords: ["issymmetric", "symmetric matrix", "skew-symmetric", "matrix structure", "gpu"]
summary: "Test whether a matrix is symmetric or skew-symmetric, optionally within a tolerance."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Falls back to gathering GPU operands when providers lack a device-side symmetry predicate."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::structure::issymmetric::tests"
  integration: "builtins::math::linalg::structure::issymmetric::tests::issymmetric_gpu_roundtrip"
---

# What does the `issymmetric` function do in MATLAB / RunMat?
`issymmetric(A)` returns logical `true` when a numeric or logical matrix `A` is symmetric
about its main diagonal (`A == A.'`) and `false` otherwise.

## How does the `issymmetric` function behave in MATLAB / RunMat?
- Works with scalars, vectors (treated as matrices), and higher-rank arrays whose trailing
  dimensions are singleton (MATLAB-compatible matrix semantics). An error is raised when
  the input has more than two non-singleton dimensions.
- Non-square matrices immediately return `false`.
- Logical inputs are promoted to double precision (`true → 1.0`, `false → 0.0`) before
  testing.
- Complex inputs are compared without conjugation; use `ishermitian` for conjugate symmetry.
- Floating-point tolerance can be supplied to account for numerical noise.
- Pass `'skew'` to test skew-symmetry (`A == -A.'`). Diagonal entries must be zero (within
  the tolerance) when `skew` mode is active.

## `issymmetric` Function GPU Execution Behaviour
RunMat keeps GPU tensors resident whenever feasible. When the active acceleration provider
exposes a symmetry predicate hook (`issymmetric`), the test runs entirely on the device and
returns a host logical scalar. Providers without that hook gracefully fall back to downloading
the matrix, so results remain correct even without GPU specialisation.

## Examples of using the `issymmetric` function in MATLAB / RunMat

### Checking whether a matrix is symmetric

```matlab
A = [2 1 1; 1 3 4; 1 4 5];
tf = issymmetric(A);
```

Expected output:

```matlab
tf = logical
   1
```

### Allowing numerical noise with a tolerance

```matlab
A = [1 1+1e-12; 1-1e-12 1];
tf = issymmetric(A, 1e-9);
```

Expected output:

```matlab
tf = logical
   1
```

### Detecting a skew-symmetric matrix

```matlab
B = [0 -2 4; 2 0 -3; -4 3 0];
tf = issymmetric(B, 'skew');
```

Expected output:

```matlab
tf = logical
   1
```

### Handling non-square matrices

```matlab
C = [1 2 3; 4 5 6];
tf = issymmetric(C);
```

Expected output:

```matlab
tf = logical
   0
```

### Working with complex-valued matrices

```matlab
Z = [1+2i 3-4i; 3-4i 5+6i];
tf = issymmetric(Z);
```

Expected output:

```matlab
tf = logical
   1
```

### Inspecting a GPU-resident matrix

```matlab
G = gpuArray([0 5; 5 9]);
tf = issymmetric(G);
```

Expected output:

```matlab
tf = logical
   1
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Manual `gpuArray` / `gather` calls are optional. When a provider implements the symmetry hook,
RunMat executes the predicate in-place on the GPU and only transfers the scalar result.
Otherwise the runtime gathers the tensor transparently and reuses the CPU path, preserving
correctness with a minor residency cost.

## FAQ

### Does `issymmetric` accept tolerance arguments?
Yes. Pass a non-negative scalar tolerance as the second or third argument. The comparison
uses an absolute tolerance on the element-wise difference (or sum for skew tests).

### What strings are accepted for the skew flag?
Use `'skew'` to test skew-symmetry and `'nonskew'` (or omit the flag) for the default
symmetry test.

### How are diagonal elements handled in skew mode?
Diagonal elements must be zero (within the tolerance) because a skew-symmetric matrix
satisfies `A(i,i) = -A(i,i)`.

### Are NaN values considered symmetric?
No. Any NaN encountered off or on the diagonal causes the test to return `false`, matching
MATLAB behaviour.

### Do logical matrices work?
Yes. Logical inputs are promoted to double precision and checked using the same rules as
numeric matrices.

### Does `issymmetric` conjugate complex inputs?
No. The builtin compares complex entries without conjugation. Use `ishermitian` if you
need conjugate symmetry.

### What happens with higher-dimensional arrays?
`issymmetric` raises an error when the input has more than two non-singleton dimensions.
Reshape the data to a 2-D matrix before calling it.

### Can I combine the skew flag and tolerance?
Yes. You can call `issymmetric(A, 'skew', tol)` or `issymmetric(A, tol, 'skew')`. The order
of the optional arguments does not matter.

### Is an empty matrix symmetric?
Yes. Empty square matrices (`0x0`) return logical `true`, while non-square empty matrices
return `false`.

### Does the result depend on GPU availability?
No. You receive the same logical answer regardless of whether a GPU provider is registered.
Only the execution strategy changes (device-side predicate vs. host fallback).

## See Also
[bandwidth](./bandwidth), [chol](../factor/chol), [eig](../factor/eig), [gpuArray](../../../acceleration/gpu/gpuArray), [gather](../../../acceleration/gpu/gather)

## Source & Feedback
- View the source: [`crates/runmat-runtime/src/builtins/math/linalg/structure/issymmetric.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/structure/issymmetric.rs)
- Found a bug or behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "issymmetric",
    op_kind: GpuOpKind::Custom("structure_analysis"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("issymmetric")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may supply a symmetry predicate hook; otherwise the runtime gathers the tensor and evaluates on the host.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "issymmetric",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Returns a host logical scalar and acts as a fusion sink.",
};

#[runtime_builtin(
    name = "issymmetric",
    category = "math/linalg/structure",
    summary = "Test whether a matrix is symmetric or skew-symmetric.",
    keywords = "issymmetric,symmetric,skew-symmetric,matrix structure,gpu",
    accel = "metadata"
)]
fn issymmetric_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let (mode, tol) = parse_optional_args(&rest)?;
    match value {
        Value::GpuTensor(handle) => issymmetric_gpu(handle, mode, tol),
        other => {
            let matrix = MatrixInput::from_value(other)?;
            let result = evaluate_matrix(matrix, mode, tol);
            Ok(Value::Bool(result))
        }
    }
}

fn issymmetric_gpu(handle: GpuTensorHandle, mode: SymmetryMode, tol: f64) -> Result<Value, String> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let kind = match mode {
            SymmetryMode::Symmetric => ProviderSymmetryKind::Symmetric,
            SymmetryMode::Skew => ProviderSymmetryKind::Skew,
        };
        let tried = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            provider.issymmetric(&handle, kind, tol)
        }));
        if let Ok(Ok(flag)) = tried {
            return Ok(Value::Bool(flag));
        }
        log::debug!("issymmetric: provider path failed or panicked; falling back to host");
    }

    let tensor = gpu_helpers::gather_tensor(&handle)?;
    let matrix = MatrixInput::from_value(Value::Tensor(tensor))?;
    let result = evaluate_matrix(matrix, mode, tol);
    Ok(Value::Bool(result))
}

#[derive(Clone, Copy)]
enum SymmetryMode {
    Symmetric,
    Skew,
}

struct MatrixInput {
    data: MatrixData,
    rows: usize,
    cols: usize,
}

enum MatrixData {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl MatrixData {
    fn shape(&self) -> &[usize] {
        match self {
            MatrixData::Real(t) => &t.shape,
            MatrixData::Complex(t) => &t.shape,
        }
    }
}

impl MatrixInput {
    fn from_value(value: Value) -> Result<Self, String> {
        let data = match value {
            Value::Tensor(tensor) => MatrixData::Real(tensor),
            Value::LogicalArray(logical) => {
                let tensor = tensor::logical_to_tensor(&logical)?;
                MatrixData::Real(tensor)
            }
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor(&handle)?;
                MatrixData::Real(tensor)
            }
            Value::ComplexTensor(tensor) => MatrixData::Complex(tensor),
            Value::Complex(re, im) => {
                let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                    .map_err(|e| format!("issymmetric: {e}"))?;
                MatrixData::Complex(tensor)
            }
            Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
                let tensor = tensor::value_into_tensor_for("issymmetric", value)?;
                MatrixData::Real(tensor)
            }
            other => {
                return Err(format!(
                    "issymmetric: unsupported input type {:?}; expected numeric or logical matrix",
                    other
                ));
            }
        };

        let shape = data.shape();
        let (rows, cols) = matrix_dimensions_for("issymmetric", shape)?;
        Ok(Self { data, rows, cols })
    }
}

fn evaluate_matrix(matrix: MatrixInput, mode: SymmetryMode, tol: f64) -> bool {
    if matrix.rows != matrix.cols {
        return false;
    }
    match matrix.data {
        MatrixData::Real(tensor) => is_symmetric_real(&tensor, mode, tol),
        MatrixData::Complex(tensor) => is_symmetric_complex(&tensor, mode, tol),
    }
}

fn parse_optional_args(args: &[Value]) -> Result<(SymmetryMode, f64), String> {
    if args.len() > 2 {
        return Err("issymmetric: too many input arguments".to_string());
    }

    let mut mode = SymmetryMode::Symmetric;
    let mut mode_set = false;
    let mut tol: Option<f64> = None;

    for arg in args {
        if let Some(flag) = parse_mode_flag(arg)? {
            if mode_set {
                return Err("issymmetric: duplicate symmetry flag".to_string());
            }
            mode = flag;
            mode_set = true;
            continue;
        }

        if tol.is_some() {
            return Err("issymmetric: tolerance specified more than once".to_string());
        }

        let local = parse_single_tolerance(arg)?;
        tol = Some(local);
    }

    Ok((mode, tol.unwrap_or(0.0)))
}

fn parse_mode_flag(value: &Value) -> Result<Option<SymmetryMode>, String> {
    let text = match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    };

    let Some(raw) = text else {
        return Ok(None);
    };

    let lowered = raw.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "skew" => Ok(Some(SymmetryMode::Skew)),
        "nonskew" | "symmetric" => Ok(Some(SymmetryMode::Symmetric)),
        other => Err(format!("issymmetric: unknown flag '{other}'")),
    }
}

fn parse_single_tolerance(arg: &Value) -> Result<f64, String> {
    match parse_tolerance_arg("issymmetric", std::slice::from_ref(arg))? {
        Some(value) => Ok(value),
        None => Err("issymmetric: tolerance must be a real scalar".to_string()),
    }
}

fn is_symmetric_real(tensor: &Tensor, mode: SymmetryMode, tol: f64) -> bool {
    let rows = tensor.rows();
    let cols = tensor.cols();
    debug_assert_eq!(rows, cols, "is_symmetric_real requires a square matrix");
    let data = &tensor.data;

    for col in 0..cols {
        if matches!(mode, SymmetryMode::Skew) {
            let diag = data[col + col * rows];
            if !real_within(diag, 0.0, tol) {
                return false;
            }
        }
        for row in 0..col {
            let a = data[row + col * rows];
            let b = data[col + row * rows];
            let (diff, reference) = match mode {
                SymmetryMode::Symmetric => (a, b),
                SymmetryMode::Skew => (a, -b),
            };
            if !real_within(diff, reference, tol) {
                return false;
            }
        }
    }
    true
}

fn is_symmetric_complex(tensor: &ComplexTensor, mode: SymmetryMode, tol: f64) -> bool {
    let rows = tensor.rows;
    let cols = tensor.cols;
    debug_assert_eq!(rows, cols, "is_symmetric_complex requires a square matrix");
    let data = &tensor.data;

    for col in 0..cols {
        if matches!(mode, SymmetryMode::Skew) {
            let (re, im) = data[col + col * rows];
            if !complex_within(re, im, 0.0, 0.0, tol) {
                return false;
            }
        }
        for row in 0..col {
            let (ar, ai) = data[row + col * rows];
            let (br, bi) = data[col + row * rows];
            let (target_r, target_i) = match mode {
                SymmetryMode::Symmetric => (br, bi),
                SymmetryMode::Skew => (-br, -bi),
            };
            if !complex_within(ar, ai, target_r, target_i, tol) {
                return false;
            }
        }
    }
    true
}

fn real_within(value: f64, reference: f64, tol: f64) -> bool {
    if value == reference {
        return true;
    }
    if !value.is_finite() || !reference.is_finite() {
        return false;
    }
    let diff = (value - reference).abs();
    diff <= tol
}

fn complex_within(re: f64, im: f64, ref_re: f64, ref_im: f64, tol: f64) -> bool {
    if re == ref_re && im == ref_im {
        return true;
    }
    if !re.is_finite() || !im.is_finite() || !ref_re.is_finite() || !ref_im.is_finite() {
        return false;
    }
    let diff_r = re - ref_re;
    let diff_i = im - ref_im;
    diff_r.hypot(diff_i) <= tol
}

pub fn ensure_matrix_shape(shape: &[usize]) -> Result<(usize, usize), String> {
    matrix_dimensions_for("issymmetric", shape)
}

pub fn issymmetric_host_real_tensor(tensor: &Tensor, skew: bool, tol: f64) -> Result<bool, String> {
    let (rows, cols) = matrix_dimensions_for("issymmetric", &tensor.shape)?;
    if rows != cols {
        return Ok(false);
    }
    let mode = if skew {
        SymmetryMode::Skew
    } else {
        SymmetryMode::Symmetric
    };
    Ok(is_symmetric_real(tensor, mode, tol))
}

pub fn issymmetric_host_complex_tensor(
    tensor: &ComplexTensor,
    skew: bool,
    tol: f64,
) -> Result<bool, String> {
    let (rows, cols) = matrix_dimensions_for("issymmetric", &tensor.shape)?;
    if rows != cols {
        return Ok(false);
    }
    let mode = if skew {
        SymmetryMode::Skew
    } else {
        SymmetryMode::Symmetric
    };
    Ok(is_symmetric_complex(tensor, mode, tol))
}

pub fn issymmetric_host_real_data(
    shape: &[usize],
    data: &[f64],
    skew: bool,
    tol: f64,
) -> Result<bool, String> {
    let (rows, cols) = matrix_dimensions_for("issymmetric", shape)?;
    if rows != cols {
        return Ok(false);
    }
    let tensor =
        Tensor::new(data.to_vec(), shape.to_vec()).map_err(|e| format!("issymmetric: {e}"))?;
    issymmetric_host_real_tensor(&tensor, skew, tol)
}

pub fn issymmetric_host_complex_data(
    shape: &[usize],
    data: &[(f64, f64)],
    skew: bool,
    tol: f64,
) -> Result<bool, String> {
    let (rows, cols) = matrix_dimensions_for("issymmetric", shape)?;
    if rows != cols {
        return Ok(false);
    }
    let tensor = ComplexTensor::new(data.to_vec(), shape.to_vec())
        .map_err(|e| format!("issymmetric: {e}"))?;
    issymmetric_host_complex_tensor(&tensor, skew, tol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    use runmat_builtins::{IntValue, LogicalArray};

    #[test]
    fn symmetric_matrix_returns_true() {
        let tensor = Tensor::new(
            vec![2.0, 1.0, 1.0, 1.0, 3.0, 4.0, 1.0, 4.0, 5.0],
            vec![3, 3],
        )
        .unwrap();
        let result = issymmetric_builtin(Value::Tensor(tensor), Vec::new()).expect("issymmetric");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn nonsymmetric_matrix_returns_false() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = issymmetric_builtin(Value::Tensor(tensor), Vec::new()).expect("issymmetric");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn skew_symmetric_flag_requires_zero_diagonal() {
        let tensor = Tensor::new(vec![0.0, 2.0, -2.0, 0.0], vec![2, 2]).unwrap();
        let result =
            issymmetric_builtin(Value::Tensor(tensor), vec![Value::from("skew")]).expect("skew");
        assert_eq!(result, Value::Bool(true));

        let tensor = Tensor::new(vec![1.0, -2.0, 2.0, 1.0], vec![2, 2]).unwrap();
        let result =
            issymmetric_builtin(Value::Tensor(tensor), vec![Value::from("skew")]).expect("skew");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn tolerance_allows_small_deviations() {
        let tensor = Tensor::new(vec![1.0, 1.0 + 1e-12, 1.0 - 1e-12, 1.0], vec![2, 2]).unwrap();
        let result = issymmetric_builtin(Value::Tensor(tensor), vec![Value::Num(1e-9)])
            .expect("issymmetric");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn complex_matrix_symmetry() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 2.0), (3.0, -4.0), (3.0, -4.0), (5.0, 6.0)],
            vec![2, 2],
        )
        .unwrap();
        let result =
            issymmetric_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("issymmetric");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn logical_matrix_promoted() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let result =
            issymmetric_builtin(Value::LogicalArray(logical), Vec::new()).expect("issymmetric");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn non_square_returns_false() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let result = issymmetric_builtin(Value::Tensor(tensor), Vec::new()).expect("issymmetric");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn higher_dimensional_inputs_error() {
        let tensor = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let err = issymmetric_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err();
        assert!(
            err.contains("inputs must be 2-D matrices"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn tolerance_and_flag_in_any_order() {
        let tensor = Tensor::new(vec![0.0, 1.0, -1.0000000001, 0.0], vec![2, 2]).unwrap();
        let result = issymmetric_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("skew"), Value::Num(1e-9)],
        )
        .expect("issymmetric");
        assert_eq!(result, Value::Bool(true));

        let result = issymmetric_builtin(
            Value::Tensor(tensor),
            vec![Value::Num(1e-9), Value::from("skew")],
        )
        .expect("issymmetric");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn invalid_flag_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err =
            issymmetric_builtin(Value::Tensor(tensor), vec![Value::from("diagonal")]).unwrap_err();
        assert!(
            err.contains("unknown flag"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn duplicate_tolerance_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err = issymmetric_builtin(
            Value::Tensor(tensor),
            vec![Value::Num(1e-9), Value::Num(1e-6)],
        )
        .unwrap_err();
        assert!(
            err.contains("tolerance specified more than once"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn negative_tolerance_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err = issymmetric_builtin(Value::Tensor(tensor), vec![Value::Num(-1.0)]).unwrap_err();
        assert!(
            err.contains("tolerance must be >= 0"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn scalar_inputs_follow_rules() {
        let result = issymmetric_builtin(Value::Num(5.0), Vec::new()).expect("issymmetric scalar");
        assert_eq!(result, Value::Bool(true));

        let result = issymmetric_builtin(Value::Int(IntValue::I32(0)), vec![Value::from("skew")])
            .expect("issymmetric skew scalar");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn issymmetric_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                issymmetric_builtin(Value::GpuTensor(handle), Vec::new()).expect("issymmetric");
            assert_eq!(result, Value::Bool(true));
        });
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn issymmetric_wgpu_matches_cpu() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());

        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let cpu = issymmetric_builtin(Value::Tensor(tensor.clone()), Vec::new()).unwrap();

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().unwrap();
        let handle = provider.upload(&view).unwrap();
        let gpu = issymmetric_builtin(Value::GpuTensor(handle.clone()), Vec::new()).unwrap();
        assert_eq!(cpu, gpu);

        let skew = Tensor::new(vec![0.0, 1.0, -1.0 - 1.0e-9, 0.0], vec![2, 2]).unwrap();
        let cpu_skew = issymmetric_builtin(
            Value::Tensor(skew.clone()),
            vec![Value::from("skew"), Value::Num(1.0e-6)],
        )
        .unwrap();
        let view_skew = runmat_accelerate_api::HostTensorView {
            data: &skew.data,
            shape: &skew.shape,
        };
        let handle_skew = provider.upload(&view_skew).unwrap();
        let gpu_skew = issymmetric_builtin(
            Value::GpuTensor(handle_skew.clone()),
            vec![Value::from("skew"), Value::Num(1.0e-6)],
        )
        .unwrap();
        assert_eq!(cpu_skew, gpu_skew);

        let _ = provider.free(&handle);
        let _ = provider.free(&handle_skew);
    }
}
