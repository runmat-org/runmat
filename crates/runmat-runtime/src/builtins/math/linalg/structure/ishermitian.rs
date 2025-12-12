//! MATLAB-compatible `ishermitian` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, ProviderHermitianKind};
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
    runmat_macros::register_doc_text(
        name = "ishermitian",
        builtin_path = "crate::builtins::math::linalg::structure::ishermitian"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "ishermitian"
category: "math/linalg/structure"
keywords: ["ishermitian", "hermitian", "skew-hermitian", "matrix structure", "gpu"]
summary: "Determine whether a matrix is Hermitian or skew-Hermitian."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Falls back to gathering GPU operands when providers lack a device-side Hermitian predicate."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::structure::ishermitian::tests"
  integration: "builtins::math::linalg::structure::ishermitian::tests::ishermitian_gpu_roundtrip"
---

# What does the `ishermitian` function do in MATLAB / RunMat?
`ishermitian(A)` returns logical `true` when `A` equals its conjugate transpose (`A == A'`).
The builtin works with real and complex matrices, logical arrays, and scalars.

## How does the `ishermitian` function behave in MATLAB / RunMat?
- `ishermitian(A)` tests for Hermitian structure (`A == conj(A').`).
- Pass `'skew'` (or `'skewhermitian'`) to test for skew-Hermitian structure (`A == -conj(A')`).
- Non-square matrices immediately return `false`.
- Diagonal entries must be real for Hermitian matrices and purely imaginary (within tolerance) for skew-Hermitian matrices.
- Logical inputs are promoted to double precision before testing.
- NaN values anywhere in the matrix cause the predicate to return `false`.
- Additional singleton trailing dimensions are allowed; higher-dimensional inputs with more than two non-singleton axes raise an error.

## `ishermitian` Function GPU Execution Behaviour
RunMat keeps GPU tensors resident whenever possible. When the active acceleration provider
exposes an `ishermitian` predicate hook the comparison runs entirely on the device and only a
logical scalar is transferred back. Providers without a dedicated hook fall back to gathering the
matrix and running the CPU implementation transparently, so correctness is preserved even without
GPU specialisation.

## Examples of using the `ishermitian` function in MATLAB / RunMat

### How to check if a complex matrix is Hermitian

```matlab
A = [2   1-3i; 1+3i   5];
tf = ishermitian(A);
```

Expected output:

```matlab
tf = logical
   1
```

### How to determine that a real matrix is Hermitian

```matlab
B = [4 2 2; 2 7 3; 2 3 9];
tf = ishermitian(B);
```

Expected output:

```matlab
tf = logical
   1
```

### How to spot a matrix that is not Hermitian

```matlab
C = [1 2+i; 2-i 1+0.02i];
tf = ishermitian(C);
```

Expected output:

```matlab
tf = logical
   0
```

### How to allow numerical noise with a tolerance

```matlab
D = [1 1+1e-12; 1-1e-12 1];
tf = ishermitian(D, 1e-9);
```

Expected output:

```matlab
tf = logical
   1
```

### How to test for skew-Hermitian structure

```matlab
S = [0 2+3i; -2+3i 0];
tf = ishermitian(S, 'skew');
```

Expected output:

```matlab
tf = logical
   1
```

### How to inspect a GPU-resident matrix

```matlab
G = gpuArray([3 4-2i; 4+2i 6]);
tf = ishermitian(G);
```

Expected output:

```matlab
tf = logical
   1
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Manual `gpuArray` / `gather` calls are optional. When a provider implements the Hermitian predicate
the computation runs entirely on the GPU and only the boolean result is read back. Otherwise the
runtime gathers the tensor automatically and evaluates the predicate on the host.

## FAQ

### Does `ishermitian` work with complex matrices?
Yes. Off-diagonal elements are compared against the conjugate transpose, and diagonal elements must
be real (for Hermitian) or purely imaginary (for skew-Hermitian) within the supplied tolerance.

### What tolerance values are supported?
Pass a finite, non-negative scalar tolerance. The comparison uses the tolerance on the magnitude of
the element-wise difference between the matrix and its (signed) conjugate transpose.

### Can I test for skew-Hermitian matrices?
Yes. Provide `'skew'` or `'skewhermitian'` (optionally along with a tolerance) to require
`A == -conj(A')`.

### How are diagonal elements handled?
For Hermitian matrices the diagonal entries must be real. For skew-Hermitian matrices the diagonal
entries must have zero real part. Imaginary parts are unconstrained in skew mode.

### How does the builtin treat NaN values?
If any element of the matrix is NaN the predicate returns `false`, matching MATLAB behaviour.

### Do I need to reshape vectors before calling `ishermitian`?
Vectors are treated as matrices; column vectors are `n×1` matrices and row vectors are `1×n`
matrices. Non-square shapes always return `false`.

### What happens with empty matrices?
Empty square matrices (`0×0`) return `true`, whereas empty rectangular matrices return `false`.

### Do I need to call `gpuArray` to use `ishermitian` on the GPU?
No. RunMat automatically keeps tensors resident on the GPU when it is profitable. Explicit
`gpuArray` calls remain available for MATLAB compatibility.

## See Also
[issymmetric](./issymmetric), [eig](../factor/eig), [chol](../factor/chol), [gpuArray](../../../acceleration/gpu/gpuArray), [gather](../../../acceleration/gpu/gather)

## Source & Feedback
- View the source: [`crates/runmat-runtime/src/builtins/math/linalg/structure/ishermitian.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/structure/ishermitian.rs)
- Found a bug or behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::linalg::structure::ishermitian"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ishermitian",
    op_kind: GpuOpKind::Custom("structure_analysis"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("ishermitian")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may expose a Hermitian predicate hook; otherwise the runtime gathers the matrix and evaluates on the host.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::structure::ishermitian"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ishermitian",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Returns a host logical scalar and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "ishermitian",
    category = "math/linalg/structure",
    summary = "Determine whether a matrix is Hermitian or skew-Hermitian.",
    keywords = "ishermitian,hermitian,skew-hermitian,matrix structure,gpu",
    accel = "metadata",
    builtin_path = "crate::builtins::math::linalg::structure::ishermitian"
)]
fn ishermitian_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let (mode, tol) = parse_optional_args(&rest)?;
    match value {
        Value::GpuTensor(handle) => ishermitian_gpu(handle, mode, tol),
        other => {
            let matrix = MatrixInput::from_value(other)?;
            let result = evaluate_matrix(&matrix, mode, tol);
            Ok(Value::Bool(result))
        }
    }
}

fn ishermitian_gpu(
    handle: GpuTensorHandle,
    mode: HermitianMode,
    tol: f64,
) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let provider_mode = match mode {
            HermitianMode::Hermitian => ProviderHermitianKind::Hermitian,
            HermitianMode::Skew => ProviderHermitianKind::Skew,
        };
        match provider.ishermitian(&handle, provider_mode, tol) {
            Ok(result) => return Ok(Value::Bool(result)),
            Err(err) => {
                log::debug!("ishermitian: provider hook unavailable, falling back to host: {err}");
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    let matrix = MatrixInput::from_value(Value::Tensor(tensor))?;
    let result = evaluate_matrix(&matrix, mode, tol);
    Ok(Value::Bool(result))
}

#[derive(Clone, Copy)]
enum HermitianMode {
    Hermitian,
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
            Value::ComplexTensor(tensor) => MatrixData::Complex(tensor),
            Value::Complex(re, im) => {
                let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                    .map_err(|e| format!("ishermitian: {e}"))?;
                MatrixData::Complex(tensor)
            }
            v @ Value::Num(_) | v @ Value::Int(_) | v @ Value::Bool(_) => {
                let tensor = tensor::value_into_tensor_for("ishermitian", v)?;
                MatrixData::Real(tensor)
            }
            other => {
                return Err(format!(
                    "ishermitian: unsupported input type {:?}; expected numeric or logical matrix",
                    other
                ));
            }
        };

        let shape = data.shape();
        let (rows, cols) = matrix_dimensions_for("ishermitian", shape)?;
        Ok(Self { data, rows, cols })
    }
}

fn evaluate_matrix(matrix: &MatrixInput, mode: HermitianMode, tol: f64) -> bool {
    if matrix.rows != matrix.cols {
        return false;
    }
    match &matrix.data {
        MatrixData::Real(tensor) => is_hermitian_real(tensor, mode, tol),
        MatrixData::Complex(tensor) => is_hermitian_complex(tensor, mode, tol),
    }
}

fn parse_optional_args(args: &[Value]) -> Result<(HermitianMode, f64), String> {
    if args.len() > 2 {
        return Err("ishermitian: too many input arguments".to_string());
    }

    let mut mode = HermitianMode::Hermitian;
    let mut mode_set = false;
    let mut tol: Option<f64> = None;

    for arg in args {
        if let Some(flag) = parse_mode_flag(arg)? {
            if mode_set {
                return Err("ishermitian: duplicate symmetry flag".to_string());
            }
            mode = flag;
            mode_set = true;
            continue;
        }

        if tol.is_some() {
            return Err("ishermitian: tolerance specified more than once".to_string());
        }

        let local = parse_single_tolerance(arg)?;
        tol = Some(local);
    }

    Ok((mode, tol.unwrap_or(0.0)))
}

fn parse_mode_flag(value: &Value) -> Result<Option<HermitianMode>, String> {
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
        "skew" | "skewhermitian" | "skew-hermitian" => Ok(Some(HermitianMode::Skew)),
        "hermitian" | "nonskew" | "non-skew" | "symmetric" => Ok(Some(HermitianMode::Hermitian)),
        other => Err(format!("ishermitian: unknown flag '{other}'")),
    }
}

fn parse_single_tolerance(arg: &Value) -> Result<f64, String> {
    match parse_tolerance_arg("ishermitian", std::slice::from_ref(arg))? {
        Some(value) => Ok(value),
        None => Err("ishermitian: tolerance must be a real scalar".to_string()),
    }
}

fn is_hermitian_real(tensor: &Tensor, mode: HermitianMode, tol: f64) -> bool {
    let rows = tensor.rows();
    let cols = tensor.cols();
    debug_assert_eq!(rows, cols, "is_hermitian_real requires a square matrix");
    let data = &tensor.data;

    for col in 0..cols {
        let diag = data[col + col * rows];
        if diag.is_nan() {
            return false;
        }
        if matches!(mode, HermitianMode::Skew) && !real_within(diag, 0.0, tol) {
            return false;
        }
        for row in 0..col {
            let a = data[row + col * rows];
            let b = data[col + row * rows];
            let target = match mode {
                HermitianMode::Hermitian => b,
                HermitianMode::Skew => -b,
            };
            if !real_within(a, target, tol) {
                return false;
            }
        }
    }
    true
}

fn is_hermitian_complex(tensor: &ComplexTensor, mode: HermitianMode, tol: f64) -> bool {
    let rows = tensor.rows;
    let cols = tensor.cols;
    debug_assert_eq!(rows, cols, "is_hermitian_complex requires a square matrix");
    let data = &tensor.data;

    for col in 0..cols {
        let (diag_re, diag_im) = data[col + col * rows];
        if diag_re.is_nan() || diag_im.is_nan() {
            return false;
        }
        match mode {
            HermitianMode::Hermitian => {
                if !real_within(diag_im, 0.0, tol) {
                    return false;
                }
            }
            HermitianMode::Skew => {
                if !real_within(diag_re, 0.0, tol) {
                    return false;
                }
            }
        }
        for row in 0..col {
            let (ar, ai) = data[row + col * rows];
            let (br, bi) = data[col + row * rows];
            let (target_r, target_i) = match mode {
                HermitianMode::Hermitian => (br, -bi),
                HermitianMode::Skew => (-br, bi),
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
    (value - reference).abs() <= tol
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
    matrix_dimensions_for("ishermitian", shape)
}

pub fn ishermitian_host_real_tensor(tensor: &Tensor, skew: bool, tol: f64) -> Result<bool, String> {
    let (rows, cols) = matrix_dimensions_for("ishermitian", &tensor.shape)?;
    if rows != cols {
        return Ok(false);
    }
    let mode = if skew {
        HermitianMode::Skew
    } else {
        HermitianMode::Hermitian
    };
    Ok(is_hermitian_real(tensor, mode, tol))
}

pub fn ishermitian_host_complex_tensor(
    tensor: &ComplexTensor,
    skew: bool,
    tol: f64,
) -> Result<bool, String> {
    let (rows, cols) = matrix_dimensions_for("ishermitian", &tensor.shape)?;
    if rows != cols {
        return Ok(false);
    }
    let mode = if skew {
        HermitianMode::Skew
    } else {
        HermitianMode::Hermitian
    };
    Ok(is_hermitian_complex(tensor, mode, tol))
}

pub fn ishermitian_host_real_data(
    shape: &[usize],
    data: &[f64],
    skew: bool,
    tol: f64,
) -> Result<bool, String> {
    let (rows, cols) = matrix_dimensions_for("ishermitian", shape)?;
    if rows != cols {
        return Ok(false);
    }
    let tensor =
        Tensor::new(data.to_vec(), shape.to_vec()).map_err(|e| format!("ishermitian: {e}"))?;
    ishermitian_host_real_tensor(&tensor, skew, tol)
}

pub fn ishermitian_host_complex_data(
    shape: &[usize],
    data: &[(f64, f64)],
    skew: bool,
    tol: f64,
) -> Result<bool, String> {
    let (rows, cols) = matrix_dimensions_for("ishermitian", shape)?;
    if rows != cols {
        return Ok(false);
    }
    let tensor = ComplexTensor::new(data.to_vec(), shape.to_vec())
        .map_err(|e| format!("ishermitian: {e}"))?;
    ishermitian_host_complex_tensor(&tensor, skew, tol)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hermitian_real_matrix_returns_true() {
        let tensor = Tensor::new(
            vec![4.0, 2.0, 2.0, 2.0, 7.0, 3.0, 2.0, 3.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let result = ishermitian_builtin(Value::Tensor(tensor), Vec::new()).expect("ishermitian");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hermitian_complex_matrix_returns_true() {
        let tensor = ComplexTensor::new(
            vec![(2.0, 0.0), (1.0, 3.0), (1.0, -3.0), (5.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        let result =
            ishermitian_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("ishermitian");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_hermitian_matrix_returns_false() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 0.0), (2.0, 1.0), (2.0, -2.0), (1.0, 0.02)],
            vec![2, 2],
        )
        .unwrap();
        let result =
            ishermitian_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("ishermitian");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn skew_hermitian_flag_requires_pure_imaginary_diagonal() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (-2.0, 3.0), (2.0, 3.0), (0.0, 1.0)],
            vec![2, 2],
        )
        .unwrap();
        let result = ishermitian_builtin(Value::ComplexTensor(tensor), vec![Value::from("skew")])
            .expect("ishermitian");
        assert_eq!(result, Value::Bool(true));

        let tensor = ComplexTensor::new(
            vec![(0.01, 0.0), (-2.0, 3.0), (2.0, 3.0), (0.0, 1.0)],
            vec![2, 2],
        )
        .unwrap();
        let result = ishermitian_builtin(Value::ComplexTensor(tensor), vec![Value::from("skew")])
            .expect("ishermitian");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tolerance_enables_small_complex_deviations() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 0.0), (1.0, 1e-12), (1.0, -1e-12), (2.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        let result = ishermitian_builtin(Value::ComplexTensor(tensor), vec![Value::Num(1e-9)])
            .expect("ishermitian");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_matrix_is_promoted() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let result =
            ishermitian_builtin(Value::LogicalArray(logical), Vec::new()).expect("ishermitian");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_square_returns_false() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let result = ishermitian_builtin(Value::Tensor(tensor), Vec::new()).expect("ishermitian");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scalar_inputs_are_supported() {
        let result = ishermitian_builtin(Value::Num(3.0), Vec::new()).expect("ishermitian");
        assert_eq!(result, Value::Bool(true));

        let result =
            ishermitian_builtin(Value::Int(IntValue::I32(2)), Vec::new()).expect("ishermitian");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn skew_flag_accepts_tolerance_and_order_variants() {
        let tensor = ComplexTensor::new(
            vec![
                (0.0, 0.0),
                (-2.0, 3.0 + 1e-12),
                (2.0, 3.0 - 1e-12),
                (0.0, 1e-12),
            ],
            vec![2, 2],
        )
        .unwrap();
        let args_one = vec![Value::Num(1e-9), Value::from("skew")];
        let args_two = vec![Value::from("skewhermitian"), Value::Num(1e-9)];

        let res_one = ishermitian_builtin(Value::ComplexTensor(tensor.clone()), args_one)
            .expect("ishermitian");
        let res_two =
            ishermitian_builtin(Value::ComplexTensor(tensor), args_two).expect("ishermitian");

        assert_eq!(res_one, Value::Bool(true));
        assert_eq!(res_two, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diag_imaginary_requires_tolerance_in_hermitian_mode() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1e-10), (1.0, 0.0), (1.0, 0.0), (2.0, -1e-10)],
            vec![2, 2],
        )
        .unwrap();
        let without_tol = ishermitian_builtin(Value::ComplexTensor(tensor.clone()), Vec::new())
            .expect("ishermitian");
        assert_eq!(without_tol, Value::Bool(false));

        let with_tol = ishermitian_builtin(Value::ComplexTensor(tensor), vec![Value::Num(1e-9)])
            .expect("ishermitian");
        assert_eq!(with_tol, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nan_entries_cause_false() {
        let tensor =
            Tensor::new(vec![f64::NAN, 1.0, 1.0, 2.0], vec![2, 2]).expect("tensor construction");
        let result = ishermitian_builtin(Value::Tensor(tensor), Vec::new()).expect("ishermitian");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_unknown_flag() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = ishermitian_builtin(Value::Tensor(tensor), vec![Value::from("not-a-flag")])
            .expect_err("ishermitian should error on unknown flag");
        assert!(err.contains("unknown flag"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_negative_tolerance() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = ishermitian_builtin(Value::Tensor(tensor), vec![Value::Num(-1.0)])
            .expect_err("ishermitian should error on negative tolerance");
        assert!(err.contains("tolerance must be >= 0"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_scalar_tolerance() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let tolerance = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = ishermitian_builtin(Value::Tensor(tensor), vec![Value::Tensor(tolerance)])
            .expect_err("ishermitian should error on non-scalar tolerance");
        assert!(err.contains("tolerance must be a real scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_excess_arguments() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = ishermitian_builtin(
            Value::Tensor(tensor),
            vec![Value::Num(0.0), Value::from("skew"), Value::Num(0.0)],
        )
        .expect_err("ishermitian should error on too many inputs");
        assert!(err.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_unsupported_input_type() {
        let err = ishermitian_builtin(Value::String("abc".into()), Vec::new())
            .expect_err("ishermitian should reject strings");
        assert!(err.contains("unsupported input type"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ishermitian_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 4.0, 4.0, 6.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                ishermitian_builtin(Value::GpuTensor(handle), Vec::new()).expect("ishermitian");
            assert_eq!(result, Value::Bool(true));
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
    fn hermitian_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![5.0, 2.0, 2.0, 7.0], vec![2, 2]).unwrap();
        let cpu = ishermitian_builtin(Value::Tensor(tensor.clone()), Vec::new()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = ishermitian_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
        assert_eq!(cpu, gpu);
    }
}
