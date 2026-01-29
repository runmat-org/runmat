//! MATLAB-compatible `ishermitian` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, ProviderHermitianKind};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

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

const BUILTIN_NAME: &str = "ishermitian";

fn runtime_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

#[runtime_builtin(
    name = "ishermitian",
    category = "math/linalg/structure",
    summary = "Determine whether a matrix is Hermitian or skew-Hermitian.",
    keywords = "ishermitian,hermitian,skew-hermitian,matrix structure,gpu",
    accel = "metadata",
    builtin_path = "crate::builtins::math::linalg::structure::ishermitian"
)]
async fn ishermitian_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (mode, tol) = parse_optional_args(&rest)?;
    match value {
        Value::GpuTensor(handle) => ishermitian_gpu(handle, mode, tol).await,
        other => {
            let matrix = MatrixInput::from_value(other)?;
            let result = evaluate_matrix(&matrix, mode, tol);
            Ok(Value::Bool(result))
        }
    }
}

async fn ishermitian_gpu(
    handle: GpuTensorHandle,
    mode: HermitianMode,
    tol: f64,
) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let provider_mode = match mode {
            HermitianMode::Hermitian => ProviderHermitianKind::Hermitian,
            HermitianMode::Skew => ProviderHermitianKind::Skew,
        };
        match provider.ishermitian(&handle, provider_mode, tol).await {
            Ok(result) => return Ok(Value::Bool(result)),
            Err(err) => {
                log::debug!("ishermitian: provider hook unavailable, falling back to host: {err}");
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
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
    fn from_value(value: Value) -> BuiltinResult<Self> {
        let data = match value {
            Value::Tensor(tensor) => MatrixData::Real(tensor),
            Value::LogicalArray(logical) => {
                let tensor = logical_to_tensor(BUILTIN_NAME, &logical)?;
                MatrixData::Real(tensor)
            }
            Value::ComplexTensor(tensor) => MatrixData::Complex(tensor),
            Value::Complex(re, im) => {
                let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                    .map_err(|e| runtime_error(BUILTIN_NAME, format!("{BUILTIN_NAME}: {e}")))?;
                MatrixData::Complex(tensor)
            }
            v @ Value::Num(_) | v @ Value::Int(_) | v @ Value::Bool(_) => {
                let tensor = value_into_tensor_for(BUILTIN_NAME, v)?;
                MatrixData::Real(tensor)
            }
            other => {
                return Err(runtime_error(
                    BUILTIN_NAME,
                    format!(
                        "ishermitian: unsupported input type {:?}; expected numeric or logical matrix",
                        other
                    ),
                ));
            }
        };

        let shape = data.shape();
        let (rows, cols) = matrix_dimensions_for(BUILTIN_NAME, shape)?;
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

fn parse_optional_args(args: &[Value]) -> BuiltinResult<(HermitianMode, f64)> {
    if args.len() > 2 {
        return Err(runtime_error(
            BUILTIN_NAME,
            "ishermitian: too many input arguments",
        ));
    }

    let mut mode = HermitianMode::Hermitian;
    let mut mode_set = false;
    let mut tol: Option<f64> = None;

    for arg in args {
        if let Some(flag) = parse_mode_flag(arg)? {
            if mode_set {
                return Err(runtime_error(
                    BUILTIN_NAME,
                    "ishermitian: duplicate symmetry flag",
                ));
            }
            mode = flag;
            mode_set = true;
            continue;
        }

        if tol.is_some() {
            return Err(runtime_error(
                BUILTIN_NAME,
                "ishermitian: tolerance specified more than once",
            ));
        }

        let local = parse_single_tolerance(arg)?;
        tol = Some(local);
    }

    Ok((mode, tol.unwrap_or(0.0)))
}

fn parse_mode_flag(value: &Value) -> BuiltinResult<Option<HermitianMode>> {
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
        other => Err(runtime_error(
            BUILTIN_NAME,
            format!("ishermitian: unknown flag '{other}'"),
        )),
    }
}

fn parse_single_tolerance(arg: &Value) -> BuiltinResult<f64> {
    let value = parse_tolerance_value(BUILTIN_NAME, arg)?;
    Ok(value)
}

fn parse_tolerance_value(name: &str, value: &Value) -> BuiltinResult<f64> {
    let raw = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => t.data[0],
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Value::LogicalArray(l) if l.len() == 1 => {
            if l.data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(runtime_error(
                name,
                format!("{name}: tolerance must be a real scalar, got {other:?}"),
            ))
        }
    };
    if !raw.is_finite() {
        return Err(runtime_error(
            name,
            format!("{name}: tolerance must be finite"),
        ));
    }
    if raw < 0.0 {
        return Err(runtime_error(
            name,
            format!("{name}: tolerance must be >= 0"),
        ));
    }
    Ok(raw)
}

fn matrix_dimensions_for(name: &str, shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    match shape.len() {
        0 => Ok((1, 1)),
        1 => Ok((shape[0], 1)),
        _ => {
            if shape.len() > 2 && shape.iter().skip(2).any(|&dim| dim != 1) {
                Err(runtime_error(
                    name,
                    format!("{name}: inputs must be 2-D matrices or vectors"),
                ))
            } else {
                Ok((shape[0], shape[1]))
            }
        }
    }
}

fn value_into_tensor_for(name: &str, value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(t) => Ok(t),
        Value::LogicalArray(logical) => logical_to_tensor(name, &logical),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1])
            .map_err(|e| runtime_error(name, format!("{name}: {e}"))),
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1])
            .map_err(|e| runtime_error(name, format!("{name}: {e}"))),
        Value::Bool(b) => Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|e| runtime_error(name, format!("{name}: {e}"))),
        other => Err(runtime_error(
            name,
            format!(
                "{name}: unsupported input type {:?}; expected numeric or logical values",
                other
            ),
        )),
    }
}

fn logical_to_tensor(name: &str, logical: &LogicalArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = logical
        .data
        .iter()
        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
        .collect();
    Tensor::new(data, logical.shape.clone())
        .map_err(|e| runtime_error(name, format!("{name}: {e}")))
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

pub fn ensure_matrix_shape(shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    matrix_dimensions_for(BUILTIN_NAME, shape)
}

pub fn ishermitian_host_real_tensor(tensor: &Tensor, skew: bool, tol: f64) -> BuiltinResult<bool> {
    let (rows, cols) = matrix_dimensions_for(BUILTIN_NAME, &tensor.shape)?;
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
) -> BuiltinResult<bool> {
    let (rows, cols) = matrix_dimensions_for(BUILTIN_NAME, &tensor.shape)?;
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
) -> BuiltinResult<bool> {
    let (rows, cols) = matrix_dimensions_for(BUILTIN_NAME, shape)?;
    if rows != cols {
        return Ok(false);
    }
    let tensor = Tensor::new(data.to_vec(), shape.to_vec())
        .map_err(|e| runtime_error(BUILTIN_NAME, format!("{BUILTIN_NAME}: {e}")))?;
    ishermitian_host_real_tensor(&tensor, skew, tol)
}

pub fn ishermitian_host_complex_data(
    shape: &[usize],
    data: &[(f64, f64)],
    skew: bool,
    tol: f64,
) -> BuiltinResult<bool> {
    let (rows, cols) = matrix_dimensions_for(BUILTIN_NAME, shape)?;
    if rows != cols {
        return Ok(false);
    }
    let tensor = ComplexTensor::new(data.to_vec(), shape.to_vec())
        .map_err(|e| runtime_error(BUILTIN_NAME, format!("{BUILTIN_NAME}: {e}")))?;
    ishermitian_host_complex_tensor(&tensor, skew, tol)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
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
        let message = err.to_string();
        assert!(message.contains("unknown flag"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_negative_tolerance() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = ishermitian_builtin(Value::Tensor(tensor), vec![Value::Num(-1.0)])
            .expect_err("ishermitian should error on negative tolerance");
        let message = err.to_string();
        assert!(message.contains("tolerance must be >= 0"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_scalar_tolerance() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let tolerance = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = ishermitian_builtin(Value::Tensor(tensor), vec![Value::Tensor(tolerance)])
            .expect_err("ishermitian should error on non-scalar tolerance");
        let message = err.to_string();
        assert!(message.contains("tolerance must be a real scalar"));
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
        let message = err.to_string();
        assert!(message.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_unsupported_input_type() {
        let err = ishermitian_builtin(Value::String("abc".into()), Vec::new())
            .expect_err("ishermitian should reject strings");
        let message = err.to_string();
        assert!(message.contains("unsupported input type"));
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

    fn ishermitian_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::ishermitian_builtin(value, rest))
    }
}
