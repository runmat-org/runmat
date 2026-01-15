//! MATLAB-compatible `cond` builtin with GPU-aware fallbacks.

use nalgebra::{linalg::SVD, DMatrix};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, ProviderCondNorm};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::linalg::matrix_dimensions_for;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

const NAME: &str = "cond";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::cond")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("cond"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("cond")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may expose a direct condition-number kernel; the reference backends gather to the host, evaluate the shared implementation, and upload the scalar result.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::cond")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fusible; cond consumes an entire matrix and returns a scalar diagnostic.",
};

#[runtime_builtin(
    name = "cond",
    category = "math/linalg/solve",
    summary = "Compute the matrix condition number with MATLAB-compatible norms.",
    keywords = "cond,condition number,norm,gpu",
    accel = "cond",
    builtin_path = "crate::builtins::math::linalg::solve::cond"
)]
fn cond_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let norm = parse_norm_argument(&rest)?;
    let result = match value {
        Value::GpuTensor(handle) => return cond_gpu(handle, norm),
        Value::ComplexTensor(matrix) => cond_complex_tensor(&matrix, norm)?,
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| format!("{NAME}: {e}"))?;
            cond_complex_tensor(&tensor, norm)?
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other)?;
            cond_real_tensor(&tensor, norm)?
        }
    };
    Ok(Value::Num(result))
}

fn cond_gpu(handle: GpuTensorHandle, norm: CondNorm) -> Result<Value, String> {
    let maybe_provider = runmat_accelerate_api::provider();

    if let Some(provider) = maybe_provider {
        if let Ok(Some(value)) = cond_gpu_via_provider(provider, &handle, norm) {
            return Ok(value);
        }
    }

    let gathered = gpu_helpers::gather_value(&Value::GpuTensor(handle.clone()))
        .map_err(|e| format!("{NAME}: {e}"))?;

    let cond_value = match gathered {
        Value::Tensor(tensor) => cond_real_tensor(&tensor, norm)?,
        Value::ComplexTensor(tensor) => cond_complex_tensor(&tensor, norm)?,
        Value::Num(n) => {
            if n == 0.0 {
                f64::INFINITY
            } else {
                1.0
            }
        }
        Value::Complex(re, im) => {
            if re == 0.0 && im == 0.0 {
                f64::INFINITY
            } else {
                1.0
            }
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other)?;
            cond_real_tensor(&tensor, norm)?
        }
    };

    if let Some(provider) = maybe_provider {
        if let Ok(uploaded) = upload_scalar(provider, cond_value) {
            return Ok(Value::GpuTensor(uploaded));
        }
    }

    Ok(Value::Num(cond_value))
}

fn cond_gpu_via_provider(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
    norm: CondNorm,
) -> Result<Option<Value>, String> {
    let provider_norm = ProviderCondNorm::from(norm);
    match provider.cond(handle, provider_norm) {
        Ok(result) => Ok(Some(Value::GpuTensor(result))),
        Err(_err) => Ok(None),
    }
}

fn cond_real_tensor(matrix: &Tensor, norm: CondNorm) -> Result<f64, String> {
    let (rows, cols) = matrix_dimensions_for(NAME, &matrix.shape)?;
    if rows == 0 || cols == 0 {
        return Ok(0.0);
    }
    if matrix.data.len() == 1 {
        return Ok(if matrix.data[0] == 0.0 {
            f64::INFINITY
        } else {
            1.0
        });
    }

    match norm {
        CondNorm::Two => cond_two_norm_real(matrix, rows, cols),
        _ => {
            if rows != cols {
                return Err(format!(
                    "{NAME}: matrix must be square for the requested norm."
                ));
            }
            cond_inverse_based_real(matrix, rows, norm)
        }
    }
}

fn cond_complex_tensor(matrix: &ComplexTensor, norm: CondNorm) -> Result<f64, String> {
    let (rows, cols) = matrix_dimensions_for(NAME, &matrix.shape)?;
    if rows == 0 || cols == 0 {
        return Ok(0.0);
    }
    if matrix.data.len() == 1 {
        let (re, im) = matrix.data[0];
        let magnitude = re.hypot(im);
        return Ok(if magnitude == 0.0 { f64::INFINITY } else { 1.0 });
    }

    match norm {
        CondNorm::Two => cond_two_norm_complex(matrix, rows, cols),
        _ => {
            if rows != cols {
                return Err(format!(
                    "{NAME}: matrix must be square for the requested norm."
                ));
            }
            cond_inverse_based_complex(matrix, rows, norm)
        }
    }
}

fn cond_two_norm_real(matrix: &Tensor, rows: usize, cols: usize) -> Result<f64, String> {
    let a = DMatrix::from_column_slice(rows, cols, &matrix.data);
    let svd = SVD::new(a, false, false);
    Ok(singular_value_cond(svd.singular_values.as_slice()))
}

fn cond_two_norm_complex(matrix: &ComplexTensor, rows: usize, cols: usize) -> Result<f64, String> {
    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let a = DMatrix::from_column_slice(rows, cols, &data);
    let svd = SVD::new(a, false, false);
    Ok(singular_value_cond(svd.singular_values.as_slice()))
}

fn cond_inverse_based_real(matrix: &Tensor, order: usize, norm: CondNorm) -> Result<f64, String> {
    let dm = DMatrix::from_column_slice(order, order, &matrix.data);
    if let Some(inv) = dm.try_inverse() {
        let norm_a = matrix_norm_real(matrix.data.as_slice(), order, order, norm);
        let norm_inv = matrix_norm_real(inv.as_slice(), order, order, norm);
        let cond = norm_a * norm_inv;
        if cond.is_finite() {
            Ok(cond)
        } else {
            Ok(f64::INFINITY)
        }
    } else {
        Ok(f64::INFINITY)
    }
}

fn cond_inverse_based_complex(
    matrix: &ComplexTensor,
    order: usize,
    norm: CondNorm,
) -> Result<f64, String> {
    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let dm = DMatrix::from_column_slice(order, order, &data);
    if let Some(inv) = dm.try_inverse() {
        let norm_a = matrix_norm_complex(&data, order, order, norm);
        let norm_inv = matrix_norm_complex(inv.as_slice(), order, order, norm);
        let cond = norm_a * norm_inv;
        if cond.is_finite() {
            Ok(cond)
        } else {
            Ok(f64::INFINITY)
        }
    } else {
        Ok(f64::INFINITY)
    }
}

fn matrix_norm_real(data: &[f64], rows: usize, cols: usize, norm: CondNorm) -> f64 {
    match norm {
        CondNorm::One => {
            let mut max_sum: f64 = 0.0;
            for c in 0..cols {
                let mut sum = 0.0;
                for r in 0..rows {
                    sum += data[r + c * rows].abs();
                }
                max_sum = max_sum.max(sum);
            }
            max_sum
        }
        CondNorm::Inf => {
            let mut max_sum: f64 = 0.0;
            for r in 0..rows {
                let mut sum = 0.0;
                for c in 0..cols {
                    sum += data[r + c * rows].abs();
                }
                max_sum = max_sum.max(sum);
            }
            max_sum
        }
        CondNorm::Fro => {
            let sum_sq: f64 = data.iter().map(|v| v * v).sum();
            sum_sq.sqrt()
        }
        CondNorm::Two => unreachable!("matrix_norm_real not used for 2-norm"),
    }
}

fn matrix_norm_complex(data: &[Complex64], rows: usize, cols: usize, norm: CondNorm) -> f64 {
    match norm {
        CondNorm::One => {
            let mut max_sum: f64 = 0.0;
            for c in 0..cols {
                let mut sum = 0.0;
                for r in 0..rows {
                    sum += data[r + c * rows].norm();
                }
                max_sum = max_sum.max(sum);
            }
            max_sum
        }
        CondNorm::Inf => {
            let mut max_sum: f64 = 0.0;
            for r in 0..rows {
                let mut sum = 0.0;
                for c in 0..cols {
                    sum += data[r + c * rows].norm();
                }
                max_sum = max_sum.max(sum);
            }
            max_sum
        }
        CondNorm::Fro => {
            let sum_sq: f64 = data.iter().map(|v| v.norm_sqr()).sum();
            sum_sq.sqrt()
        }
        CondNorm::Two => unreachable!("matrix_norm_complex not used for 2-norm"),
    }
}

fn singular_value_cond(singular_values: &[f64]) -> f64 {
    if singular_values.is_empty() {
        return 0.0;
    }
    let mut min_sv = f64::INFINITY;
    let mut max_sv = 0.0_f64;
    for &sv in singular_values {
        let abs = sv.abs();
        if !abs.is_finite() {
            return f64::INFINITY;
        }
        min_sv = min_sv.min(abs);
        max_sv = max_sv.max(abs);
    }
    if min_sv == 0.0 {
        f64::INFINITY
    } else {
        max_sv / min_sv
    }
}

fn parse_norm_argument(args: &[Value]) -> Result<CondNorm, String> {
    match args.len() {
        0 => Ok(CondNorm::Two),
        1 => parse_norm_value(&args[0]),
        _ => Err(format!("{NAME}: too many input arguments")),
    }
}

fn parse_norm_value(value: &Value) -> Result<CondNorm, String> {
    if let Some(text) = tensor::value_to_string(value) {
        return parse_norm_string(&text);
    }
    match value {
        Value::Num(n) => parse_norm_numeric(*n),
        Value::Int(i) => parse_norm_numeric(i.to_f64()),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => parse_norm_numeric(t.data[0]),
        Value::Bool(b) => {
            if *b {
                Ok(CondNorm::One)
            } else {
                Err(format!("{NAME}: norm must be 1, 2, Inf, or 'fro'"))
            }
        }
        Value::LogicalArray(logical) if logical.len() == 1 => {
            if logical.data[0] != 0 {
                Ok(CondNorm::One)
            } else {
                Err(format!("{NAME}: norm must be 1, 2, Inf, or 'fro'"))
            }
        }
        _ => Err(format!("{NAME}: norm must be 1, 2, Inf, or 'fro'")),
    }
}

fn parse_norm_numeric(raw: f64) -> Result<CondNorm, String> {
    if raw == 1.0 {
        Ok(CondNorm::One)
    } else if raw == 2.0 {
        Ok(CondNorm::Two)
    } else if raw.is_infinite() && raw.is_sign_positive() {
        Ok(CondNorm::Inf)
    } else {
        Err(format!("{NAME}: norm must be 1, 2, Inf, or 'fro'"))
    }
}

fn parse_norm_string(text: &str) -> Result<CondNorm, String> {
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "2" | "two" => Ok(CondNorm::Two),
        "1" | "one" => Ok(CondNorm::One),
        "inf" | "infinity" => Ok(CondNorm::Inf),
        "fro" | "frobenius" => Ok(CondNorm::Fro),
        _ => Err(format!("{NAME}: unrecognised norm '{text}'")),
    }
}

fn upload_scalar(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    value: f64,
) -> Result<GpuTensorHandle, String> {
    let data = [value];
    let shape = [1usize, 1usize];
    let view = HostTensorView {
        data: &data,
        shape: &shape,
    };
    provider.upload(&view).map_err(|e| format!("{NAME}: {e}"))
}

/// Helper for provider backends that reuse the host implementation.
pub fn cond_host_real_for_provider(matrix: &Tensor, norm: ProviderCondNorm) -> Result<f64, String> {
    cond_real_tensor(matrix, CondNorm::from(norm))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CondNorm {
    Two,
    One,
    Inf,
    Fro,
}

impl From<CondNorm> for ProviderCondNorm {
    fn from(value: CondNorm) -> Self {
        match value {
            CondNorm::Two => ProviderCondNorm::Two,
            CondNorm::One => ProviderCondNorm::One,
            CondNorm::Inf => ProviderCondNorm::Inf,
            CondNorm::Fro => ProviderCondNorm::Fro,
        }
    }
}

impl From<ProviderCondNorm> for CondNorm {
    fn from(value: ProviderCondNorm) -> Self {
        match value {
            ProviderCondNorm::Two => CondNorm::Two,
            ProviderCondNorm::One => CondNorm::One,
            ProviderCondNorm::Inf => CondNorm::Inf,
            ProviderCondNorm::Fro => CondNorm::Fro,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, Tensor, Value};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_identity_is_one() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let result = cond_builtin(Value::Tensor(tensor), Vec::new()).expect("cond");
        match result {
            Value::Num(value) => assert!((value - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_zero_is_infinite() {
        let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = cond_builtin(Value::Tensor(tensor), Vec::new()).expect("cond");
        match result {
            Value::Num(value) => assert!(value.is_infinite()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_rectangular_two_norm() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]).unwrap();
        let result = cond_builtin(Value::Tensor(tensor), Vec::new()).expect("cond");
        match result {
            Value::Num(value) => assert!((value - 2.414213562).abs() < 1e-9),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_one_norm_matches_manual() {
        let tensor = Tensor::new(vec![4.0, 2.0, -1.0, 3.0], vec![2, 2]).unwrap();
        let result =
            cond_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))]).expect("cond");
        match result {
            Value::Num(value) => assert!((value - 2.142_857_142_857_143).abs() < 1e-9),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_infinity_norm() {
        let tensor = Tensor::new(vec![4.0, 2.0, -1.0, 3.0], vec![2, 2]).unwrap();
        let result = cond_builtin(Value::Tensor(tensor), vec![Value::from("inf")]).expect("cond");
        match result {
            Value::Num(value) => assert!((value - 2.142_857_142_857_143).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_frobenius_norm() {
        let tensor = Tensor::new(vec![5.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let result = cond_builtin(Value::Tensor(tensor), vec![Value::from("fro")]).expect("cond");
        match result {
            Value::Num(value) => assert!((value - 2.9).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_complex_matrix_supported() {
        let data = vec![(1.0, 2.0), (0.0, 0.0), (0.0, 3.0), (2.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = cond_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("cond");
        match result {
            Value::Num(value) => assert!(value.is_finite() && value >= 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_rejects_non_square_for_other_norms() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let err = cond_builtin(Value::Tensor(tensor), vec![Value::from("inf")]).unwrap_err();
        assert_eq!(err, "cond: matrix must be square for the requested norm.");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_empty_returns_zero() {
        let tensor = Tensor::new(vec![], vec![0, 0]).unwrap();
        let result = cond_builtin(Value::Tensor(tensor), Vec::new()).expect("cond");
        match result {
            Value::Num(value) => assert_eq!(value, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = cond_builtin(Value::GpuTensor(handle), Vec::new()).expect("cond");
            let gathered = test_support::gather(gpu_value).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            let expected = cond_builtin(Value::Tensor(tensor.clone()), Vec::new())
                .map(|v| match v {
                    Value::Num(n) => n,
                    _ => unreachable!(),
                })
                .expect("cpu cond");
            assert!((gathered.data[0] - expected).abs() < 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cond_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![2.0, 0.0, 0.0, 0.2], vec![2, 2]).unwrap();
        let cpu = cond_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("cpu");
        let cpu_value = match cpu {
            Value::Num(n) => n,
            _ => unreachable!(),
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu = cond_builtin(Value::GpuTensor(handle), Vec::new()).expect("cond");
        let gathered = test_support::gather(gpu).expect("gather");
        assert!((gathered.data[0] - cpu_value).abs() < 1e-9);
    }
}
