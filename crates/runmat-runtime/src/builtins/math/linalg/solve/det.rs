//! MATLAB-compatible `det` builtin with GPU-aware semantics for RunMat.

use nalgebra::{DMatrix, LU};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::linalg::type_resolvers::numeric_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "det";

#[derive(Debug, Clone, Copy)]
enum Determinant {
    Real(f64),
    Complex(f64, f64),
}

impl Determinant {
    fn apply_sign(self, sign: f64) -> Self {
        match self {
            Self::Real(value) => Self::Real(value * sign),
            Self::Complex(re, im) => Self::Complex(re * sign, im * sign),
        }
    }

    fn into_value(self) -> Value {
        match self {
            Self::Real(value) => Value::Num(value),
            Self::Complex(re, im) => Value::Complex(re, im),
        }
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::det")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("det"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("lu")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Real inputs re-upload their determinant to preserve residency; complex inputs currently return host scalars when LU hooks are available.",
};

fn builtin_error(message: String) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

fn interaction_pending_error() -> RuntimeError {
    build_runtime_error("interaction pending...")
        .with_builtin(NAME)
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::det")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Determinant evaluation is a terminal scalar operation and does not participate in fusion plans.",
};

#[runtime_builtin(
    name = "det",
    category = "math/linalg/solve",
    summary = "Compute the determinant of a square matrix.",
    keywords = "det,determinant,linear algebra,matrix,gpu",
    accel = "det",
    type_resolver(numeric_scalar_type),
    builtin_path = "crate::builtins::math::linalg::solve::det"
)]
async fn det_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => det_gpu(handle).await,
        Value::ComplexTensor(tensor) => det_complex_value(tensor),
        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            det_real_value(tensor)
        }
    }
}

async fn det_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        match det_gpu_via_provider(provider, &handle).await {
            Ok(Some(value)) => return Ok(value),
            Ok(None) => {}
            Err(err) => {
                if err.message() == "interaction pending..." {
                    return Err(interaction_pending_error());
                }
                return Err(err);
            }
        }
    }

    let gathered_value = {
        let proxy = Value::GpuTensor(handle.clone());
        gpu_helpers::gather_value_async(&proxy).await?
    };

    let det_result = determinant_from_value(gathered_value)?;
    match det_result {
        Determinant::Real(det) => {
            if let Some(provider) = runmat_accelerate_api::provider() {
                match upload_scalar(provider, det) {
                    Ok(uploaded) => return Ok(Value::GpuTensor(uploaded)),
                    Err(err) => {
                        if err.message() == "interaction pending..." {
                            return Err(interaction_pending_error());
                        }
                    }
                }
            }
            Ok(Value::Num(det))
        }
        Determinant::Complex(re, im) => Ok(Value::Complex(re, im)),
    }
}

fn det_real_value(tensor: Tensor) -> BuiltinResult<Value> {
    Ok(Determinant::Real(det_real_tensor(&tensor)?).into_value())
}

fn det_complex_value(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let (re, im) = det_complex_tensor(&tensor)?;
    Ok(Determinant::Complex(re, im).into_value())
}

fn determinant_from_value(value: Value) -> BuiltinResult<Determinant> {
    match value {
        Value::Num(n) => Ok(Determinant::Real(n)),
        Value::Tensor(tensor) => det_real_tensor(&tensor).map(Determinant::Real),
        Value::ComplexTensor(tensor) => {
            det_complex_tensor(&tensor).map(|(re, im)| Determinant::Complex(re, im))
        }
        Value::Complex(re, im) => Ok(Determinant::Complex(re, im)),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(builtin_error)?;
            det_real_tensor(&tensor).map(Determinant::Real)
        }
        Value::Int(int_value) => Ok(Determinant::Real(int_value.to_f64())),
        Value::Bool(flag) => Ok(Determinant::Real(if flag { 1.0 } else { 0.0 })),
        other => Err(builtin_error(format!(
            "{NAME}: unsupported input type {:?}; expected numeric or logical values",
            other
        ))),
    }
}

fn det_real_tensor(matrix: &Tensor) -> BuiltinResult<f64> {
    let (rows, cols) = matrix_dimensions(matrix.shape.as_slice())?;
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }
    if rows == 0 && cols == 0 {
        return Ok(1.0);
    }
    if matrix.data.len() == 1 {
        return Ok(matrix.data[0]);
    }
    let lu = LU::new(DMatrix::from_column_slice(rows, cols, &matrix.data));
    Ok(lu.determinant())
}

fn det_complex_tensor(matrix: &ComplexTensor) -> BuiltinResult<(f64, f64)> {
    let (rows, cols) = matrix_dimensions(matrix.shape.as_slice())?;
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }
    if rows == 0 && cols == 0 {
        return Ok((1.0, 0.0));
    }
    if matrix.data.len() == 1 {
        return Ok(matrix.data[0]);
    }
    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let lu = LU::new(DMatrix::from_column_slice(rows, cols, &data));
    let det = lu.determinant();
    Ok((det.re, det.im))
}

fn matrix_dimensions(shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    match shape {
        [] => Ok((1, 1)),
        [rows] => {
            if *rows == 1 {
                Ok((1, 1))
            } else {
                Err(builtin_error(format!(
                    "{NAME}: input must be a square matrix."
                )))
            }
        }
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        ))),
    }
}

fn upload_scalar(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    value: f64,
) -> BuiltinResult<GpuTensorHandle> {
    let data = [value];
    let shape = [1usize, 1usize];
    let view = HostTensorView {
        data: &data,
        shape: &shape,
    };
    provider
        .upload(&view)
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

async fn det_gpu_via_provider(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<Option<Value>> {
    let (rows, cols) = matrix_dimensions(handle.shape.as_slice())?;
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }
    if rows == 0 {
        let uploaded = upload_scalar(provider, 1.0)?;
        return Ok(Some(Value::GpuTensor(uploaded)));
    }

    let lu_result = match provider.lu(handle).await {
        Ok(result) => result,
        Err(_) => return Ok(None),
    };

    let handles_to_free = [
        lu_result.combined.clone(),
        lu_result.lower.clone(),
        lu_result.upper.clone(),
        lu_result.perm_matrix.clone(),
        lu_result.perm_vector.clone(),
    ];

    let outcome = {
        async {
            enum UpperFactor {
                Real(Tensor),
                Complex(ComplexTensor),
            }

            let upper_factor = match gpu_helpers::gather_tensor_async(&lu_result.upper).await {
                Ok(tensor) => UpperFactor::Real(tensor),
                Err(err) => {
                    if err.message() == "interaction pending..." {
                        return Err(interaction_pending_error());
                    }
                    let value = Value::GpuTensor(lu_result.upper.clone());
                    match gpu_helpers::gather_value_async(&value).await {
                        Ok(Value::Tensor(tensor)) => UpperFactor::Real(tensor),
                        Ok(Value::ComplexTensor(tensor)) => UpperFactor::Complex(tensor),
                        Ok(Value::Num(n)) => {
                            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(builtin_error)?;
                            UpperFactor::Real(tensor)
                        }
                        Ok(_) => return Ok(None),
                        Err(err) => {
                            if err.message() == "interaction pending..." {
                                return Err(interaction_pending_error());
                            }
                            return Ok(None);
                        }
                    }
                }
            };

            let pivot_tensor = match gpu_helpers::gather_tensor_async(&lu_result.perm_vector).await
            {
                Ok(tensor) => tensor,
                Err(err) => {
                    if err.message() == "interaction pending..." {
                        return Err(interaction_pending_error());
                    }
                    return Ok(None);
                }
            };

            let determinant = match upper_factor {
                UpperFactor::Real(tensor) => match diagonal_product_real(&tensor, rows) {
                    Ok(value) => Determinant::Real(value),
                    Err(err) => {
                        if err.message() == "interaction pending..." {
                            return Err(interaction_pending_error());
                        }
                        return Ok(None);
                    }
                },
                UpperFactor::Complex(tensor) => match diagonal_product_complex(&tensor, rows) {
                    Ok((re, im)) => Determinant::Complex(re, im),
                    Err(err) => {
                        if err.message() == "interaction pending..." {
                            return Err(interaction_pending_error());
                        }
                        return Ok(None);
                    }
                },
            };

            let permutation_sign = match permutation_sign_from_tensor(&pivot_tensor, rows) {
                Ok(value) => value,
                Err(err) => {
                    if err.message() == "interaction pending..." {
                        return Err(interaction_pending_error());
                    }
                    return Ok(None);
                }
            };

            let determinant = determinant.apply_sign(permutation_sign);

            match determinant {
                Determinant::Real(value) => match upload_scalar(provider, value) {
                    Ok(handle) => Ok(Some(Value::GpuTensor(handle))),
                    Err(err) => {
                        if err.message() == "interaction pending..." {
                            Err(interaction_pending_error())
                        } else {
                            Ok(None)
                        }
                    }
                },
                Determinant::Complex(re, im) => Ok(Some(Value::Complex(re, im))),
            }
        }
        .await
    };

    for handle_to_free in &handles_to_free {
        let _ = provider.free(handle_to_free);
    }

    outcome
}

fn diagonal_product_real(upper: &Tensor, dimension: usize) -> BuiltinResult<f64> {
    if dimension == 0 {
        return Ok(1.0);
    }
    let rows = upper.rows();
    let cols = upper.cols();
    if rows < dimension || cols < dimension {
        return Err(builtin_error(format!(
            "{NAME}: upper factor shape mismatch"
        )));
    }
    let mut product = 1.0f64;
    for i in 0..dimension {
        let idx = i + i * rows;
        let value = *upper
            .data
            .get(idx)
            .ok_or_else(|| builtin_error(format!("{NAME}: upper factor diagonal out of range")))?;
        product *= value;
    }
    Ok(product)
}

fn diagonal_product_complex(upper: &ComplexTensor, dimension: usize) -> BuiltinResult<(f64, f64)> {
    if dimension == 0 {
        return Ok((1.0, 0.0));
    }
    let rows = upper.rows;
    let cols = upper.cols;
    if rows < dimension || cols < dimension {
        return Err(builtin_error(format!(
            "{NAME}: upper factor shape mismatch"
        )));
    }
    let mut product = Complex64::new(1.0, 0.0);
    for i in 0..dimension {
        let idx = i + i * rows;
        let (re, im) = *upper
            .data
            .get(idx)
            .ok_or_else(|| builtin_error(format!("{NAME}: upper factor diagonal out of range")))?;
        product *= Complex64::new(re, im);
    }
    Ok((product.re, product.im))
}

fn permutation_sign_from_tensor(pivots: &Tensor, expected_len: usize) -> BuiltinResult<f64> {
    if expected_len == 0 {
        return Ok(1.0);
    }
    if pivots.data.len() != expected_len {
        return Err(builtin_error(format!(
            "{NAME}: pivot vector length mismatch"
        )));
    }
    let len = pivots.data.len();
    let mut permutation = Vec::with_capacity(len);
    let mut seen = vec![false; len];
    for &raw in &pivots.data {
        if !raw.is_finite() {
            return Err(builtin_error(format!(
                "{NAME}: pivot vector contains non-finite entries"
            )));
        }
        let rounded = raw.round();
        if (rounded - raw).abs() > 1.0e-6 {
            return Err(builtin_error(format!(
                "{NAME}: pivot vector must contain integer values"
            )));
        }
        if rounded < 1.0 {
            return Err(builtin_error(format!(
                "{NAME}: pivot vector index out of range"
            )));
        }
        let idx = (rounded as isize - 1) as usize;
        if idx >= len {
            return Err(builtin_error(format!(
                "{NAME}: pivot vector index out of range"
            )));
        }
        if seen[idx] {
            return Err(builtin_error(format!(
                "{NAME}: pivot vector must describe a permutation"
            )));
        }
        seen[idx] = true;
        permutation.push(idx);
    }
    Ok(permutation_sign(&permutation))
}

fn permutation_sign(permutation: &[usize]) -> f64 {
    let mut visited = vec![false; permutation.len()];
    let mut sign = 1.0f64;
    for start in 0..permutation.len() {
        if visited[start] {
            continue;
        }
        let mut length = 0usize;
        let mut current = start;
        while current < permutation.len() && !visited[current] {
            visited[current] = true;
            current = permutation[current];
            length += 1;
        }
        if length > 0 && length.is_multiple_of(2) {
            sign = -sign;
        }
    }
    sign
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use futures::executor::block_on;
    use runmat_builtins::Type;
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[cfg(feature = "wgpu")]
    fn det_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::det_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn det_basic_2x2() {
        let tensor = Tensor::new(vec![4.0, 1.0, -2.0, 3.0], vec![2, 2]).unwrap();
        let result = det_real_value(tensor).expect("det");
        match result {
            Value::Num(v) => assert!((v - 14.0).abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn det_type_returns_scalar() {
        let out = numeric_scalar_type(&[Type::Tensor {
            shape: Some(vec![Some(3), Some(3)]),
        }]);
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn det_non_square_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let err = unwrap_error(det_real_value(tensor).unwrap_err());
        assert!(err
            .message()
            .contains("det: input must be a square matrix."));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn det_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(
            vec![3.0, 1.0, 0.0, 0.0, 5.0, 2.0, 4.0, 0.0, 6.0],
            vec![3, 3],
        )
        .unwrap();
        let cpu_det = det_real_tensor(&tensor).expect("cpu det");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = det_builtin(Value::GpuTensor(handle)).expect("det");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 1]);
        let det_gpu = gathered.data[0];
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-5,
        };
        assert!(
            (det_gpu - cpu_det).abs() < tol,
            "gpu det {det_gpu} differs from cpu det {cpu_det}"
        );
    }
}
