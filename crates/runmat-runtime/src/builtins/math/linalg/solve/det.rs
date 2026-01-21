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
    builtin_path = "crate::builtins::math::linalg::solve::det"
)]
fn det_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => det_gpu(handle),
        Value::ComplexTensor(tensor) => det_complex_value(tensor),
        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other)?;
            det_real_value(tensor)
        }
    }
}

fn det_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        match det_gpu_via_provider(provider, &handle) {
            Ok(Some(value)) => return Ok(value),
            Ok(None) => {}
            Err(err) => return Err(err),
        }
    }

    let gathered_value = {
        let proxy = Value::GpuTensor(handle.clone());
        gpu_helpers::gather_value(&proxy).map_err(|e| format!("{NAME}: {e}"))?
    };

    let det_result = determinant_from_value(gathered_value)?;
    match det_result {
        Determinant::Real(det) => {
            if let Some(provider) = runmat_accelerate_api::provider() {
                if let Ok(uploaded) = upload_scalar(provider, det) {
                    return Ok(Value::GpuTensor(uploaded));
                }
            }
            Ok(Value::Num(det))
        }
        Determinant::Complex(re, im) => Ok(Value::Complex(re, im)),
    }
}

fn det_real_value(tensor: Tensor) -> Result<Value, String> {
    Ok(Determinant::Real(det_real_tensor(&tensor)?).into_value())
}

fn det_complex_value(tensor: ComplexTensor) -> Result<Value, String> {
    let (re, im) = det_complex_tensor(&tensor)?;
    Ok(Determinant::Complex(re, im).into_value())
}

fn determinant_from_value(value: Value) -> Result<Determinant, String> {
    match value {
        Value::Num(n) => Ok(Determinant::Real(n)),
        Value::Tensor(tensor) => det_real_tensor(&tensor).map(Determinant::Real),
        Value::ComplexTensor(tensor) => {
            det_complex_tensor(&tensor).map(|(re, im)| Determinant::Complex(re, im))
        }
        Value::Complex(re, im) => Ok(Determinant::Complex(re, im)),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            det_real_tensor(&tensor).map(Determinant::Real)
        }
        Value::Int(int_value) => Ok(Determinant::Real(int_value.to_f64())),
        Value::Bool(flag) => Ok(Determinant::Real(if flag { 1.0 } else { 0.0 })),
        other => Err(format!(
            "{NAME}: unsupported input type {:?}; expected numeric or logical values",
            other
        )),
    }
}

fn det_real_tensor(matrix: &Tensor) -> Result<f64, String> {
    let (rows, cols) = matrix_dimensions(matrix.shape.as_slice())?;
    if rows != cols {
        return Err(format!("{NAME}: input must be a square matrix."));
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

fn det_complex_tensor(matrix: &ComplexTensor) -> Result<(f64, f64), String> {
    let (rows, cols) = matrix_dimensions(matrix.shape.as_slice())?;
    if rows != cols {
        return Err(format!("{NAME}: input must be a square matrix."));
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

fn matrix_dimensions(shape: &[usize]) -> Result<(usize, usize), String> {
    match shape {
        [] => Ok((1, 1)),
        [rows] => {
            if *rows == 1 {
                Ok((1, 1))
            } else {
                Err(format!("{NAME}: input must be a square matrix."))
            }
        }
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(format!("{NAME}: input must be a square matrix.")),
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

fn det_gpu_via_provider(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
) -> Result<Option<Value>, String> {
    let (rows, cols) = matrix_dimensions(handle.shape.as_slice())?;
    if rows != cols {
        return Err(format!("{NAME}: input must be a square matrix."));
    }
    if rows == 0 {
        let uploaded = upload_scalar(provider, 1.0)?;
        return Ok(Some(Value::GpuTensor(uploaded)));
    }

    let lu_result = match provider.lu(handle) {
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

    let outcome = (|| -> Result<Option<Value>, String> {
        enum UpperFactor {
            Real(Tensor),
            Complex(ComplexTensor),
        }

        let upper_factor = match gpu_helpers::gather_tensor(&lu_result.upper) {
            Ok(tensor) => UpperFactor::Real(tensor),
            Err(_) => {
                let value = Value::GpuTensor(lu_result.upper.clone());
                match gpu_helpers::gather_value(&value) {
                    Ok(Value::Tensor(tensor)) => UpperFactor::Real(tensor),
                    Ok(Value::ComplexTensor(tensor)) => UpperFactor::Complex(tensor),
                    Ok(Value::Num(n)) => {
                        let tensor =
                            Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("{NAME}: {e}"))?;
                        UpperFactor::Real(tensor)
                    }
                    _ => return Ok(None),
                }
            }
        };

        let pivot_tensor = match gpu_helpers::gather_tensor(&lu_result.perm_vector) {
            Ok(tensor) => tensor,
            Err(_) => return Ok(None),
        };

        let determinant = match upper_factor {
            UpperFactor::Real(tensor) => match diagonal_product_real(&tensor, rows) {
                Ok(value) => Determinant::Real(value),
                Err(_) => return Ok(None),
            },
            UpperFactor::Complex(tensor) => match diagonal_product_complex(&tensor, rows) {
                Ok((re, im)) => Determinant::Complex(re, im),
                Err(_) => return Ok(None),
            },
        };

        let permutation_sign = match permutation_sign_from_tensor(&pivot_tensor, rows) {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };

        let determinant = determinant.apply_sign(permutation_sign);

        match determinant {
            Determinant::Real(value) => {
                let uploaded = match upload_scalar(provider, value) {
                    Ok(handle) => handle,
                    Err(_) => return Ok(None),
                };
                Ok(Some(Value::GpuTensor(uploaded)))
            }
            Determinant::Complex(re, im) => Ok(Some(Value::Complex(re, im))),
        }
    })();

    for handle_to_free in &handles_to_free {
        let _ = provider.free(handle_to_free);
    }

    outcome
}

fn diagonal_product_real(upper: &Tensor, dimension: usize) -> Result<f64, String> {
    if dimension == 0 {
        return Ok(1.0);
    }
    let rows = upper.rows();
    let cols = upper.cols();
    if rows < dimension || cols < dimension {
        return Err(format!("{NAME}: upper factor shape mismatch"));
    }
    let mut product = 1.0f64;
    for i in 0..dimension {
        let idx = i + i * rows;
        let value = *upper
            .data
            .get(idx)
            .ok_or_else(|| format!("{NAME}: upper factor diagonal out of range"))?;
        product *= value;
    }
    Ok(product)
}

fn diagonal_product_complex(upper: &ComplexTensor, dimension: usize) -> Result<(f64, f64), String> {
    if dimension == 0 {
        return Ok((1.0, 0.0));
    }
    let rows = upper.rows;
    let cols = upper.cols;
    if rows < dimension || cols < dimension {
        return Err(format!("{NAME}: upper factor shape mismatch"));
    }
    let mut product = Complex64::new(1.0, 0.0);
    for i in 0..dimension {
        let idx = i + i * rows;
        let (re, im) = *upper
            .data
            .get(idx)
            .ok_or_else(|| format!("{NAME}: upper factor diagonal out of range"))?;
        product *= Complex64::new(re, im);
    }
    Ok((product.re, product.im))
}

fn permutation_sign_from_tensor(pivots: &Tensor, expected_len: usize) -> Result<f64, String> {
    if expected_len == 0 {
        return Ok(1.0);
    }
    if pivots.data.len() != expected_len {
        return Err(format!("{NAME}: pivot vector length mismatch"));
    }
    let len = pivots.data.len();
    let mut permutation = Vec::with_capacity(len);
    let mut seen = vec![false; len];
    for &raw in &pivots.data {
        if !raw.is_finite() {
            return Err(format!("{NAME}: pivot vector contains non-finite entries"));
        }
        let rounded = raw.round();
        if (rounded - raw).abs() > 1.0e-6 {
            return Err(format!("{NAME}: pivot vector must contain integer values"));
        }
        if rounded < 1.0 {
            return Err(format!("{NAME}: pivot vector index out of range"));
        }
        let idx = (rounded as isize - 1) as usize;
        if idx >= len {
            return Err(format!("{NAME}: pivot vector index out of range"));
        }
        if seen[idx] {
            return Err(format!("{NAME}: pivot vector must describe a permutation"));
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
    use crate::builtins::common::test_support;
    use runmat_builtins::LogicalArray;

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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn det_non_square_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let err = det_real_value(tensor).unwrap_err();
        assert!(err.contains("det: input must be a square matrix."));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn det_complex_matrix() {
        let data = vec![(1.0, 2.0), (0.0, 0.0), (0.0, 3.0), (4.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let value = det_complex_value(tensor).expect("det");
        match value {
            Value::Complex(re, im) => {
                assert!((re - 6.0).abs() < 1e-12);
                assert!((im - 7.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn det_complex_scalar_input() {
        let value = det_builtin(Value::Complex(3.0, -2.0)).expect("det");
        match value {
            Value::Complex(re, im) => {
                assert_eq!(re, 3.0);
                assert_eq!(im, -2.0);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn det_logical_matrix_promotes() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let result = det_builtin(Value::LogicalArray(logical)).expect("det");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn det_singular_returns_zero() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = det_real_value(tensor).expect("det");
        match result {
            Value::Num(v) => assert!(v.abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn det_empty_matrix_returns_one() {
        let tensor = Tensor::new(vec![], vec![0, 0]).unwrap();
        let result = det_real_value(tensor).expect("det");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn det_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(
                vec![2.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
                vec![3, 3],
            )
            .unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = det_builtin(Value::GpuTensor(handle)).expect("det");
            let gathered = test_support::gather(gpu_value).expect("gather");
            assert!(
                (gathered.data[0] - 24.0).abs() < 1e-12,
                "got {}",
                gathered.data[0]
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn det_gpu_permutation_sign() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = det_builtin(Value::GpuTensor(handle)).expect("det");
            let gathered = test_support::gather(gpu_value).expect("gather");
            assert!(
                (gathered.data[0] + 1.0).abs() < 1e-12,
                "expected -1, got {}",
                gathered.data[0]
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn det_rejects_higher_rank_arrays() {
        let tensor = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let err = det_real_value(tensor).unwrap_err();
        assert!(err.contains("det: input must be a square matrix."));
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
