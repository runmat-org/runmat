//! MATLAB-compatible `transpose` builtin with GPU-aware semantics for RunMat.
//!
//! This module mirrors MATLAB's `transpose` function (non-conjugating) across numeric,
//! logical, string, char, and cell arrays while integrating with RunMat Accelerate to
//! preserve GPU residency whenever possible.

use crate::builtins::array::shape::permute::{
    permute_complex_tensor, permute_logical_array, permute_string_array, permute_tensor,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use log::warn;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

const NAME: &str = "transpose";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::transpose")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Transpose,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("transpose")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Uses the provider transpose hook when available; otherwise gathers, transposes on the host, and uploads the result back to the GPU.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::ops::transpose"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Transposes act as fusion boundaries; downstream kernels see the updated shape metadata.",
};

#[runtime_builtin(
    name = "transpose",
    category = "math/linalg/ops",
    summary = "Swap the first two dimensions of arrays without conjugating complex values.",
    keywords = "transpose,swap rows and columns,non-conjugate",
    accel = "transpose",
    builtin_path = "crate::builtins::math::linalg::ops::transpose"
)]
fn transpose_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => transpose_gpu(handle),
        Value::Tensor(t) => transpose_tensor(t).map(tensor::tensor_into_value),
        Value::ComplexTensor(ct) => transpose_complex_tensor(ct).map(Value::ComplexTensor),
        Value::LogicalArray(la) => transpose_logical_array(la).map(Value::LogicalArray),
        Value::CharArray(ca) => transpose_char_array(ca).map(Value::CharArray),
        Value::StringArray(sa) => transpose_string_array(sa).map(Value::StringArray),
        Value::Cell(ca) => transpose_cell_array(ca).map(Value::Cell),
        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
        Value::Num(n) => Ok(Value::Num(n)),
        Value::Int(i) => Ok(Value::Int(i)),
        Value::Bool(b) => Ok(Value::Bool(b)),
        Value::String(s) => Ok(Value::String(s)),
        other => Err(format!("transpose: unsupported input type {other:?}")),
    }
}

fn transpose_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let rank = tensor.shape.len();
    if rank <= 2 {
        transpose_tensor_matrix(&tensor)
    } else {
        let order = transpose_order(rank);
        permute_tensor(tensor, &order)
    }
}

fn transpose_complex_tensor(ct: ComplexTensor) -> Result<ComplexTensor, String> {
    let rank = ct.shape.len();
    if rank == 0 {
        return Ok(ct);
    }
    if rank <= 2 {
        ComplexTensor::new(transpose_complex_matrix(&ct), vec![ct.cols, ct.rows])
            .map_err(|e| format!("{NAME}: {e}"))
    } else {
        let order = transpose_order(rank);
        permute_complex_tensor(ct, &order)
    }
}

fn transpose_logical_array(la: LogicalArray) -> Result<LogicalArray, String> {
    let rank = la.shape.len();
    if rank == 0 {
        return Ok(la);
    }
    if rank <= 2 {
        let rows = la.shape.first().copied().unwrap_or(1);
        let cols = if rank >= 2 {
            la.shape.get(1).copied().unwrap_or(1)
        } else {
            1
        };
        let mut out = vec![0u8; la.data.len()];
        for i in 0..rows {
            for j in 0..cols {
                let src = i + j * rows;
                let dst = j + i * cols;
                if src < la.data.len() && dst < out.len() {
                    out[dst] = la.data[src];
                }
            }
        }
        let new_shape = vec![cols, rows];
        LogicalArray::new(out, new_shape).map_err(|e| format!("{NAME}: {e}"))
    } else {
        let order = transpose_order(rank);
        permute_logical_array(la, &order)
    }
}

fn transpose_char_array(ca: CharArray) -> Result<CharArray, String> {
    let rows = ca.rows;
    let cols = ca.cols;
    if ca.data.is_empty() {
        return CharArray::new(Vec::new(), cols, rows).map_err(|e| format!("{NAME}: {e}"));
    }
    let mut out = vec!['\0'; ca.data.len()];
    for r in 0..rows {
        for c in 0..cols {
            let src = r * cols + c;
            let dst = c * rows + r;
            if src < ca.data.len() && dst < out.len() {
                out[dst] = ca.data[src];
            }
        }
    }
    CharArray::new(out, cols, rows).map_err(|e| format!("{NAME}: {e}"))
}

fn transpose_string_array(sa: StringArray) -> Result<StringArray, String> {
    let rank = sa.shape.len();
    if rank == 0 {
        return Ok(sa);
    }
    if rank <= 2 {
        let rows = sa.rows;
        let cols = sa.cols;
        let mut out = vec![String::new(); sa.data.len()];
        for r in 0..rows {
            for c in 0..cols {
                let src = r + c * rows;
                let dst = c + r * cols;
                if src < sa.data.len() && dst < out.len() {
                    out[dst] = sa.data[src].clone();
                }
            }
        }
        let new_shape = if rank >= 2 {
            let mut shape = sa.shape.clone();
            if shape.len() >= 2 {
                shape.swap(0, 1);
                shape
            } else {
                vec![cols, rows]
            }
        } else {
            vec![cols, rows]
        };
        StringArray::new(out, new_shape).map_err(|e| format!("{NAME}: {e}"))
    } else {
        let order = transpose_order(rank);
        permute_string_array(sa, &order)
    }
}

fn transpose_cell_array(ca: CellArray) -> Result<CellArray, String> {
    let rows = ca.rows;
    let cols = ca.cols;
    let mut out = Vec::with_capacity(ca.data.len());
    for c in 0..cols {
        for r in 0..rows {
            let idx = r * cols + c;
            out.push(ca.data[idx].clone());
        }
    }
    CellArray::new_handles(out, cols, rows).map_err(|e| format!("{NAME}: {e}"))
}

fn transpose_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    let rank = handle.shape.len();
    if rank == 0 {
        return Ok(Value::GpuTensor(handle));
    }
    if rank <= 2 {
        if let Some(provider) = runmat_accelerate_api::provider() {
            match provider.transpose(&handle) {
                Ok(out) => return Ok(Value::GpuTensor(out)),
                Err(err) => {
                    let info = provider.device_info_struct();
                    warn!(
                        "transpose: provider {} (backend: {}) is missing transpose support; falling back ({err})",
                        info.name,
                        info.backend.as_deref().unwrap_or("unknown")
                    );
                }
            }
        }
    }
    let host = gpu_helpers::gather_tensor(&handle)?;
    let transposed = transpose_tensor(host)?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &transposed.data,
            shape: &transposed.shape,
        };
        match provider.upload(&view) {
            Ok(uploaded) => return Ok(Value::GpuTensor(uploaded)),
            Err(upload_err) => warn!(
                "transpose: re-upload after host fallback failed; returning host tensor ({upload_err})"
            ),
        }
    }
    Ok(tensor::tensor_into_value(transposed))
}

fn transpose_order(rank: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (1..=rank.max(2)).collect();
    if order.len() >= 2 {
        order.swap(0, 1);
    }
    if order.len() > rank && rank < 2 {
        order.truncate(rank.max(2));
    }
    order
}

fn transpose_tensor_matrix(tensor: &Tensor) -> Result<Tensor, String> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    if tensor.data.is_empty() {
        return Tensor::new(Vec::new(), vec![cols, rows]).map_err(|e| format!("{NAME}: {e}"));
    }
    let mut out = vec![0.0; tensor.data.len()];
    for r in 0..rows {
        for c in 0..cols {
            let src = r + c * rows;
            let dst = c + r * cols;
            if src < tensor.data.len() && dst < out.len() {
                out[dst] = tensor.data[src];
            }
        }
    }
    Tensor::new(out, vec![cols, rows]).map_err(|e| format!("{NAME}: {e}"))
}

fn transpose_complex_matrix(ct: &ComplexTensor) -> Vec<(f64, f64)> {
    let rows = ct.rows;
    let cols = ct.cols;
    if ct.data.is_empty() {
        return Vec::new();
    }
    let mut out = vec![(0.0, 0.0); ct.data.len()];
    for r in 0..rows {
        for c in 0..cols {
            let src = r + c * rows;
            let dst = c + r * cols;
            if src < ct.data.len() && dst < out.len() {
                out[dst] = ct.data[src];
            }
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_backend;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, LogicalArray, Tensor};

    fn tensor(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_numeric_matrix() {
        let input = tensor(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]);
        let value = transpose_builtin(Value::Tensor(input)).expect("transpose");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 2]);
                assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_vector_to_column() {
        let input = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
        let value = transpose_builtin(Value::Tensor(input)).expect("transpose");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(out.data, vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_complex_does_not_conjugate() {
        let data = vec![(1.0, 2.0), (3.0, -4.0)];
        let ct = ComplexTensor::new(data, vec![1, 2]).unwrap();
        let value = transpose_builtin(Value::ComplexTensor(ct)).expect("transpose");
        match value {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![(1.0, 2.0), (3.0, -4.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_high_dim_tensor() {
        let data: Vec<f64> = (1..=24).map(|n| n as f64).collect();
        let tensor = tensor(&data, &[2, 3, 4]);
        let value = transpose_builtin(Value::Tensor(tensor)).expect("transpose");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 2, 4]);
                assert_eq!(out.data.len(), 24);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_logical_mask() {
        let la = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let value = transpose_builtin(Value::LogicalArray(la)).expect("transpose");
        match value {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![1, 0, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_char_matrix() {
        let ca = CharArray::new("runmat".chars().collect(), 2, 3).unwrap();
        let value = transpose_builtin(Value::CharArray(ca)).expect("transpose");
        match value {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 3);
                assert_eq!(out.cols, 2);
                assert_eq!(out.data, vec!['r', 'm', 'u', 'a', 'n', 't']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_string_array() {
        let sa = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![1, 3]).unwrap();
        let value = transpose_builtin(Value::StringArray(sa)).expect("transpose");
        match value {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(
                    out.data,
                    vec!["a".to_string(), "b".to_string(), "c".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_cell_array() {
        let cells = vec![
            Value::from(1),
            Value::from(2),
            Value::from(3),
            Value::from(4),
        ];
        let cell_array = CellArray::new(cells, 2, 2).unwrap();
        let value = transpose_builtin(Value::Cell(cell_array)).expect("transpose");
        match value {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 2);
                let v00 = out.get(0, 0).unwrap();
                let v01 = out.get(0, 1).unwrap();
                let v10 = out.get(1, 0).unwrap();
                let v11 = out.get(1, 1).unwrap();
                assert_eq!(v00, Value::from(1));
                assert_eq!(v01, Value::from(3));
                assert_eq!(v10, Value::from(2));
                assert_eq!(v11, Value::from(4));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_scalar_types_identity() {
        assert_eq!(
            transpose_builtin(Value::Num(std::f64::consts::PI)).unwrap(),
            Value::Num(std::f64::consts::PI)
        );
        assert_eq!(
            transpose_builtin(Value::Complex(1.0, -2.0)).unwrap(),
            Value::Complex(1.0, -2.0)
        );
        assert_eq!(
            transpose_builtin(Value::Int(IntValue::I32(5))).unwrap(),
            Value::Int(IntValue::I32(5))
        );
        assert_eq!(
            transpose_builtin(Value::Bool(true)).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let t = tensor(&[1.0, 4.0, 2.0, 5.0], &[2, 2]);
            let view = HostTensorView {
                data: &t.data,
                shape: &t.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = transpose_builtin(Value::GpuTensor(handle)).expect("transpose");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 2.0, 4.0, 5.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn transpose_wgpu_matches_cpu() {
        let _ = wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let data: Vec<f64> = (1..=12).map(|n| n as f64).collect();
        let tensor = Tensor::new(data, vec![3, 4]).expect("tensor");
        let cpu_value = transpose_builtin(Value::Tensor(tensor.clone())).expect("cpu transpose");
        let cpu_tensor = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = transpose_builtin(Value::GpuTensor(handle)).expect("gpu transpose");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu_tensor.shape);
        assert_eq!(gathered.data, cpu_tensor.data);
    }
}
