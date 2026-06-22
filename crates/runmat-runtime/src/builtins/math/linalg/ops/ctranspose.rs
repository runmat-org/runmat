//! MATLAB-compatible `ctranspose` builtin with GPU-aware semantics for RunMat.
//!
//! This module mirrors MATLAB's conjugate-transpose operator (`A'`) across numeric,
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
use crate::builtins::math::linalg::ops::transpose_real_sparse_tensor;
use crate::builtins::math::linalg::type_resolvers::transpose_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use log::warn;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, HostTensorView};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

const NAME: &str = "ctranspose";

const CTRANSPOSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value with first two dimensions swapped and complex conjugated.",
}];

const CTRANSPOSE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar/array value.",
}];

const CTRANSPOSE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = ctranspose(A)",
    inputs: &CTRANSPOSE_INPUTS,
    outputs: &CTRANSPOSE_OUTPUT,
}];

const CTRANSPOSE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CTRANSPOSE.INVALID_ARGUMENT",
    identifier: Some("RunMat:ctranspose:InvalidArgument"),
    when: "Call does not provide exactly one input argument.",
    message: "ctranspose: invalid argument",
};

const CTRANSPOSE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CTRANSPOSE.INVALID_INPUT",
    identifier: Some("RunMat:ctranspose:InvalidInput"),
    when: "Input type is unsupported for conjugate transpose.",
    message: "ctranspose: unsupported input type",
};

const CTRANSPOSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CTRANSPOSE.INTERNAL",
    identifier: Some("RunMat:ctranspose:Internal"),
    when: "Runtime cannot materialize conjugate-transpose output.",
    message: "ctranspose: internal runtime failure",
};

const CTRANSPOSE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CTRANSPOSE_ERROR_INVALID_ARGUMENT,
    CTRANSPOSE_ERROR_INVALID_INPUT,
    CTRANSPOSE_ERROR_INTERNAL,
];

pub const CTRANSPOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CTRANSPOSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CTRANSPOSE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::ctranspose")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Transpose,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("transpose"),
        ProviderHook::Custom("permute"),
        ProviderHook::Unary { name: "unary_conj" },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Uses provider transpose/permute hooks followed by unary_conj; falls back to host conjugate transpose when either hook is missing.",
};

fn builtin_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    builtin_error_with_message(error.message, error)
}

fn builtin_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    builtin_error_with_message(message, &CTRANSPOSE_ERROR_INVALID_ARGUMENT)
}

fn invalid_input(message: impl Into<String>) -> RuntimeError {
    builtin_error_with_message(message, &CTRANSPOSE_ERROR_INVALID_INPUT)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    builtin_error_with_message(message, &CTRANSPOSE_ERROR_INTERNAL)
}

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::ops::ctranspose"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Conjugate transposes act as fusion boundaries so downstream kernels observe the updated layout.",
};

#[runtime_builtin(
    name = "ctranspose",
    category = "math/linalg/ops",
    summary = "Conjugate-transpose arrays.",
    keywords = "ctranspose,conjugate transpose,hermitian,gpu",
    accel = "transpose",
    type_resolver(transpose_type),
    descriptor(crate::builtins::math::linalg::ops::ctranspose::CTRANSPOSE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::ops::ctranspose"
)]
async fn ctranspose_builtin(mut args: Vec<Value>) -> BuiltinResult<Value> {
    let value = match args.len() {
        0 => return Err(builtin_error(&CTRANSPOSE_ERROR_INVALID_ARGUMENT)),
        1 => args.remove(0),
        _ => return Err(invalid_argument("ctranspose: too many input arguments")),
    };
    match value {
        Value::GpuTensor(handle) => ctranspose_gpu(handle).await,
        Value::Complex(re, im) => ctranspose_complex_scalar(re, im),
        Value::ComplexTensor(ct) => ctranspose_complex_tensor(ct),
        Value::Tensor(t) => Ok(tensor::tensor_into_value(ctranspose_tensor(t)?)),
        Value::SparseTensor(s) => Ok(Value::SparseTensor(
            transpose_real_sparse_tensor(s).map_err(|e| internal_error(format!("{NAME}: {e}")))?,
        )),
        Value::LogicalArray(la) => Ok(Value::LogicalArray(ctranspose_logical_array(la)?)),
        Value::CharArray(ca) => Ok(Value::CharArray(ctranspose_char_array(ca)?)),
        Value::StringArray(sa) => Ok(Value::StringArray(ctranspose_string_array(sa)?)),
        Value::Cell(ca) => Ok(Value::Cell(ctranspose_cell_array(ca)?)),
        Value::Num(n) => Ok(Value::Num(n)),
        Value::Int(i) => Ok(Value::Int(i)),
        Value::Bool(b) => Ok(Value::Bool(b)),
        Value::String(s) => Ok(Value::String(s)),
        other => Err(invalid_input(format!(
            "ctranspose: unsupported input type {other:?}"
        ))),
    }
}

fn ctranspose_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let rank = tensor.shape.len();
    if rank <= 2 {
        ctranspose_tensor_matrix(&tensor)
    } else {
        let order = ctranspose_order(rank);
        permute_tensor(NAME, tensor, &order)
    }
}

fn ctranspose_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let rank = ct.shape.len();
    if rank == 0 {
        return ctranspose_complex_tensor_value(ct);
    }
    if rank <= 2 {
        let data = ctranspose_complex_matrix(&ct);
        let shape = vec![ct.cols, ct.rows];
        let transposed = ComplexTensor::new(data, shape.clone())
            .map_err(|e| internal_error(format!("{NAME}: {e}")))?;
        ctranspose_complex_tensor_value(transposed)
    } else {
        let order = ctranspose_order(rank);
        let permuted = permute_complex_tensor(NAME, ct, &order)?;
        ctranspose_complex_tensor_value(permuted)
    }
}

fn ctranspose_complex_tensor_preserve_complex(ct: ComplexTensor) -> BuiltinResult<ComplexTensor> {
    let rank = ct.shape.len();
    let transposed = if rank == 0 {
        ct
    } else if rank <= 2 {
        let data = ctranspose_complex_matrix(&ct);
        ComplexTensor::new(data, vec![ct.cols, ct.rows])
            .map_err(|e| internal_error(format!("{NAME}: {e}")))?
    } else {
        let order = ctranspose_order(rank);
        permute_complex_tensor(NAME, ct, &order)?
    };
    let shape = transposed.shape.clone();
    let data = transposed
        .data
        .into_iter()
        .map(|(re, im)| (re, -im))
        .collect();
    ComplexTensor::new(data, shape).map_err(|e| internal_error(format!("{NAME}: {e}")))
}

fn ctranspose_complex_tensor_value(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let data = ct.data;
    let mut all_real = true;
    let mut conj_data = Vec::with_capacity(data.len());
    for (re, im) in data {
        let imag = -im;
        if imag != 0.0 || imag.is_nan() {
            all_real = false;
        }
        conj_data.push((re, imag));
    }
    if all_real {
        let real: Vec<f64> = conj_data.iter().map(|(re, _)| *re).collect();
        let tensor =
            Tensor::new(real, shape).map_err(|e| internal_error(format!("{NAME}: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    } else {
        let tensor = ComplexTensor::new(conj_data, shape)
            .map_err(|e| internal_error(format!("{NAME}: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn ctranspose_complex_scalar(re: f64, im: f64) -> BuiltinResult<Value> {
    let imag = -im;
    if imag == 0.0 && !imag.is_nan() {
        Ok(Value::Num(re))
    } else {
        Ok(Value::Complex(re, imag))
    }
}

fn ctranspose_logical_array(la: LogicalArray) -> BuiltinResult<LogicalArray> {
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
        LogicalArray::new(out, new_shape).map_err(|e| internal_error(format!("{NAME}: {e}")))
    } else {
        let order = ctranspose_order(rank);
        permute_logical_array(NAME, la, &order)
    }
}

fn ctranspose_char_array(ca: CharArray) -> BuiltinResult<CharArray> {
    let rows = ca.rows;
    let cols = ca.cols;
    if ca.data.is_empty() {
        return CharArray::new(Vec::new(), cols, rows)
            .map_err(|e| internal_error(format!("{NAME}: {e}")));
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
    CharArray::new(out, cols, rows).map_err(|e| internal_error(format!("{NAME}: {e}")))
}

fn ctranspose_string_array(sa: StringArray) -> BuiltinResult<StringArray> {
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
            }
            shape
        } else {
            vec![cols, rows]
        };
        StringArray::new(out, new_shape).map_err(|e| internal_error(format!("{NAME}: {e}")))
    } else {
        let order = ctranspose_order(rank);
        permute_string_array(NAME, sa, &order)
    }
}

fn ctranspose_cell_array(ca: CellArray) -> BuiltinResult<CellArray> {
    let rows = ca.rows;
    let cols = ca.cols;
    let mut out = Vec::with_capacity(ca.data.len());
    for c in 0..cols {
        for r in 0..rows {
            let idx = r * cols + c;
            out.push(ca.data[idx].clone());
        }
    }
    CellArray::new_handles(out, cols, rows).map_err(|e| internal_error(format!("{NAME}: {e}")))
}

async fn ctranspose_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let rank = handle.shape.len();
    if rank == 0 {
        return Ok(Value::GpuTensor(handle));
    }
    let input_complex =
        runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved;

    if let Some(provider) =
        runmat_accelerate_api::provider_for_handle(&handle).or_else(runmat_accelerate_api::provider)
    {
        let mut transposed: Option<GpuTensorHandle> = None;
        if rank <= 2 {
            match provider.transpose(&handle) {
                Ok(out) => transposed = Some(out),
                Err(err) => {
                    let info = provider.device_info_struct();
                    warn!(
                        "ctranspose: provider {} (backend: {}) missing transpose hook; falling back ({err})",
                        info.name,
                        info.backend.as_deref().unwrap_or("unknown")
                    );
                }
            }
        } else {
            let order = ctranspose_order(rank);
            let zero_based: Vec<usize> = order.iter().map(|&idx| idx - 1).collect();
            match provider.permute(&handle, &zero_based) {
                Ok(out) => transposed = Some(out),
                Err(err) => {
                    let info = provider.device_info_struct();
                    warn!(
                        "ctranspose: provider {} (backend: {}) missing permute hook; falling back ({err})",
                        info.name,
                        info.backend.as_deref().unwrap_or("unknown")
                    );
                }
            }
        }

        if let Some(transposed_handle) = transposed {
            match provider.unary_conj(&transposed_handle).await {
                Ok(conjugated) => {
                    if let Some(info) =
                        runmat_accelerate_api::handle_transpose_info(&transposed_handle)
                    {
                        runmat_accelerate_api::record_handle_transpose(
                            &conjugated,
                            info.base_rows,
                            info.base_cols,
                        );
                    }
                    if input_complex
                        || runmat_accelerate_api::handle_storage(&conjugated)
                            == GpuTensorStorage::ComplexInterleaved
                    {
                        return Ok(gpu_helpers::complex_gpu_value(conjugated));
                    }
                    return Ok(gpu_helpers::resident_gpu_value(conjugated));
                }
                Err(err) => {
                    let info = provider.device_info_struct();
                    warn!(
                        "ctranspose: provider {} (backend: {}) missing unary_conj hook; falling back ({err})",
                        info.name,
                        info.backend.as_deref().unwrap_or("unknown")
                    );
                }
            }
        }
    }

    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?;
    let transposed_value = match gathered {
        Value::Tensor(tensor) => tensor::tensor_into_value(ctranspose_tensor(tensor)?),
        Value::ComplexTensor(tensor) if input_complex => {
            Value::ComplexTensor(ctranspose_complex_tensor_preserve_complex(tensor)?)
        }
        Value::ComplexTensor(tensor) => ctranspose_complex_tensor(tensor)?,
        Value::LogicalArray(array) => Value::LogicalArray(ctranspose_logical_array(array)?),
        Value::Num(n) => Value::Num(n),
        Value::Complex(re, im) => ctranspose_complex_scalar(re, im)?,
        Value::Bool(flag) => Value::Bool(flag),
        Value::Int(value) => Value::Int(value),
        other => {
            return Err(invalid_input(format!(
                "ctranspose: unsupported gathered gpu value {other:?}"
            )));
        }
    };
    if let Some(provider) =
        runmat_accelerate_api::provider_for_handle(&handle).or_else(runmat_accelerate_api::provider)
    {
        match &transposed_value {
            Value::Tensor(transposed) => {
                let view = HostTensorView {
                    data: &transposed.data,
                    shape: &transposed.shape,
                };
                match provider.upload(&view) {
                    Ok(uploaded) => return Ok(gpu_helpers::resident_gpu_value(uploaded)),
                    Err(err) => warn!(
                        "ctranspose: re-upload after host fallback failed; returning host tensor ({err})"
                    ),
                }
            }
            Value::ComplexTensor(transposed) => {
                match gpu_helpers::upload_complex_tensor(provider, transposed) {
                    Ok(uploaded) => return Ok(gpu_helpers::complex_gpu_value(uploaded)),
                    Err(err) => warn!(
                        "ctranspose: complex re-upload after host fallback failed; returning host tensor ({err})"
                    ),
                }
            }
            Value::LogicalArray(transposed) => {
                let data: Vec<f64> = transposed
                    .data
                    .iter()
                    .map(|&bit| if bit == 0 { 0.0 } else { 1.0 })
                    .collect();
                let view = HostTensorView {
                    data: &data,
                    shape: &transposed.shape,
                };
                match provider.upload(&view) {
                    Ok(uploaded) => return Ok(gpu_helpers::logical_gpu_value(uploaded)),
                    Err(err) => warn!(
                        "ctranspose: logical re-upload after host fallback failed; returning host logical array ({err})"
                    ),
                }
            }
            _ => {}
        }
    }
    Ok(transposed_value)
}

fn ctranspose_order(rank: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (1..=rank.max(2)).collect();
    if order.len() >= 2 {
        order.swap(0, 1);
    }
    if order.len() > rank && rank < 2 {
        order.truncate(rank.max(2));
    }
    order
}

fn ctranspose_tensor_matrix(tensor: &Tensor) -> BuiltinResult<Tensor> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    if tensor.data.is_empty() {
        return Tensor::new(Vec::new(), vec![cols, rows])
            .map_err(|e| internal_error(format!("{NAME}: {e}")));
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
    Tensor::new(out, vec![cols, rows]).map_err(|e| internal_error(format!("{NAME}: {e}")))
}

fn ctranspose_complex_matrix(ct: &ComplexTensor) -> Vec<(f64, f64)> {
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
    use crate::builtins::array::shape::permute::permute_tensor;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_backend;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        IntValue, LogicalArray, ResolveContext, StringArray, StructValue, Tensor, Type,
    };

    fn call_ctranspose(value: Value) -> BuiltinResult<Value> {
        block_on(super::ctranspose_builtin(vec![value]))
    }

    fn call_ctranspose_args(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::ctranspose_builtin(args))
    }

    fn tensor(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_real_matrix_matches_transpose() {
        let input = tensor(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]);
        let value = call_ctranspose(Value::Tensor(input)).expect("ctranspose");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 2]);
                assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn ctranspose_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = CTRANSPOSE_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"B = ctranspose(A)"));
    }

    #[test]
    fn ctranspose_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = CTRANSPOSE_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.CTRANSPOSE.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.CTRANSPOSE.INVALID_INPUT"));
        assert!(codes.contains(&"RM.CTRANSPOSE.INTERNAL"));
    }

    #[test]
    fn ctranspose_invalid_argument_identifier_is_stable() {
        match call_ctranspose_args(Vec::new()) {
            Err(err) => assert_eq!(
                err.identifier(),
                CTRANSPOSE_ERROR_INVALID_ARGUMENT.identifier
            ),
            Ok(_) => panic!("expected invalid argument error"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_complex_matrix_conjugates() {
        let data = vec![(1.0, 2.0), (3.0, -4.0), (5.0, 0.0), (6.0, -7.0)];
        let ct = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let value = call_ctranspose(Value::ComplexTensor(ct)).expect("ctranspose");
        match value {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(
                    out.data,
                    vec![(1.0, -2.0), (5.0, -0.0), (3.0, 4.0), (6.0, 7.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_complex_tensor_realises_real_when_imag_zero() {
        let data = vec![(1.0, 0.0), (2.0, -0.0)];
        let ct = ComplexTensor::new(data, vec![1, 2]).unwrap();
        let value = call_ctranspose(Value::ComplexTensor(ct)).expect("ctranspose");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![1.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_complex_scalar() {
        let result = call_ctranspose(Value::Complex(2.0, 3.0)).expect("ctranspose");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 2.0).abs() < 1e-12);
                assert!((im + 3.0).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_logical_mask() {
        let la = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let value = call_ctranspose(Value::LogicalArray(la)).expect("ctranspose");
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
    fn ctranspose_char_matrix() {
        let ca = CharArray::new("runmat".chars().collect(), 2, 3).unwrap();
        let value = call_ctranspose(Value::CharArray(ca)).expect("ctranspose");
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
    fn ctranspose_string_array_transposes() {
        let data = vec![
            "r0c0".to_string(),
            "r1c0".to_string(),
            "r0c1".to_string(),
            "r1c1".to_string(),
        ];
        let sa = StringArray::new(data, vec![2, 2]).unwrap();
        let value = call_ctranspose(Value::StringArray(sa)).expect("ctranspose");
        match value {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(
                    out.data,
                    vec![
                        "r0c0".to_string(),
                        "r0c1".to_string(),
                        "r1c0".to_string(),
                        "r1c1".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_cell_array() {
        let cells = vec![
            Value::from(1),
            Value::from(2),
            Value::from(3),
            Value::from(4),
        ];
        let cell_array = CellArray::new(cells, 2, 2).unwrap();
        let value = call_ctranspose(Value::Cell(cell_array)).expect("ctranspose");
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
    fn ctranspose_scalar_types_identity() {
        assert_eq!(
            call_ctranspose(Value::Num(std::f64::consts::PI)).unwrap(),
            Value::Num(std::f64::consts::PI)
        );
        assert_eq!(
            call_ctranspose(Value::Int(IntValue::I32(5))).unwrap(),
            Value::Int(IntValue::I32(5))
        );
        assert_eq!(
            call_ctranspose(Value::Bool(true)).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_tensor_swaps_first_two_dims_for_nd() {
        let data: Vec<f64> = (1..=12).map(|n| n as f64).collect();
        let input = tensor(&data, &[2, 3, 2]);
        let expected = permute_tensor(NAME, input.clone(), &[2, 1, 3]).unwrap();
        let value = call_ctranspose(Value::Tensor(input)).expect("ctranspose");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_struct_unsupported() {
        let mut st = StructValue::new();
        st.fields.insert("field".to_string(), Value::Num(1.0));
        let err = unwrap_error(call_ctranspose(Value::Struct(st)).unwrap_err());
        assert!(err.message().contains("unsupported input type"));
    }

    #[test]
    fn ctranspose_type_swaps_dims() {
        let out = transpose_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(5)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(5), Some(2)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let t = tensor(&[1.0, 4.0, 2.0, 5.0], &[2, 2]);
            let view = HostTensorView {
                data: &t.data,
                shape: &t.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = call_ctranspose(Value::GpuTensor(handle)).expect("ctranspose");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 2.0, 4.0, 5.0]);
        });
    }

    #[test]
    fn ctranspose_complex_gpu_inprocess_stays_resident() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(
                vec![
                    (1.0, 2.0),
                    (3.0, -4.0),
                    (5.0, 6.0),
                    (7.0, -8.0),
                    (9.0, 10.0),
                    (11.0, -12.0),
                ],
                vec![2, 3],
            )
            .unwrap();
            let expected = match call_ctranspose(Value::ComplexTensor(complex.clone()))
                .expect("cpu ctranspose")
            {
                Value::ComplexTensor(tensor) => tensor,
                other => panic!("expected complex tensor result, got {other:?}"),
            };

            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let result = call_ctranspose(Value::GpuTensor(handle)).expect("gpu ctranspose");
            let Value::GpuTensor(out_handle) = result else {
                panic!("expected complex gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out_handle),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(
                crate::builtins::math::fft::common::gather_gpu_complex_tensor(&out_handle, NAME),
            )
            .expect("gather complex");
            assert_eq!(gathered.shape, expected.shape);
            assert_eq!(gathered.data, expected.data);
        });
    }

    #[test]
    fn ctranspose_complex_gpu_all_real_input_preserves_complex_storage() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(
                vec![(1.0, 0.0), (2.0, -0.0), (3.0, 0.0), (4.0, -0.0)],
                vec![2, 2],
            )
            .unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let result = call_ctranspose(Value::GpuTensor(handle)).expect("gpu ctranspose");
            let Value::GpuTensor(out_handle) = result else {
                panic!("expected complex gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out_handle),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(
                crate::builtins::math::fft::common::gather_gpu_complex_tensor(&out_handle, NAME),
            )
            .expect("gather complex");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(
                gathered.data,
                vec![(1.0, -0.0), (3.0, -0.0), (2.0, 0.0), (4.0, 0.0)]
            );
        });
    }

    #[test]
    fn ctranspose_complex_gpu_inprocess_nd_stays_resident() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(
                vec![
                    (1.0, -1.0),
                    (2.0, -2.0),
                    (3.0, -3.0),
                    (4.0, -4.0),
                    (5.0, -5.0),
                    (6.0, -6.0),
                    (7.0, -7.0),
                    (8.0, -8.0),
                ],
                vec![2, 2, 2],
            )
            .unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let result = call_ctranspose(Value::GpuTensor(handle)).expect("gpu ctranspose");
            let Value::GpuTensor(out_handle) = result else {
                panic!("expected complex gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out_handle),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(
                crate::builtins::math::fft::common::gather_gpu_complex_tensor(&out_handle, NAME),
            )
            .expect("gather complex");
            assert_eq!(gathered.shape, vec![2, 2, 2]);
            assert_eq!(
                gathered.data,
                vec![
                    (1.0, 1.0),
                    (3.0, 3.0),
                    (2.0, 2.0),
                    (4.0, 4.0),
                    (5.0, 5.0),
                    (7.0, 7.0),
                    (6.0, 6.0),
                    (8.0, 8.0),
                ]
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ctranspose_wgpu_matches_cpu() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) =
            wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default())
        else {
            return;
        };
        let data: Vec<f64> = (1..=12).map(|n| n as f64).collect();
        let tensor = Tensor::new(data, vec![3, 4]).expect("tensor");
        let cpu_value = call_ctranspose(Value::Tensor(tensor.clone())).expect("cpu");
        let cpu_tensor = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = call_ctranspose(Value::GpuTensor(handle)).expect("gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu_tensor.shape);
        assert_eq!(gathered.data, cpu_tensor.data);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn ctranspose_wgpu_complex_matches_cpu_and_stays_resident() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) =
            wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default())
        else {
            return;
        };
        let complex = ComplexTensor::new(
            vec![
                (1.0, 2.0),
                (3.0, -4.0),
                (5.0, 6.0),
                (7.0, -8.0),
                (9.0, 10.0),
                (11.0, -12.0),
            ],
            vec![2, 3],
        )
        .unwrap();
        let expected =
            match call_ctranspose(Value::ComplexTensor(complex.clone())).expect("cpu ctranspose") {
                Value::ComplexTensor(tensor) => tensor,
                other => panic!("expected complex tensor result, got {other:?}"),
            };

        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let result = call_ctranspose(Value::GpuTensor(handle)).expect("gpu ctranspose");
        let Value::GpuTensor(out_handle) = result else {
            panic!("expected complex gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out_handle),
            GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(
            crate::builtins::math::fft::common::gather_gpu_complex_tensor(&out_handle, NAME),
        )
        .expect("gather complex");
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn ctranspose_wgpu_complex_nd_matches_cpu_and_stays_resident() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) =
            wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default())
        else {
            return;
        };
        let complex = ComplexTensor::new(
            vec![
                (1.0, -1.0),
                (2.0, -2.0),
                (3.0, -3.0),
                (4.0, -4.0),
                (5.0, -5.0),
                (6.0, -6.0),
                (7.0, -7.0),
                (8.0, -8.0),
            ],
            vec![2, 2, 2],
        )
        .unwrap();
        let expected =
            ctranspose_complex_tensor_preserve_complex(complex.clone()).expect("cpu ctranspose");

        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let result = call_ctranspose(Value::GpuTensor(handle)).expect("gpu ctranspose");
        let Value::GpuTensor(out_handle) = result else {
            panic!("expected complex gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out_handle),
            GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(
            crate::builtins::math::fft::common::gather_gpu_complex_tensor(&out_handle, NAME),
        )
        .expect("gather complex");
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }
}
