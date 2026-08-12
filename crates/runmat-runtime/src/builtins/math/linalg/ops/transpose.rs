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
use crate::builtins::math::linalg::ops::transpose_real_sparse_tensor;
use crate::builtins::math::linalg::type_resolvers::matrix_transpose_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use log::warn;
use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_builtins::IntegerStorage;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

const NAME: &str = "transpose";

const TRANSPOSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value with first two dimensions swapped.",
}];

const TRANSPOSE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar/array value.",
}];

const TRANSPOSE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = transpose(A)",
    inputs: &TRANSPOSE_INPUTS,
    outputs: &TRANSPOSE_OUTPUT,
}];

const TRANSPOSE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRANSPOSE.INVALID_ARGUMENT",
    identifier: Some("RunMat:transpose:InvalidArgument"),
    when: "Call does not provide exactly one input argument.",
    message: "transpose: invalid argument",
};

const TRANSPOSE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRANSPOSE.INVALID_INPUT",
    identifier: Some("RunMat:transpose:InvalidInput"),
    when: "Input type is unsupported for transpose or the input has a nonsingleton dimension above two.",
    message: "transpose: unsupported input type",
};

const TRANSPOSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRANSPOSE.INTERNAL",
    identifier: Some("RunMat:transpose:Internal"),
    when: "Runtime cannot materialize transpose output.",
    message: "transpose: internal runtime failure",
};

const TRANSPOSE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    TRANSPOSE_ERROR_INVALID_ARGUMENT,
    TRANSPOSE_ERROR_INVALID_INPUT,
    TRANSPOSE_ERROR_INTERNAL,
];

pub const TRANSPOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRANSPOSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TRANSPOSE_ERRORS,
};

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
    builtin_error_with_message(message, &TRANSPOSE_ERROR_INVALID_ARGUMENT)
}

fn invalid_input(message: impl Into<String>) -> RuntimeError {
    builtin_error_with_message(message, &TRANSPOSE_ERROR_INVALID_INPUT)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    builtin_error_with_message(message, &TRANSPOSE_ERROR_INTERNAL)
}

fn ensure_vector_or_matrix_shape(shape: &[usize]) -> BuiltinResult<()> {
    if !super::is_vector_or_matrix_shape(shape) {
        return Err(invalid_input(
            "transpose: N-D arrays are not supported; use permute to reorder N-D dimensions",
        ));
    }
    Ok(())
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(err.message()).with_builtin(NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = err.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !err.context.call_stack.is_empty() {
        builder = builder.with_call_stack(err.context.call_stack.clone());
    }
    if let Some(phase) = err.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(err).build()
}

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
    summary = "Transpose vectors and matrices without conjugation.",
    keywords = "transpose,swap rows and columns,non-conjugate",
    accel = "transpose",
    type_resolver(matrix_transpose_type),
    descriptor(crate::builtins::math::linalg::ops::transpose::TRANSPOSE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::ops::transpose"
)]
async fn transpose_builtin(mut args: Vec<Value>) -> BuiltinResult<Value> {
    let value = match args.len() {
        0 => return Err(builtin_error(&TRANSPOSE_ERROR_INVALID_ARGUMENT)),
        1 => args.remove(0),
        _ => return Err(invalid_argument("transpose: too many input arguments")),
    };
    match value {
        Value::GpuTensor(handle) => transpose_gpu(handle).await,
        Value::Tensor(t) => Ok(tensor::tensor_into_value(transpose_tensor(t)?)),
        Value::SparseTensor(s) => Ok(Value::SparseTensor(
            transpose_real_sparse_tensor(s).map_err(|e| internal_error(format!("{NAME}: {e}")))?,
        )),
        Value::ComplexTensor(ct) => Ok(Value::ComplexTensor(transpose_complex_tensor(ct)?)),
        Value::LogicalArray(la) => Ok(Value::LogicalArray(transpose_logical_array(la)?)),
        Value::CharArray(ca) => Ok(Value::CharArray(transpose_char_array(ca)?)),
        Value::StringArray(sa) => Ok(Value::StringArray(transpose_string_array(sa)?)),
        Value::Cell(ca) => Ok(Value::Cell(transpose_cell_array(ca)?)),
        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
        Value::Num(n) => Ok(Value::Num(n)),
        Value::Int(i) => Ok(Value::Int(i)),
        Value::Bool(b) => Ok(Value::Bool(b)),
        Value::String(s) => Ok(Value::String(s)),
        other => Err(invalid_input(format!(
            "transpose: unsupported input type {other:?}"
        ))),
    }
}

fn transpose_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    ensure_vector_or_matrix_shape(&tensor.shape)?;
    let rank = tensor.shape.len();
    if rank == 0 {
        return Ok(tensor);
    }
    let order = transpose_order(rank);
    permute_tensor(NAME, tensor, &order)
}

fn transpose_complex_tensor(ct: ComplexTensor) -> BuiltinResult<ComplexTensor> {
    ensure_vector_or_matrix_shape(&ct.shape)?;
    let rank = ct.shape.len();
    if rank == 0 {
        return Ok(ct);
    }
    let order = transpose_order(rank);
    permute_complex_tensor(NAME, ct, &order)
}

#[cfg(test)]
pub(crate) fn transpose_integer_storage(
    storage: IntegerStorage,
    rows: usize,
    cols: usize,
) -> IntegerStorage {
    fn transpose_values<T: Copy>(values: Vec<T>, rows: usize, cols: usize) -> Vec<T> {
        if values.is_empty() {
            return values;
        }
        let mut output = values.clone();
        for row in 0..rows {
            for col in 0..cols {
                output[col + row * cols] = values[row + col * rows];
            }
        }
        output
    }

    match storage {
        IntegerStorage::I8(values) => IntegerStorage::I8(transpose_values(values, rows, cols)),
        IntegerStorage::I16(values) => IntegerStorage::I16(transpose_values(values, rows, cols)),
        IntegerStorage::I32(values) => IntegerStorage::I32(transpose_values(values, rows, cols)),
        IntegerStorage::I64(values) => IntegerStorage::I64(transpose_values(values, rows, cols)),
        IntegerStorage::U8(values) => IntegerStorage::U8(transpose_values(values, rows, cols)),
        IntegerStorage::U16(values) => IntegerStorage::U16(transpose_values(values, rows, cols)),
        IntegerStorage::U32(values) => IntegerStorage::U32(transpose_values(values, rows, cols)),
        IntegerStorage::U64(values) => IntegerStorage::U64(transpose_values(values, rows, cols)),
    }
}

fn transpose_logical_array(la: LogicalArray) -> BuiltinResult<LogicalArray> {
    ensure_vector_or_matrix_shape(&la.shape)?;
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
        let order = transpose_order(rank);
        permute_logical_array(NAME, la, &order)
    }
}

fn transpose_char_array(ca: CharArray) -> BuiltinResult<CharArray> {
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

fn transpose_string_array(sa: StringArray) -> BuiltinResult<StringArray> {
    ensure_vector_or_matrix_shape(&sa.shape)?;
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
        StringArray::new(out, new_shape).map_err(|e| internal_error(format!("{NAME}: {e}")))
    } else {
        let order = transpose_order(rank);
        permute_string_array(NAME, sa, &order)
    }
}

fn transpose_cell_array(ca: CellArray) -> BuiltinResult<CellArray> {
    let rows = ca.rows;
    let cols = ca.cols;
    let mut out = Vec::with_capacity(ca.data.len());
    for c in 0..cols {
        for r in 0..rows {
            let idx = r * cols + c;
            out.push(ca.data[idx].clone());
        }
    }
    CellArray::new(out, cols, rows).map_err(|e| internal_error(format!("{NAME}: {e}")))
}

async fn transpose_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    ensure_vector_or_matrix_shape(&handle.shape)?;
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
    let host = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(map_control_flow)?;
    let transposed = transpose_tensor(host)?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        match gpu_helpers::upload_tensor(provider, &transposed) {
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_backend;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, ResolveContext, Tensor, Type,
    };

    fn call_transpose(value: Value) -> BuiltinResult<Value> {
        block_on(super::transpose_builtin(vec![value]))
    }

    fn call_transpose_args(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::transpose_builtin(args))
    }

    fn all_integer_matrix_storages() -> [IntegerStorage; 8] {
        [
            IntegerStorage::I8(vec![1, 2, 3, 4]),
            IntegerStorage::I16(vec![1, 2, 3, 4]),
            IntegerStorage::I32(vec![1, 2, 3, 4]),
            IntegerStorage::I64(vec![1, 2, 3, 4]),
            IntegerStorage::U8(vec![1, 2, 3, 4]),
            IntegerStorage::U16(vec![1, 2, 3, 4]),
            IntegerStorage::U32(vec![1, 2, 3, 4]),
            IntegerStorage::U64(vec![1, 2, 3, 4]),
        ]
    }

    #[test]
    fn transpose_preserves_every_real_and_complex_integer_class() {
        for storage in all_integer_matrix_storages() {
            let expected = transpose_integer_storage(storage.clone(), 2, 2);
            let class = storage.class_name();
            let real = Tensor::new_integer(storage.clone(), vec![2, 2]).expect("real tensor");
            let Value::Tensor(real_result) =
                call_transpose(Value::Tensor(real)).expect("real transpose")
            else {
                panic!("expected real tensor for {class}");
            };
            assert_eq!(real_result.integer_storage(), Some(&expected), "{class}");

            let complex = ComplexTensor::new_integer(
                IntegerComplexStorage::new(storage.clone(), storage).expect("complex storage"),
                vec![2, 2],
            )
            .expect("complex tensor");
            let Value::ComplexTensor(complex_result) =
                call_transpose(Value::ComplexTensor(complex)).expect("complex transpose")
            else {
                panic!("expected complex tensor for {class}");
            };
            let result_storage = complex_result.integer_storage().expect("integer storage");
            assert_eq!(result_storage.real, expected, "{class} real");
            assert_eq!(
                result_storage.imag, result_storage.real,
                "{class} imaginary"
            );
        }
    }

    #[test]
    fn transpose_preserves_typed_complex_integer_components_exactly() {
        let input = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![u64::MAX, 2, 3, 4, 5, 1_u64 << 63]),
                IntegerStorage::U64(vec![1, 2, 3, 4, 5, 6]),
            )
            .expect("storage"),
            vec![2, 3],
        )
        .expect("tensor");

        let Value::ComplexTensor(result) =
            call_transpose(Value::ComplexTensor(input)).expect("transpose")
        else {
            panic!("expected complex tensor");
        };
        let storage = result.integer_storage().expect("exact integer storage");
        assert_eq!(result.shape, vec![3, 2]);
        assert_eq!(
            storage.real,
            IntegerStorage::U64(vec![u64::MAX, 3, 5, 2, 4, 1_u64 << 63])
        );
        assert_eq!(storage.imag, IntegerStorage::U64(vec![1, 3, 5, 2, 4, 6]));
    }

    #[test]
    fn transpose_preserves_typed_real_integer_storage_exactly() {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 2, 3, 4, 5, 1_u64 << 63]),
            vec![2, 3],
        )
        .expect("tensor");

        let Value::Tensor(result) = call_transpose(Value::Tensor(input)).expect("transpose") else {
            panic!("expected tensor");
        };
        assert_eq!(result.shape, vec![3, 2]);
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                u64::MAX,
                3,
                5,
                2,
                4,
                1_u64 << 63
            ]))
        );
    }

    #[test]
    fn transpose_preserves_empty_typed_real_integer_storage() {
        let input =
            Tensor::new_integer(IntegerStorage::I64(Vec::new()), vec![0, 3]).expect("tensor");

        let Value::Tensor(result) = call_transpose(Value::Tensor(input)).expect("transpose") else {
            panic!("expected tensor");
        };
        assert_eq!(result.shape, vec![3, 0]);
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::I64(Vec::new()))
        );
    }

    #[test]
    fn transpose_preserves_native_single_storage() {
        let input = Tensor::from_f32(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3])
            .expect("single tensor");
        let Value::Tensor(result) = call_transpose(Value::Tensor(input)).expect("transpose") else {
            panic!("expected tensor");
        };
        assert_eq!(result.shape, vec![3, 2]);
        assert_eq!(
            result.into_numeric_storage().unwrap(),
            runmat_builtins::NumericStorage::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );
    }

    fn tensor(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_numeric_matrix() {
        let input = tensor(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]);
        let value = call_transpose(Value::Tensor(input)).expect("transpose");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 2]);
                assert_eq!(out.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn transpose_type_swaps_first_two_dims() {
        let out = matrix_transpose_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(4)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(2)])
            }
        );
    }

    #[test]
    fn transpose_type_does_not_claim_nd_shapes() {
        let context = ResolveContext::new(Vec::new());
        assert_eq!(
            matrix_transpose_type(
                &[Type::Tensor {
                    shape: Some(vec![Some(2), Some(3), Some(4)]),
                }],
                &context,
            ),
            Type::Unknown
        );
        assert_eq!(
            matrix_transpose_type(
                &[Type::Logical {
                    shape: Some(vec![Some(2), Some(3), None]),
                }],
                &context,
            ),
            Type::Unknown
        );
        assert_eq!(
            matrix_transpose_type(
                &[Type::Tensor {
                    shape: Some(vec![Some(2), Some(3), Some(1)]),
                }],
                &context,
            ),
            Type::Tensor {
                shape: Some(vec![Some(3), Some(2), Some(1)])
            }
        );
    }

    #[test]
    fn transpose_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = TRANSPOSE_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"B = transpose(A)"));
    }

    #[test]
    fn transpose_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = TRANSPOSE_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.TRANSPOSE.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.TRANSPOSE.INVALID_INPUT"));
        assert!(codes.contains(&"RM.TRANSPOSE.INTERNAL"));
    }

    #[test]
    fn transpose_invalid_argument_identifier_is_stable() {
        match call_transpose_args(Vec::new()) {
            Err(err) => assert_eq!(
                err.identifier(),
                TRANSPOSE_ERROR_INVALID_ARGUMENT.identifier
            ),
            Ok(_) => panic!("expected invalid argument error"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_vector_to_column() {
        let input = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
        let value = call_transpose(Value::Tensor(input)).expect("transpose");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(out.materialize_f64(), vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_complex_does_not_conjugate() {
        let data = vec![(1.0, 2.0), (3.0, -4.0)];
        let ct = ComplexTensor::new(data, vec![1, 2]).unwrap();
        let value = call_transpose(Value::ComplexTensor(ct)).expect("transpose");
        match value {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.materialize_f64(), vec![(1.0, 2.0), (3.0, -4.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_rejects_nd_tensor_and_directs_to_permute() {
        let data: Vec<f64> = (1..=24).map(|n| n as f64).collect();
        let tensor = tensor(&data, &[2, 3, 4]);
        let err = call_transpose(Value::Tensor(tensor)).expect_err("N-D transpose must fail");
        assert_eq!(err.identifier(), Some("RunMat:transpose:InvalidInput"));
        assert!(err.message().contains("use permute"));
    }

    #[test]
    fn transpose_accepts_explicit_trailing_singleton_dimensions() {
        let tensor = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2, 1]);
        let Value::Tensor(result) =
            call_transpose(Value::Tensor(tensor)).expect("effective matrix")
        else {
            panic!("expected tensor");
        };
        assert_eq!(result.shape, vec![2, 2, 1]);
        assert_eq!(result.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn transpose_rejects_nd_complex_logical_and_string_arrays() {
        let values = [
            (
                "complex",
                Value::ComplexTensor(
                    ComplexTensor::new(vec![(1.0, 1.0); 4], vec![2, 1, 2]).expect("complex tensor"),
                ),
            ),
            (
                "logical",
                Value::LogicalArray(
                    LogicalArray::new(vec![1, 0, 1, 0], vec![2, 1, 2]).expect("logical array"),
                ),
            ),
            (
                "string",
                Value::StringArray(
                    StringArray::new(
                        vec!["a".into(), "b".into(), "c".into(), "d".into()],
                        vec![2, 1, 2],
                    )
                    .expect("string array"),
                ),
            ),
        ];
        for (label, value) in values {
            let err = call_transpose(value).expect_err("N-D transpose must fail");
            assert_eq!(
                err.identifier(),
                Some("RunMat:transpose:InvalidInput"),
                "{label}"
            );
            assert!(err.message().contains("use permute"), "{label}");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_logical_mask() {
        let la = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let value = call_transpose(Value::LogicalArray(la)).expect("transpose");
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
        let value = call_transpose(Value::CharArray(ca)).expect("transpose");
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
        let value = call_transpose(Value::StringArray(sa)).expect("transpose");
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
        let value = call_transpose(Value::Cell(cell_array)).expect("transpose");
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
            call_transpose(Value::Num(std::f64::consts::PI)).unwrap(),
            Value::Num(std::f64::consts::PI)
        );
        assert_eq!(
            call_transpose(Value::Complex(1.0, -2.0)).unwrap(),
            Value::Complex(1.0, -2.0)
        );
        assert_eq!(
            call_transpose(Value::Int(IntValue::I32(5))).unwrap(),
            Value::Int(IntValue::I32(5))
        );
        assert_eq!(
            call_transpose(Value::Bool(true)).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let t = tensor(&[1.0, 4.0, 2.0, 5.0], &[2, 2]);
            let view = HostTensorView {
                data: &t.materialize_f64(),
                shape: &t.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = call_transpose(Value::GpuTensor(handle)).expect("transpose");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 2.0, 4.0, 5.0]);
        });
    }

    #[test]
    fn transpose_gpu_preserves_every_real_integer_class() {
        test_support::with_test_provider(|provider| {
            for storage in all_integer_matrix_storages() {
                let expected = transpose_integer_storage(storage.clone(), 2, 2);
                let class = storage.class_name();
                let input = Tensor::new_integer(storage, vec![2, 2]).expect("integer tensor");
                let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
                let input_type = runmat_accelerate_api::handle_integer_type(&handle);
                let result = call_transpose(Value::GpuTensor(handle)).expect("resident transpose");
                let Value::GpuTensor(result_handle) = &result else {
                    panic!("expected resident result for {class}");
                };
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(result_handle),
                    input_type,
                    "{class}"
                );
                let gathered = test_support::gather(result).expect("integer gather");
                assert_eq!(gathered.integer_storage(), Some(&expected), "{class}");
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn transpose_wgpu_matches_cpu() {
        let Ok(provider) =
            wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default())
        else {
            return;
        };
        let data: Vec<f64> = (1..=12).map(|n| n as f64).collect();
        let tensor = Tensor::new(data, vec![3, 4]).expect("tensor");
        let cpu_value = call_transpose(Value::Tensor(tensor.clone())).expect("cpu transpose");
        let cpu_tensor = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = call_transpose(Value::GpuTensor(handle)).expect("gpu transpose");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu_tensor.shape);
        assert_eq!(gathered.materialize_f64(), cpu_tensor.materialize_f64());
    }
}
