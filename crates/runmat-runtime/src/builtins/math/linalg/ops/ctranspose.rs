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
use crate::builtins::math::linalg::ops::transpose_sparse_tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinErrorDescriptor, CTRANSPOSE_ERROR_INTERNAL, CTRANSPOSE_ERROR_INVALID_ARGUMENT,
    CTRANSPOSE_ERROR_INVALID_INPUT, CTRANSPOSE_INTEGER_SPARSE_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{CTRANSPOSE_DESCRIPTOR, CTRANSPOSE_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, ComplexStorage, ComplexTensor, LogicalArray, StringArray, Tensor, Value,
};

const NAME: &str = "ctranspose";

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

fn ensure_vector_or_matrix_shape(shape: &[usize]) -> BuiltinResult<()> {
    if !super::is_vector_or_matrix_shape(shape) {
        return Err(invalid_input(
            "ctranspose: N-D arrays are not supported; use permute to reorder N-D dimensions",
        ));
    }
    Ok(())
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
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::ops::ctranspose"
)]
pub(super) async fn ctranspose_builtin(
    mut args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    let value = match args.len() {
        0 => return Err(builtin_error(&CTRANSPOSE_ERROR_INVALID_ARGUMENT)),
        1 => args
            .pop_front()
            .expect("validated single transpose argument"),
        _ => return Err(invalid_argument("ctranspose: too many input arguments")),
    };
    let result = match value {
        Value::GpuTensor(handle) => ctranspose_gpu(handle).await,
        Value::Complex(re, im) => ctranspose_complex_scalar(re, im),
        Value::ComplexTensor(ct) => ctranspose_complex_tensor(ct),
        Value::Tensor(t) => Ok(tensor::tensor_into_value(ctranspose_tensor(t)?)),
        Value::SparseTensor(s) => {
            if s.integer_storage().is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &CTRANSPOSE_INTEGER_SPARSE_EXTENSION,
                    NAME,
                )?;
            }
            Ok(Value::SparseTensor(
                transpose_sparse_tensor(s, true)
                    .map_err(|e| internal_error(format!("{NAME}: {e}")))?,
            ))
        }
        Value::LogicalArray(la) => Ok(Value::LogicalArray(ctranspose_logical_array(la)?)),
        Value::CharArray(ca) => Ok(Value::CharArray(ctranspose_char_array(ca)?)),
        Value::StringArray(sa) => Ok(Value::StringArray(ctranspose_string_array(sa)?)),
        Value::Cell(ca) => Ok(Value::Cell(ctranspose_cell_array(ca)?)),
        Value::StructArray(array) => {
            ensure_vector_or_matrix_shape(array.shape())?;
            let order = ctranspose_order(array.shape().len());
            crate::aggregate::structure::operations::permute(array, &order)
                .map(Value::StructArray)
                .map_err(|error| {
                    crate::aggregate::structure::operations::remap(error, |message| {
                        invalid_input(message)
                    })
                })
        }
        Value::Num(n) => Ok(Value::Num(n)),
        Value::Int(i) => Ok(Value::Int(i)),
        Value::Bool(b) => Ok(Value::Bool(b)),
        Value::String(s) => Ok(Value::String(s.into())),
        Value::Struct(s) => Ok(Value::Struct(s)),
        receiver @ Value::Object(_) | receiver @ Value::HandleObject(_) => {
            let class_name = crate::object_receiver_class_name(&receiver).ok_or_else(|| {
                invalid_input("ctranspose: unsupported object receiver".to_string())
            })?;
            return crate::dispatch_object_external_member(
                class_name,
                NAME,
                vec![receiver],
                crate::current_requested_outputs(),
            )
            .await;
        }
        other => Err(invalid_input(format!(
            "ctranspose: unsupported input type {other:?}"
        ))),
    }?;
    crate::execution::RootedValue::new(result).map_err(|error| error.into_runtime_error())
}

fn ctranspose_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    ensure_vector_or_matrix_shape(tensor.shape())?;
    let rank = tensor.shape().len();
    if rank == 0 {
        return Ok(tensor);
    }
    let order = ctranspose_order(rank);
    permute_tensor(NAME, tensor, &order)
}

fn ctranspose_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    if ct.integer_storage().is_some() {
        return Err(invalid_input(
            "ctranspose: complex integer input is not supported",
        ));
    }
    ensure_vector_or_matrix_shape(ct.shape())?;
    let rank = ct.shape().len();
    let transposed = if rank == 0 {
        ct
    } else {
        let order = ctranspose_order(rank);
        permute_complex_tensor(NAME, ct, &order)?
    };
    ctranspose_complex_tensor_value(transposed)
}

fn ctranspose_complex_tensor_preserve_complex(ct: ComplexTensor) -> BuiltinResult<ComplexTensor> {
    ensure_vector_or_matrix_shape(ct.shape())?;
    let rank = ct.shape().len();
    let transposed = if rank == 0 {
        ct
    } else {
        let order = ctranspose_order(rank);
        permute_complex_tensor(NAME, ct, &order)?
    };
    let shape = transposed.shape().to_vec();
    let storage = match transposed.into_complex_storage() {
        ComplexStorage::F64(values) => {
            ComplexStorage::F64(values.into_iter().map(|(re, im)| (re, -im)).collect())
        }
        ComplexStorage::F32(values) => {
            ComplexStorage::F32(values.into_iter().map(|(re, im)| (re, -im)).collect())
        }
        ComplexStorage::Integer(_) => {
            return Err(invalid_input(
                "ctranspose: complex integer input is not supported",
            ));
        }
    };
    ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| internal_error(format!("{NAME}: {e}")))
}

fn ctranspose_complex_tensor_value(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape().to_vec();
    match ct.into_complex_storage() {
        ComplexStorage::F64(values) => {
            let conj_data = values
                .into_iter()
                .map(|(re, im)| {
                    let imag = -im;
                    (re, imag)
                })
                .collect::<Vec<_>>();
            ComplexTensor::from_complex_storage(ComplexStorage::F64(conj_data.into()), shape)
                .map(Value::ComplexTensor)
                .map_err(|e| internal_error(format!("{NAME}: {e}")))
        }
        ComplexStorage::F32(values) => {
            let conj_data = values
                .into_iter()
                .map(|(re, im)| {
                    let imag = -im;
                    (re, imag)
                })
                .collect::<Vec<_>>();
            ComplexTensor::from_complex_storage(ComplexStorage::F32(conj_data.into()), shape)
                .map(Value::ComplexTensor)
                .map_err(|e| internal_error(format!("{NAME}: {e}")))
        }
        ComplexStorage::Integer(_) => Err(invalid_input(
            "ctranspose: complex integer input is not supported",
        )),
    }
}

fn ctranspose_complex_scalar(re: f64, im: f64) -> BuiltinResult<Value> {
    Ok(Value::Complex(re, -im))
}

fn ctranspose_logical_array(la: LogicalArray) -> BuiltinResult<LogicalArray> {
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
        let order = ctranspose_order(rank);
        permute_logical_array(NAME, la, &order)
    }
}

fn ctranspose_char_array(ca: CharArray) -> BuiltinResult<CharArray> {
    let rows = ca.rows;
    let cols = ca.cols;
    if ca.data().is_empty() {
        return CharArray::from_code_units(Vec::new(), cols, rows)
            .map_err(|e| internal_error(format!("{NAME}: {e}")));
    }
    let mut out = vec![runmat_value::Utf16CodeUnit::from_raw(0); ca.data().len()];
    for r in 0..rows {
        for c in 0..cols {
            let src = r * cols + c;
            let dst = c * rows + r;
            if src < ca.data().len() && dst < out.len() {
                out[dst] = ca.data()[src];
            }
        }
    }
    CharArray::from_code_units(out, cols, rows).map_err(|e| internal_error(format!("{NAME}: {e}")))
}

fn ctranspose_string_array(sa: StringArray) -> BuiltinResult<StringArray> {
    ensure_vector_or_matrix_shape(&sa.shape)?;
    let rank = sa.shape.len();
    if rank == 0 {
        return Ok(sa);
    }
    if rank <= 2 {
        let rows = sa.rows;
        let cols = sa.cols;
        let mut out = vec![runmat_value::StringElement::default(); sa.data().len()];
        for r in 0..rows {
            for c in 0..cols {
                let src = r + c * rows;
                let dst = c + r * cols;
                if src < sa.data().len() && dst < out.len() {
                    out[dst] = sa.data()[src].clone();
                }
            }
        }
        let new_shape = if rank >= 2 {
            let mut shape = sa.shape.as_slice().to_vec();
            if shape.len() >= 2 {
                shape.swap(0, 1);
            }
            shape
        } else {
            vec![cols, rows]
        };
        StringArray::from_elements_unbounded(out, new_shape)
            .map_err(|e| internal_error(format!("{NAME}: {e}")))
    } else {
        let order = ctranspose_order(rank);
        permute_string_array(NAME, sa, &order)
    }
}

fn ctranspose_cell_array(ca: CellArray) -> BuiltinResult<CellArray> {
    let rows = ca.rows;
    let cols = ca.cols;
    let mut out = Vec::with_capacity(ca.data().len());
    for c in 0..cols {
        for r in 0..rows {
            let idx = r * cols + c;
            out.push(ca.data()[idx].clone());
        }
    }
    CellArray::new(out, cols, rows).map_err(|e| internal_error(format!("{NAME}: {e}")))
}

async fn ctranspose_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    use super::transpose::resident;
    ensure_vector_or_matrix_shape(handle.shape())?;
    let input_complex =
        runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved;
    if input_complex && runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        return Err(invalid_input(
            "ctranspose: complex integer input is not supported",
        ));
    }
    let _residency = gpu_helpers::ResidencyRestoreGuard::new(&handle);
    let provider = resident::owner(&handle, native_provider_error)?;
    let rank = handle.shape().len();
    if rank == 0 && !input_complex {
        return resident::adopt(&handle, handle.clone(), native_provider_error);
    }

    {
        let expected_shape = resident::shape(&handle);
        let transpose = if rank == 0 {
            Some(handle.clone())
        } else if rank <= 2 {
            match provider.transpose(&handle) {
                Ok(output) => Some(output),
                Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => None,
                Err(error) => return Err(native_provider_error(error)),
            }
        } else {
            let order = ctranspose_order(rank);
            let zero_based: Vec<usize> = order.iter().map(|&idx| idx - 1).collect();
            let output = provider
                .permute(&handle, &zero_based)
                .map_err(native_provider_error)?;
            if let Some(output) = output {
                if !gpu_helpers::permutation_output_matches(&handle, &output, provider, &zero_based)
                {
                    return Err(reject_native_output(
                        output,
                        "invalid provider permutation output",
                    ));
                }
                Some(output)
            } else {
                None
            }
        };
        let transposed = match transpose {
            Some(output)
                if native_ctranspose_output_matches(
                    &handle,
                    &output,
                    provider,
                    &expected_shape,
                ) =>
            {
                // Transpose may lawfully return a lazy view of the same allocation.
                Some(resident::live(output, native_provider_error)?)
            }
            Some(output) => return Err(reject_native_output(output, "malformed transpose output")),
            None => None,
        };
        if let Some(transposed) = transposed {
            if !input_complex {
                return resident::adopt(&handle, transposed, native_provider_error);
            }
            match provider.unary_conj(&transposed).await {
                Ok(Some(output))
                    if native_conjugate_output_matches(
                        &handle,
                        &transposed,
                        &output,
                        provider,
                        &expected_shape,
                    ) =>
                {
                    // The producer owns its layout. Copying the intermediate's
                    // transpose annotation onto a materialized output is invalid.
                    return resident::adopt(&handle, output, native_provider_error);
                }
                Ok(Some(output)) => {
                    return Err(reject_native_output(output, "malformed conjugate output"))
                }
                Ok(None) => {
                    // The transpose has completed. Adapt only its conjugation;
                    // never replay the original transpose or drop its lease.
                    let failure = |source: runmat_accelerate_api::ProviderExecutionError| {
                        native_provider_error(if transposed.shares_allocation_with(&handle) {
                            source.with_unproven_effects()
                        } else {
                            source.after_effect()
                        })
                    };
                    let host = gpu_helpers::download_value_preserving_residency_async(
                        provider,
                        &transposed,
                    )
                    .await
                    .map_err(|error| {
                        failure(runmat_accelerate_api::ProviderExecutionError::from_source(
                            error,
                        ))
                    })?;
                    let conjugated = crate::builtins::math::elementwise::complex_components::conj::conjugate_host_value(host)
                        .map_err(|error| failure(runmat_accelerate_api::ProviderExecutionError::from_source(error)))?;
                    return gpu_helpers::restore_class_preserving_value(&handle, conjugated, NAME)
                        .map_err(|error| {
                            failure(runmat_accelerate_api::ProviderExecutionError::from_source(
                                error,
                            ))
                        });
                }
                Err(error) => {
                    return Err(native_provider_error(
                        if transposed.shares_allocation_with(&handle) {
                            error.with_unproven_effects()
                        } else {
                            error.after_effect()
                        },
                    ))
                }
            }
            // The intermediate owns its lease throughout the await and retires
            // here (also on error or cancellation), without freeing borrowed aliases.
        }
    }

    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?;
    let transposed_value = match gathered {
        Value::Tensor(tensor) => Value::Tensor(ctranspose_tensor(tensor)?),
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
    gpu_helpers::restore_class_preserving_value(&handle, transposed_value, NAME)
}

fn native_ctranspose_output_matches(
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    expected_shape: &[usize],
) -> bool {
    super::transpose::resident::output_matches(input, output, provider, expected_shape)
}

fn native_conjugate_output_matches(
    original: &GpuTensorHandle,
    transposed: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    expected_shape: &[usize],
) -> bool {
    native_ctranspose_output_matches(transposed, output, provider, expected_shape)
        && !original.shares_allocation_with(output)
        && !transposed.shares_allocation_with(output)
}

fn reject_native_output(output: GpuTensorHandle, detail: &str) -> RuntimeError {
    let mut error = internal_error(format!("ctranspose: {detail}"));
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    gpu_helpers::reject_gpu_output(output, error)
}

fn native_provider_error(
    source: impl Into<runmat_accelerate_api::ProviderExecutionError>,
) -> RuntimeError {
    let source = source.into();
    let mut error = internal_error(format!("ctranspose: provider failed: {source}"));
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error = crate::runtime_error::with_provider_execution_source(error, source);
    error
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

#[cfg(test)]
#[path = "ctranspose_owned_tests.rs"]
mod owned_tests;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_backend;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{
        IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, StringArray, Tensor,
    };

    fn call_ctranspose(value: Value) -> BuiltinResult<crate::execution::RootedValue> {
        block_on(super::ctranspose_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![value]),
        ))
    }

    fn call_ctranspose_args(args: Vec<Value>) -> BuiltinResult<crate::execution::RootedValue> {
        block_on(super::ctranspose_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
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
    fn ctranspose_preserves_every_real_integer_class_and_rejects_every_complex_integer_class() {
        for storage in all_integer_matrix_storages() {
            let expected =
                super::super::transpose::transpose_integer_storage(storage.clone(), 2, 2);
            let class = storage.class_name();
            let real = Tensor::new_integer(storage.clone(), vec![2, 2]).expect("real tensor");
            let real_result = call_ctranspose(Value::Tensor(real)).expect("real ctranspose");
            let Value::Tensor(real_result) = real_result.value() else {
                panic!("expected real tensor for {class}");
            };
            assert_eq!(real_result.integer_storage(), Some(&expected), "{class}");

            let complex = ComplexTensor::new_integer(
                IntegerComplexStorage::new(storage.clone(), storage).expect("complex storage"),
                vec![2, 2],
            )
            .expect("complex tensor");
            let err = call_ctranspose(Value::ComplexTensor(complex))
                .expect_err("complex integer ctranspose must fail");
            assert_eq!(
                err.identifier(),
                Some("RunMat:ctranspose:InvalidInput"),
                "{class}"
            );
            assert!(err.message().contains("complex integer"), "{class}");
        }
    }

    #[test]
    fn ctranspose_preserves_typed_real_integer_storage_exactly() {
        let input = Tensor::new_integer(
            IntegerStorage::I64(vec![i64::MIN, 2, 3, 4, 5, i64::MAX]),
            vec![2, 3],
        )
        .expect("tensor");

        let result = call_ctranspose(Value::Tensor(input)).expect("ctranspose");
        let Value::Tensor(result) = result.value() else {
            panic!("expected tensor");
        };
        assert_eq!(result.shape(), [3, 2]);
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::I64(vec![i64::MIN, 3, 5, 2, 4, i64::MAX]))
        );
    }

    #[test]
    fn ctranspose_preserves_empty_typed_real_integer_storage() {
        let input =
            Tensor::new_integer(IntegerStorage::U64(Vec::new()), vec![0, 3]).expect("tensor");

        let result = call_ctranspose(Value::Tensor(input)).expect("ctranspose");
        let Value::Tensor(result) = result.value() else {
            panic!("expected tensor");
        };
        assert_eq!(result.shape(), [3, 0]);
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(Vec::new()))
        );
    }

    #[test]
    fn ctranspose_preserves_native_single_storage() {
        let input = Tensor::from_f32(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3])
            .expect("single tensor");
        let result = call_ctranspose(Value::Tensor(input)).expect("ctranspose");
        result.with_owned(|value| {
            let Value::Tensor(result) = value else {
                panic!("expected tensor");
            };
            assert_eq!(result.shape(), [3, 2]);
            assert_eq!(
                result.into_numeric_storage().unwrap(),
                runmat_value::NumericStorage::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            );
        });
    }

    fn tensor(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_real_matrix_matches_transpose() {
        let input = tensor(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]);
        let value = call_ctranspose(Value::Tensor(input)).expect("ctranspose");
        match value.value() {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [3, 2]);
                assert_eq!(out.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
        match value.value() {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape(), [2, 2]);
                assert_eq!(
                    out.materialize_f64(),
                    vec![(1.0, -2.0), (5.0, -0.0), (3.0, 4.0), (6.0, 7.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_complex_tensor_preserves_complex_when_imag_zero() {
        let data = vec![(1.0, 0.0), (2.0, -0.0)];
        let ct = ComplexTensor::new(data, vec![1, 2]).unwrap();
        let value = call_ctranspose(Value::ComplexTensor(ct)).expect("ctranspose");
        match value.value() {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape(), [2, 1]);
                assert_eq!(out.materialize_f64(), vec![(1.0, -0.0), (2.0, 0.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_complex_scalar() {
        let result = call_ctranspose(Value::Complex(2.0, 3.0)).expect("ctranspose");
        match result.value() {
            Value::Complex(re, im) => {
                assert!((*re - 2.0).abs() < 1e-12);
                assert!((*im + 3.0).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_logical_mask() {
        let la = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let value = call_ctranspose(Value::LogicalArray(la)).expect("ctranspose");
        match value.value() {
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
        let ca = CharArray::from_unicode_scalars("runmat".chars().collect(), 2, 3).unwrap();
        let value = call_ctranspose(Value::CharArray(ca)).expect("ctranspose");
        match value.value() {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 3);
                assert_eq!(out.cols, 2);
                assert_eq!(out.data(), runmat_value::encode_utf16_code_units("rmuant"));
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
        match value.value() {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(
                    out.data(),
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
        match value.value() {
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
    fn ctranspose_rejects_nd_tensor_and_directs_to_permute() {
        let data: Vec<f64> = (1..=12).map(|n| n as f64).collect();
        let input = tensor(&data, &[2, 3, 2]);
        let err = call_ctranspose(Value::Tensor(input)).expect_err("N-D ctranspose must fail");
        assert_eq!(err.identifier(), Some("RunMat:ctranspose:InvalidInput"));
        assert!(err.message().contains("use permute"));
    }

    #[test]
    fn ctranspose_accepts_explicit_trailing_singleton_dimensions() {
        let input = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2, 1]);
        let result = call_ctranspose(Value::Tensor(input)).expect("effective matrix");
        let Value::Tensor(result) = result.value() else {
            panic!("expected tensor");
        };
        assert_eq!(result.shape(), [2, 2, 1]);
        assert_eq!(result.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn ctranspose_rejects_nd_complex_logical_and_string_arrays() {
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
            let err = call_ctranspose(value).expect_err("N-D ctranspose must fail");
            assert_eq!(
                err.identifier(),
                Some("RunMat:ctranspose:InvalidInput"),
                "{label}"
            );
            assert!(err.message().contains("use permute"), "{label}");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_scalar_struct_is_identity() {
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.fields.insert("field".to_string(), Value::Num(1.0));
        let result = call_ctranspose(Value::Struct(st.finish())).unwrap();
        let Value::Struct(out) = result.value() else {
            panic!("expected struct");
        };
        assert_eq!(out.fields.get("field"), Some(&Value::Num(1.0)));
    }

    #[test]
    fn ctranspose_integer_sparse_is_independently_gated() {
        let sparse = runmat_value::SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            IntegerStorage::U64(vec![u64::MAX, 7]),
        )
        .unwrap();
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call_ctranspose(Value::SparseTensor(sparse.clone())).unwrap_err();
        assert_eq!(
            error.identifier(),
            CTRANSPOSE_INTEGER_SPARSE_EXTENSION.error_identifier
        );
        drop(strict);
        let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = call_ctranspose(Value::SparseTensor(sparse)).unwrap();
        let Value::SparseTensor(out) = result.value() else {
            panic!("expected sparse result");
        };
        assert_eq!(
            out.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 7]))
        );
        drop(extensions);
        assert_eq!(CTRANSPOSE_INTEGER_CAPABILITIES.len(), 2);
    }

    #[test]
    fn ctranspose_type_swaps_dims() {
        let out = crate::builtins::math::linalg::type_resolvers::matrix_transpose_type(
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
                data: &t.materialize_f64(),
                shape: t.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = call_ctranspose(Value::GpuTensor(handle)).expect("ctranspose");
            let gathered = result.with_owned(test_support::gather).expect("gather");
            assert_eq!(gathered.shape(), [2, 2]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 2.0, 4.0, 5.0]);
        });
    }

    #[test]
    fn ctranspose_native_result_contract_checks_shape_device_storage_precision_and_owner() {
        test_support::with_test_provider(|provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    shape: &[2, 3],
                })
                .unwrap();
            let output = provider.transpose(&input).unwrap();
            assert!(native_ctranspose_output_matches(
                &input,
                &output,
                provider,
                &[3, 2]
            ));
            let wrong_shape = output
                .with_view(output.view().clone().with_shape(vec![2, 3]))
                .unwrap();
            assert!(!native_ctranspose_output_matches(
                &input,
                &wrong_shape,
                provider,
                &[3, 2]
            ));

            let other_owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
            let other = other_owner
                .allocate_zeroed(
                    vec![3, 2],
                    runmat_accelerate_api::NumericElementType::F64,
                    GpuTensorStorage::Real,
                )
                .unwrap();
            assert!(!native_ctranspose_output_matches(
                &input,
                &other.handle,
                provider,
                &[3, 2]
            ));

            let complex = ComplexTensor::new(vec![(0.0, 0.0); 6], vec![3, 2]).unwrap();
            let wrong_storage = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
            assert!(!native_ctranspose_output_matches(
                &input,
                &wrong_storage,
                provider,
                &[3, 2]
            ));
            let single = runmat_accelerate_api::upload_numeric_protecting(
                provider,
                &runmat_accelerate_api::HostNumericTensorView {
                    data: runmat_accelerate_api::HostNumericDataView::F32(&[0.0; 6]),
                    shape: &[3, 2],
                    storage: GpuTensorStorage::Real,
                },
                &[&input, &output],
            )
            .unwrap();
            assert!(!native_ctranspose_output_matches(
                &input,
                &single,
                provider,
                &[3, 2]
            ));
            input.allocation_lease().validate_live().unwrap();
        });
    }

    #[test]
    fn ctranspose_rejects_conjugate_aliases_without_freeing_the_original() {
        test_support::with_test_provider(|provider| {
            for reject_original in [true, false] {
                let input = provider
                    .upload(&HostTensorView {
                        data: &[1.0, 2.0, 3.0, 4.0],
                        shape: &[2, 2],
                    })
                    .unwrap();
                let transposed = provider.transpose(&input).unwrap();
                let rejected = if reject_original {
                    input.clone()
                } else {
                    transposed.clone()
                };

                assert!(!native_conjugate_output_matches(
                    &input,
                    &transposed,
                    &rejected,
                    provider,
                    &[2, 2]
                ));
                let error = reject_native_output(rejected, "malformed conjugate output");
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);

                block_on(provider.download(&input)).expect("original input remains live");
                block_on(provider.download(&transposed))
                    .expect("independently owned intermediate remains live");
            }
        });
    }

    #[test]
    fn ctranspose_unowned_handle_does_not_borrow_the_ambient_provider() {
        let _guard = test_support::accel_test_lock();
        runmat_accelerate_api::clear_provider();
        let ambient = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(ambient));
        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let handle = owner
            .allocate_zeroed(
                vec![2, 2],
                runmat_accelerate_api::NumericElementType::F64,
                GpuTensorStorage::Real,
            )
            .unwrap()
            .handle;
        assert!(gpu_helpers::exact_provider_for_handle(&handle).is_none());
        call_ctranspose(Value::GpuTensor(handle))
            .expect_err("an unowned handle must not dispatch through the ambient provider");
    }

    #[test]
    fn ctranspose_gpu_preserves_every_real_integer_class() {
        test_support::with_test_provider(|provider| {
            for storage in all_integer_matrix_storages() {
                let expected =
                    super::super::transpose::transpose_integer_storage(storage.clone(), 2, 2);
                let class = storage.class_name();
                let input = Tensor::new_integer(storage, vec![2, 2]).expect("integer tensor");
                let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
                let input_type = runmat_accelerate_api::handle_integer_type(&handle);
                let result =
                    call_ctranspose(Value::GpuTensor(handle)).expect("resident ctranspose");
                let Value::GpuTensor(result_handle) = result.value() else {
                    panic!("expected resident result for {class}");
                };
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(result_handle),
                    input_type,
                    "{class}"
                );
                let gathered = result
                    .with_owned(test_support::gather)
                    .expect("integer gather");
                assert_eq!(gathered.integer_storage(), Some(&expected), "{class}");
            }
        });
    }

    #[test]
    fn ctranspose_gpu_rejects_complex_integer_metadata_before_provider_dispatch() {
        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        for shape in [vec![], vec![1, 2]] {
            let allocation = owner
                .allocate_zeroed(
                    shape,
                    runmat_accelerate_api::NumericElementType::I8,
                    GpuTensorStorage::ComplexInterleaved,
                )
                .unwrap();
            let error = call_ctranspose(Value::GpuTensor(allocation.handle.clone())).unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:ctranspose:InvalidInput"));
            assert!(error.message().contains("complex integer"));
            assert_eq!(allocation.probe.retirements(), 0);
            drop(allocation.handle);
            assert_eq!(allocation.probe.retirements(), 1);
        }
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
            let expected =
                call_ctranspose(Value::ComplexTensor(complex.clone())).expect("cpu ctranspose");
            let expected = match expected.value() {
                Value::ComplexTensor(tensor) => tensor,
                other => panic!("expected complex tensor result, got {other:?}"),
            };

            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let result = call_ctranspose(Value::GpuTensor(handle)).expect("gpu ctranspose");
            let Value::GpuTensor(out_handle) = result.value() else {
                panic!("expected complex gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(out_handle),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(
                crate::builtins::math::fft::common::gather_gpu_complex_tensor(out_handle, NAME),
            )
            .expect("gather complex");
            assert_eq!(gathered.shape(), expected.shape());
            assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
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
            let Value::GpuTensor(out_handle) = result.value() else {
                panic!("expected complex gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(out_handle),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(
                crate::builtins::math::fft::common::gather_gpu_complex_tensor(out_handle, NAME),
            )
            .expect("gather complex");
            assert_eq!(gathered.shape(), [2, 2]);
            assert_eq!(
                gathered.materialize_f64(),
                vec![(1.0, -0.0), (3.0, -0.0), (2.0, 0.0), (4.0, 0.0)]
            );
        });
    }

    #[test]
    fn ctranspose_complex_gpu_inprocess_rejects_nd_before_provider_dispatch() {
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
            let err = call_ctranspose(Value::GpuTensor(handle.clone()))
                .expect_err("N-D GPU ctranspose must fail");
            assert_eq!(err.identifier(), Some("RunMat:ctranspose:InvalidInput"));
            assert!(err.message().contains("use permute"));
            drop(handle);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    #[ignore = "requires an actual WGPU adapter; run the explicit hardware gate"]
    fn ctranspose_wgpu_matches_cpu() {
        use runmat_accelerate_api::AccelProvider;

        let _guard = test_support::accel_test_lock();
        let provider =
            wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default())
                .expect("requested WGPU ctranspose hardware gate requires an adapter");
        let data: Vec<f64> = (1..=12).map(|n| n as f64).collect();
        let tensor = Tensor::new(data, vec![3, 4]).expect("tensor");
        let cpu_value = call_ctranspose(Value::Tensor(tensor.clone())).expect("cpu");
        let cpu_tensor = match cpu_value.value() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = call_ctranspose(Value::GpuTensor(handle)).expect("gpu");
        let gathered = gpu_value.with_owned(test_support::gather).expect("gather");
        assert_eq!(gathered.shape(), cpu_tensor.shape());
        assert_eq!(gathered.materialize_f64(), cpu_tensor.materialize_f64());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    #[ignore = "requires an actual WGPU adapter; run the explicit hardware gate"]
    fn ctranspose_wgpu_complex_matches_cpu_and_stays_resident() {
        let _guard = test_support::accel_test_lock();
        let provider =
            wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default())
                .expect("requested WGPU ctranspose hardware gate requires an adapter");
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
            call_ctranspose(Value::ComplexTensor(complex.clone())).expect("cpu ctranspose");
        let expected = match expected.value() {
            Value::ComplexTensor(tensor) => tensor,
            other => panic!("expected complex tensor result, got {other:?}"),
        };

        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let result = call_ctranspose(Value::GpuTensor(handle)).expect("gpu ctranspose");
        let Value::GpuTensor(out_handle) = result.value() else {
            panic!("expected complex gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(out_handle),
            GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(
            crate::builtins::math::fft::common::gather_gpu_complex_tensor(out_handle, NAME),
        )
        .expect("gather complex");
        assert_eq!(gathered.shape(), expected.shape());
        assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    #[ignore = "requires an actual WGPU adapter; run the explicit hardware gate"]
    fn ctranspose_wgpu_complex_nd_is_rejected() {
        let _guard = test_support::accel_test_lock();
        let provider =
            wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default())
                .expect("requested WGPU ctranspose hardware gate requires an adapter");
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
        let err = call_ctranspose(Value::GpuTensor(handle.clone()))
            .expect_err("N-D GPU ctranspose must fail");
        assert_eq!(err.identifier(), Some("RunMat:ctranspose:InvalidInput"));
        assert!(err.message().contains("use permute"));
        drop(handle);
    }
}
