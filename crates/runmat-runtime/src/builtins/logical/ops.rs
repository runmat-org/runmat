//! MATLAB-compatible `logical` builtin with GPU-aware semantics for RunMat.

use log::trace;
use runmat_accelerate_api::{self, AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    BuiltinErrorDescriptor, LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE, LOGICAL_ERROR_GPU_GATHER_FAILED,
    LOGICAL_ERROR_INTERNAL, LOGICAL_ERROR_TOO_MANY_INPUTS, LOGICAL_STRING_ARRAY_EXTENSION,
    LOGICAL_SYMBOLIC_CONSTANT_EXTENSION,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::ComplexTensor;
use runmat_value::{CharArray, LogicalArray, StringArray, Tensor, Value};

use crate::builtins::common::{
    gpu_helpers,
    shape::{canonical_scalar_shape, normalize_scalar_shape},
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
    },
    tensor,
};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod preparation;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::ops")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "logical",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_ne",
        commutative: true,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Preferred path issues elem_ne(X, 0) on the device; missing hooks trigger a gather → host cast → re-upload sequence flagged as logical.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::ops")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "logical",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion support will arrive alongside a dedicated WGSL template; today the builtin executes outside fusion plans.",
};

const BUILTIN_NAME: &str = "logical";

fn logical_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "logical",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::ops"
)]
async fn logical_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(logical_error_with_message(
            runmat_builtins::LOGICAL_ERROR_TOO_MANY_OUTPUTS.message,
            &runmat_builtins::LOGICAL_ERROR_TOO_MANY_OUTPUTS,
        ));
    }
    if !rest.is_empty() {
        return Err(logical_error_with_message(
            LOGICAL_ERROR_TOO_MANY_INPUTS.message,
            &LOGICAL_ERROR_TOO_MANY_INPUTS,
        ));
    }
    match value {
        Value::Tensor(tensor) => {
            validate_tensor(&tensor)?;
            preparation::execute(tensor).await
        }
        // Specialized sparse, symbolic, text, scalar and explicit-resident
        // routes retain their existing conversion/provider admission.
        value => convert_value_to_logical(value).await,
    }
}

async fn convert_value_to_logical(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Bool(_) | Value::LogicalArray(_) => Ok(value),
        Value::Num(n) if n.is_nan() => Err(conversion_error("NaN")),
        Value::Num(n) => Ok(Value::Bool(n != 0.0)),
        Value::Int(i) => Ok(Value::Bool(!i.is_zero())),
        Value::Complex(_, _) => Err(conversion_error("complex")),
        Value::Tensor(tensor) => logical_from_tensor(tensor),
        Value::SparseTensor(sparse) => logical_from_sparse_tensor(sparse),
        Value::ComplexTensor(_) => Err(conversion_error("complex")),
        Value::CharArray(chars) => logical_from_char_array(chars),
        Value::StringArray(strings) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &LOGICAL_STRING_ARRAY_EXTENSION,
                BUILTIN_NAME,
            )?;
            logical_from_string_array(strings)
        }
        Value::GpuTensor(handle) => logical_from_gpu(handle).await,
        Value::String(_) => Err(conversion_error("string")),
        Value::Symbolic(expr) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &LOGICAL_SYMBOLIC_CONSTANT_EXTENSION,
                BUILTIN_NAME,
            )?;
            let value = expr
                .numeric_constant_value()
                .ok_or_else(|| conversion_error("sym"))?;
            if value.is_nan() {
                return Err(conversion_error("NaN"));
            }
            Ok(Value::Bool(value != 0.0))
        }
        Value::SymbolicArray(_) => Err(conversion_error("sym")),
        Value::Cell(_) => Err(conversion_error("cell")),
        Value::Struct(_) | Value::StructArray(_) => Err(conversion_error("struct")),
        Value::ObjectArray(array) => Err(conversion_error(array.class_name().display_name())),
        Value::Object(obj) => Err(conversion_error(obj.class_name.display_name())),
        Value::HandleObject(handle) => Err(conversion_error(handle.class_name.display_name())),
        Value::Listener(_) => Err(conversion_error("event.listener")),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Err(conversion_error("function_handle")),
        Value::ClassRef(_) => Err(conversion_error("meta.class")),
        Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_) => Err(conversion_error("MException")),
        Value::Foreign(_) => Err(conversion_error("foreign")),
        Value::OutputList(_) => Err(conversion_error("OutputList")),
    }
}

fn logical_from_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    validate_tensor(&tensor)?;
    logical_buffer_to_host(LogicalBuffer::from_real_tensor(&tensor))
}

fn validate_tensor(tensor: &Tensor) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none()
        && tensor.materialize_f64().iter().any(|value| value.is_nan())
    {
        return Err(conversion_error("NaN"));
    }
    Ok(())
}

fn logical_from_sparse_tensor(sparse: runmat_value::SparseTensor) -> BuiltinResult<Value> {
    if sparse.is_complex() {
        return Err(conversion_error("complex"));
    }
    if sparse.is_logical() {
        return Ok(Value::SparseTensor(sparse));
    }
    if sparse.integer_storage().is_none()
        && (0..sparse.nnz()).any(|index| {
            sparse
                .numeric_value_at(index)
                .is_some_and(|value| match value {
                    runmat_value::NumericScalar::F64(value) => value.is_nan(),
                    runmat_value::NumericScalar::F32(value) => value.is_nan(),
                    _ => false,
                })
        })
    {
        return Err(conversion_error("NaN"));
    }
    let mut col_ptrs = Vec::with_capacity(sparse.cols.saturating_add(1));
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for col in 0..sparse.cols {
        for index in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            if !sparse
                .numeric_value_at(index)
                .expect("validated sparse storage index")
                .is_zero()
            {
                row_indices.push(sparse.row_indices[index]);
            }
        }
        col_ptrs.push(row_indices.len());
    }
    runmat_value::SparseTensor::new_logical(sparse.rows, sparse.cols, col_ptrs, row_indices)
        .map(Value::SparseTensor)
        .map_err(|err| {
            logical_error_with_message(
                format!("logical: failed to convert sparse input: {err}"),
                &LOGICAL_ERROR_INTERNAL,
            )
        })
}

fn logical_from_char_array(chars: CharArray) -> BuiltinResult<Value> {
    let buffer = LogicalBuffer::from_char_array(&chars);
    logical_buffer_to_host(buffer)
}

fn logical_from_string_array(strings: StringArray) -> BuiltinResult<Value> {
    let bits: Vec<u8> = strings
        .data
        .iter()
        .map(|s| if s.is_empty() { 0 } else { 1 })
        .collect();
    let shape = canonical_shape(&strings.shape, bits.len());
    logical_buffer_to_host(LogicalBuffer { bits, shape })
}

async fn logical_from_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_is_logical(&handle) {
        return Ok(Value::GpuTensor(handle));
    }

    if runmat_accelerate_api::handle_storage(&handle)
        == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    {
        return Err(conversion_error("complex"));
    }
    let provider = gpu_helpers::exact_provider_for_handle(&handle);

    if let Some(p) = provider {
        let contains_nan = match provider_input_contains_nan(p, &handle).await {
            Ok(contains_nan) => contains_nan,
            Err(_) => {
                let host =
                    gpu_helpers::download_value_preserving_residency_async(p, &handle).await?;
                value_contains_nan(&host)
            }
        };
        if contains_nan {
            return Err(conversion_error("NaN"));
        }
        match p.logical_islogical(&handle) {
            Ok(true) => {
                runmat_accelerate_api::set_handle_logical(&handle, true);
                return Ok(Value::GpuTensor(handle));
            }
            Ok(false) => {}
            Err(err) => {
                trace!("logical: provider logical_islogical hook unavailable, falling back ({err})")
            }
        }
        if let Some(mut result) = try_gpu_cast(p, &handle).await {
            copy_logical_provenance(&mut result, &handle);
            return Ok(gpu_helpers::logical_gpu_value(result));
        } else {
            trace!(
                "logical: provider elem_ne/zeros_like unavailable for buffer {} – gathering",
                handle.buffer_id
            );
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|err| {
            logical_error_with_message(
                format!("{BUILTIN_NAME}: {err}"),
                &LOGICAL_ERROR_GPU_GATHER_FAILED,
            )
        })?;
    if value_contains_nan(&Value::Tensor(tensor.clone())) {
        return Err(conversion_error("NaN"));
    }
    let buffer = LogicalBuffer::from_real_tensor(&tensor);
    logical_buffer_to_gpu(buffer, provider, &handle)
}

async fn provider_input_contains_nan(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    source: &GpuTensorHandle,
) -> BuiltinResult<bool> {
    let mask = provider
        .logical_isnan(source)
        .map_err(|error| logical_error_with_message(error.to_string(), &LOGICAL_ERROR_INTERNAL))?;
    let mask_valid = !gpu_helpers::same_gpu_handle(&mask, source)
        && mask.shape == source.shape
        && mask.device_id == source.device_id
        && gpu_helpers::exact_provider_for_handle(&mask)
            .is_some_and(|owner| std::ptr::eq(owner, provider));
    if !mask_valid {
        return Err(logical_error_with_message(
            "logical: provider returned a malformed NaN mask",
            &LOGICAL_ERROR_INTERNAL,
        ));
    }
    let maximum = provider
        .reduce_max(&mask)
        .await
        .map_err(|error| logical_error_with_message(error.to_string(), &LOGICAL_ERROR_INTERNAL))?;
    let maximum_valid = !gpu_helpers::same_gpu_handle(&maximum, source)
        && !gpu_helpers::same_gpu_handle(&maximum, &mask)
        && maximum.shape.iter().product::<usize>() == 1
        && maximum.device_id == source.device_id
        && gpu_helpers::exact_provider_for_handle(&maximum)
            .is_some_and(|owner| std::ptr::eq(owner, provider));
    if !maximum_valid {
        return Err(logical_error_with_message(
            "logical: provider returned a malformed NaN reduction",
            &LOGICAL_ERROR_INTERNAL,
        ));
    }
    let downloaded = provider
        .download(&maximum)
        .await
        .map_err(|error| logical_error_with_message(error.to_string(), &LOGICAL_ERROR_INTERNAL))?;
    Ok(downloaded.data.first().is_some_and(|value| *value != 0.0))
}

fn value_contains_nan(value: &Value) -> bool {
    match value {
        Value::Num(value) => value.is_nan(),
        Value::Tensor(tensor) => {
            tensor.integer_storage().is_none()
                && tensor.materialize_f64().iter().any(|value| value.is_nan())
        }
        _ => false,
    }
}

fn logical_buffer_to_host(buffer: LogicalBuffer) -> BuiltinResult<Value> {
    let LogicalBuffer { bits, shape } = buffer;
    if tensor::element_count(&shape) == 1 && bits.len() == 1 {
        Ok(Value::Bool(bits[0] != 0))
    } else {
        LogicalArray::new(bits, shape)
            .map(Value::LogicalArray)
            .map_err(|e| {
                logical_error_with_message(format!("logical: {e}"), &LOGICAL_ERROR_INTERNAL)
            })
    }
}

fn logical_buffer_to_gpu(
    buffer: LogicalBuffer,
    provider: Option<&'static dyn AccelProvider>,
    source: &GpuTensorHandle,
) -> BuiltinResult<Value> {
    let explicit = runmat_accelerate_api::handle_is_explicit(source);
    if let Some(p) = provider {
        let floats: Vec<f64> = buffer
            .bits
            .iter()
            .map(|&b| if b != 0 { 1.0 } else { 0.0 })
            .collect();
        let view = HostTensorView {
            data: &floats,
            shape: &buffer.shape,
        };
        match p.upload(&view) {
            Ok(mut handle) => {
                if !valid_logical_gpu_output(&handle, source, p) {
                    if explicit {
                        return Err(logical_error_with_message(
                            "logical: provider returned an invalid restored logical output",
                            &LOGICAL_ERROR_INTERNAL,
                        ));
                    }
                    return logical_buffer_to_host(buffer);
                }
                if explicit {
                    runmat_accelerate_api::mark_handle_explicit(&mut handle);
                } else {
                    runmat_accelerate_api::mark_handle_automatic(&mut handle);
                }
                Ok(gpu_helpers::logical_gpu_value(handle))
            }
            Err(err) => {
                trace!("logical: upload failed during fallback path ({err})");
                if explicit {
                    Err(logical_error_with_message(
                        format!("logical: failed to preserve explicit gpuArray residency: {err}"),
                        &LOGICAL_ERROR_INTERNAL,
                    ))
                } else {
                    logical_buffer_to_host(buffer)
                }
            }
        }
    } else if explicit {
        Err(logical_error_with_message(
            "logical: no exact owner for explicit gpuArray input",
            &LOGICAL_ERROR_GPU_GATHER_FAILED,
        ))
    } else {
        logical_buffer_to_host(buffer)
    }
}

async fn try_gpu_cast(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
) -> Option<GpuTensorHandle> {
    let zeros = provider.zeros_like(input).ok()?;
    let zeros_valid = zeros.shape == input.shape
        && zeros.device_id == input.device_id
        && !gpu_helpers::same_gpu_handle(&zeros, input)
        && runmat_accelerate_api::handle_storage(&zeros)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(&zeros)
            == runmat_accelerate_api::handle_precision(input)
        && runmat_accelerate_api::handle_integer_type(&zeros)
            == runmat_accelerate_api::handle_integer_type(input)
        && runmat_accelerate_api::handle_is_logical(&zeros)
            == runmat_accelerate_api::handle_is_logical(input)
        && gpu_helpers::exact_provider_for_handle(&zeros)
            .is_some_and(|owner| std::ptr::eq(owner, provider));
    if !zeros_valid {
        return None;
    }
    let result = provider
        .elem_ne(input, &zeros)
        .await
        .ok()
        .and_then(|output| {
            if valid_logical_gpu_output(&output, input, provider)
                && !gpu_helpers::same_gpu_handle(&output, &zeros)
            {
                Some(output)
            } else {
                None
            }
        });
    drop(zeros);
    result
}

fn copy_logical_provenance(output: &mut GpuTensorHandle, input: &GpuTensorHandle) {
    runmat_accelerate_api::set_handle_provenance(
        output,
        runmat_accelerate_api::handle_provenance(input)
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic),
    );
}

fn valid_logical_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> bool {
    output.shape == input.shape
        && output.device_id == input.device_id
        && !gpu_helpers::same_gpu_handle(output, input)
        && runmat_accelerate_api::handle_storage(output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn conversion_error(type_name: &str) -> RuntimeError {
    logical_error_with_message(
        format!(
            "logical: conversion to logical from {} is not possible",
            type_name
        ),
        &LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE,
    )
}

#[derive(Clone)]
struct LogicalBuffer {
    bits: Vec<u8>,
    shape: Vec<usize>,
}

impl LogicalBuffer {
    fn from_real_tensor(tensor: &Tensor) -> Self {
        let bits: Vec<u8> = (0..tensor.len())
            .map(|index| {
                u8::from(
                    !tensor
                        .numeric_value_at(index)
                        .expect("tensor storage is structurally valid")
                        .is_zero(),
                )
            })
            .collect();
        let shape = canonical_shape(&tensor.shape, bits.len());
        Self { bits, shape }
    }

    fn from_char_array(chars: &CharArray) -> Self {
        let bits: Vec<u8> = chars
            .to_column_major()
            .iter()
            .map(|&ch| if (ch as u32) != 0 { 1 } else { 0 })
            .collect();
        let shape = canonical_shape(&chars.shape, bits.len());
        Self { bits, shape }
    }
}

fn canonical_shape(shape: &[usize], len: usize) -> Vec<usize> {
    if tensor::element_count(shape) == len {
        return normalize_scalar_shape(shape);
    }
    if len == 0 {
        if shape.len() > 1 {
            return shape.to_vec();
        }
        return vec![0];
    }
    if len == 1 {
        canonical_scalar_shape()
    } else {
        vec![len, 1]
    }
}

#[cfg(test)]
pub(crate) mod tests;
