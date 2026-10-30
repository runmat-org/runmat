//! MATLAB-compatible `logical` builtin with GPU-aware semantics for RunMat.

use log::trace;
use runmat_accelerate_api::{self, AccelProvider, GpuTensorHandle};
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

mod nan_probe;
mod preparation;

use nan_probe::provider_input_contains_nan;

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
    context = true,
    builtin_path = "crate::builtins::logical::ops"
)]
async fn logical_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
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
            preparation::execute(runtime.as_ref(), tensor).await
        }
        // Specialized sparse, symbolic, text, scalar and explicit-resident
        // routes retain their existing conversion/provider admission.
        value => {
            // Every accepted non-prepared representation below publishes a
            // scalar, logical array, sparse numeric, or GPU tensor handle.
            // None can contain a managed GC child.
            let output = convert_value_to_logical(value).await?;
            crate::execution::RootedValue::new(output).map_err(|error| error.into_runtime_error())
        }
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
        Value::Object(obj) => Err(conversion_error(obj.class_name().display_name())),
        Value::HandleObject(handle) => Err(conversion_error(handle.class_name.display_name())),
        Value::Listener(_) => Err(conversion_error("event.listener")),
        Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
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
        .data()
        .iter()
        .map(|s| {
            s.as_str().map(|s| u8::from(!s.is_empty())).ok_or_else(|| {
                logical_error_with_message(
                    "logical: cannot convert missing strings",
                    &LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE,
                )
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
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
    let provider = Some(gpu_helpers::require_resident_compute_owner(
        &handle,
        || {
            logical_error_with_message(
                "logical: GPU input has no owning provider",
                &LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE,
            )
        },
    )?);

    if let Some(p) = provider {
        let contains_nan = match provider_input_contains_nan(p, &handle).await? {
            Some(contains_nan) => contains_nan,
            None => {
                let host =
                    gpu_helpers::download_value_preserving_residency_async(p, &handle).await?;
                value_contains_nan(&host)
            }
        };
        if contains_nan {
            return Err(conversion_error("NaN"));
        }
        if let Some(mut result) = try_gpu_cast(p, &handle).await? {
            copy_logical_provenance(&mut result, &handle)?;
            return gpu_helpers::logical_gpu_value(result);
        } else {
            trace!(
                "logical: provider elem_ne/zeros_like unavailable for buffer {} – gathering",
                handle.buffer_id()
            );
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    if value_contains_nan(&Value::Tensor(tensor.clone())) {
        return Err(conversion_error("NaN"));
    }
    let buffer = LogicalBuffer::from_real_tensor(&tensor);
    logical_buffer_to_gpu(buffer, provider, &handle)
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
    let Some(provider) = provider else {
        return if explicit {
            Err(logical_error_with_message(
                "logical: no exact owner for explicit gpuArray input",
                &LOGICAL_ERROR_GPU_GATHER_FAILED,
            ))
        } else {
            logical_buffer_to_host(buffer)
        };
    };
    let output = runmat_accelerate_api::upload_logical_protecting(
        provider,
        &runmat_accelerate_api::HostLogicalTensorView {
            data: &buffer.bits,
            shape: &buffer.shape,
        },
        &[source],
    );
    match output {
        Ok(mut output) => {
            if !valid_logical_gpu_output(&output, source, provider) {
                return Err(gpu_helpers::reject_gpu_output(
                    output,
                    logical_error_with_message(
                        "logical: provider returned an invalid restored logical output",
                        &LOGICAL_ERROR_INTERNAL,
                    ),
                ));
            }
            copy_logical_provenance(&mut output, source)?;
            gpu_helpers::logical_gpu_value(output)
        }
        Err(error) => {
            let error = runmat_accelerate_api::ProviderExecutionError::from_source(error);
            if !explicit && error.is_unsupported() {
                logical_buffer_to_host(buffer)
            } else {
                Err(logical_provider_error(error))
            }
        }
    }
}

async fn try_gpu_cast(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    if runmat_accelerate_api::handle_integer_type(input).is_some()
        || runmat_accelerate_api::handle_precision(input) != Some(provider.precision())
    {
        return Ok(None);
    }
    let zeros = match provider.zeros_like(input, input.shape()) {
        Ok(Some(zeros)) => zeros,
        Ok(None) => return Ok(None),
        Err(error) => return Err(logical_provider_error(error.with_unproven_effects())),
    };
    if zeros.shape() != input.shape()
        || zeros.device_id() != input.device_id()
        || zeros.shares_allocation_with(input)
        || zeros.physical().storage() != input.physical().storage()
        || zeros.physical().element_type() != input.physical().element_type()
        || zeros.view().is_logical() != input.view().is_logical()
        || !gpu_helpers::exact_provider_for_handle(&zeros)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
    {
        return Err(gpu_helpers::reject_gpu_output(
            zeros,
            logical_error_with_message(
                "logical: provider returned an invalid zero operand",
                &LOGICAL_ERROR_INTERNAL,
            ),
        ));
    }
    let output = match provider.elem_ne(input, &zeros).await {
        Ok(output) => output,
        // The zero operand was already materialized. A later Unsupported is
        // terminal for this route; it cannot authorize a host replay.
        Err(error) => return Err(logical_provider_error(error.after_effect())),
    };
    if !valid_logical_gpu_output(&output, input, provider) || output.shares_allocation_with(&zeros)
    {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            logical_error_with_message(
                "logical: provider returned an invalid logical conversion",
                &LOGICAL_ERROR_INTERNAL,
            ),
        ));
    }
    Ok(Some(output))
}

fn copy_logical_provenance(
    output: &mut GpuTensorHandle,
    input: &GpuTensorHandle,
) -> BuiltinResult<()> {
    gpu_helpers::propagate_output_provenance(output, [input])
}

fn logical_provider_error(error: runmat_accelerate_api::ProviderExecutionError) -> RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        logical_error_with_message(
            format!("logical: provider execution failed: {error}"),
            &LOGICAL_ERROR_INTERNAL,
        ),
        error,
    )
}

fn valid_logical_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> bool {
    output.view().is_logical()
        && output.shape() == input.shape()
        && output.device_id() == input.device_id()
        && !gpu_helpers::same_gpu_handle(output, input)
        && runmat_accelerate_api::handle_storage(output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
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
        let shape = canonical_shape(tensor.shape(), bits.len());
        Self { bits, shape }
    }

    fn from_char_array(chars: &CharArray) -> Self {
        let bits: Vec<u8> = chars
            .to_column_major()
            .iter()
            .map(|&ch| if u32::from(ch.raw()) != 0 { 1 } else { 0 })
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

#[cfg(test)]
mod owning_tests;
