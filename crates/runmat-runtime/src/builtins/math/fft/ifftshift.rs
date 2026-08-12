//! MATLAB-compatible `ifftshift` builtin with GPU-aware semantics for RunMat.
//!
//! `ifftshift` moves the zero-frequency component back to the origin, undoing
//! the reordering performed by `fftshift` and preparing spectra for inverse FFTs.

use super::common::{
    apply_shift, apply_shift_numeric_storage, build_shift_plan, compute_shift_dims,
    free_rejected_provider_fft_output, gpu_metadata_snapshot, restore_gpu_metadata,
    same_gpu_handle, ShiftKind,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::fft::type_resolvers::ifftshift_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexStorage, ComplexTensor, LogicalArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::ifftshift")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ifftshift",
    op_kind: GpuOpKind::Custom("ifftshift"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("circshift")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Delegates to provider circshift kernels; falls back to host when the hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::ifftshift")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ifftshift",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Explicit data movement; not fused with surrounding elementwise graphs.",
};

const BUILTIN_NAME: &str = "ifftshift";

const IFFTSHIFT_MULTI_DIM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifftshift-multi-dimension-selector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifftshift numeric dimension vectors and logical masks are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IfftshiftMultiDimensionExtension"),
};
pub const IFFTSHIFT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [IFFTSHIFT_MULTI_DIM_EXTENSION];

const IFFTSHIFT_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All integer classes are reordered exactly with class, shape, and complexity preserved.",
}];
const IFFTSHIFT_DIM_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "DIM",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The documented DIM form is one positive scalar parsed from authoritative storage.",
}];
const IFFTSHIFT_MULTI_DIM_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "DIMS",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes:
            "RunMat-only numeric vectors are decoded exactly; logical masks share the same gate.",
    }];
pub const IFFTSHIFT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifftshift(integer_X)", inputs: &IFFTSHIFT_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "Real and typed-complex integer storage is reordered without an f64 mirror; resident fallback restores to the owner." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifftshift(X, integer_DIM)", inputs: &IFFTSHIFT_DIM_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "All eight documented integer scalar classes select one dimension exactly." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifftshift(X, integer_DIMS)", inputs: &IFFTSHIFT_MULTI_DIM_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat-only multi-axis vector/mask selection is independently gated." },
];

const IFFTSHIFT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Shifted array with the same size and type family as X.",
}];

const IFFTSHIFT_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, complex, logical, or gpuArray input.",
}];

const IFFTSHIFT_INPUTS_DIMS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, complex, logical, or gpuArray input.",
    },
    BuiltinParamDescriptor {
        name: "DIM",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Dimension selector (scalar, numeric vector, or logical mask vector).",
    },
];

const IFFTSHIFT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = ifftshift(X)",
        inputs: &IFFTSHIFT_INPUTS_CORE,
        outputs: &IFFTSHIFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifftshift(X, DIM)",
        inputs: &IFFTSHIFT_INPUTS_DIMS,
        outputs: &IFFTSHIFT_OUTPUT,
    },
];

const IFFTSHIFT_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTSHIFT.ARG_COUNT",
    identifier: Some("RunMat:ifftshift:ArgCount"),
    when: "More than two input arguments are supplied.",
    message: "ifftshift: invalid argument count",
};

const IFFTSHIFT_ERROR_INVALID_DIMS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTSHIFT.INVALID_DIMS",
    identifier: Some("RunMat:ifftshift:InvalidDimensions"),
    when: "DIM argument is malformed or out of range.",
    message: "ifftshift: invalid dimension argument",
};

const IFFTSHIFT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTSHIFT.INVALID_INPUT",
    identifier: Some("RunMat:ifftshift:InvalidInput"),
    when: "X is not a supported numeric/logical input type.",
    message: "ifftshift: expected numeric or logical input",
};

const IFFTSHIFT_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTSHIFT.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:ifftshift:UnsupportedInput"),
    when: "X is an unsupported object/cell/function/meta runtime type.",
    message: "ifftshift: unsupported input type",
};

const IFFTSHIFT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTSHIFT.INTERNAL",
    identifier: Some("RunMat:ifftshift:Internal"),
    when: "Shifting, tensor reconstruction, or GPU transfer operations fail.",
    message: "ifftshift: internal error",
};

const IFFTSHIFT_ERRORS: [BuiltinErrorDescriptor; 5] = [
    IFFTSHIFT_ERROR_ARG_COUNT,
    IFFTSHIFT_ERROR_INVALID_DIMS,
    IFFTSHIFT_ERROR_INVALID_INPUT,
    IFFTSHIFT_ERROR_UNSUPPORTED_INPUT,
    IFFTSHIFT_ERROR_INTERNAL,
];

pub const IFFTSHIFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IFFTSHIFT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IFFTSHIFT_ERRORS,
};

fn ifftshift_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    ifftshift_error_with_message(error.message, error)
}

fn ifftshift_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    ifftshift_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn ifftshift_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ifftshift_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ifftshift_terminal_error(detail: impl AsRef<str>) -> RuntimeError {
    build_runtime_error(format!(
        "ifftshift: provider integrity error: {}",
        detail.as_ref()
    ))
    .with_builtin(BUILTIN_NAME)
    .with_identifier("RunMat:ifftshift:ProviderIntegrity")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}

fn compute_ifftshift_dims(shape: &[usize], dims_arg: Option<&Value>) -> BuiltinResult<Vec<usize>> {
    compute_shift_dims(shape, dims_arg, BUILTIN_NAME).map_err(|source| {
        ifftshift_error_with_source(
            &IFFTSHIFT_ERROR_INVALID_DIMS,
            "dimension parsing failed",
            source,
        )
    })
}

#[runtime_builtin(
    name = "ifftshift",
    category = "math/fft",
    summary = "Shift centered spectra back to origin ordering.",
    keywords = "ifftshift,inverse fft shift,frequency alignment,gpu",
    accel = "custom",
    type_resolver(ifftshift_type),
    descriptor(crate::builtins::math::fft::ifftshift::IFFTSHIFT_DESCRIPTOR),
    extensions(crate::builtins::math::fft::ifftshift::IFFTSHIFT_EXTENSIONS),
    integer_capabilities(crate::builtins::math::fft::ifftshift::IFFTSHIFT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::fft::ifftshift"
)]
async fn ifftshift_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(ifftshift_error(&IFFTSHIFT_ERROR_ARG_COUNT));
    }
    let dims_arg = rest.first();
    if dims_arg.is_some_and(ifftshift_uses_multi_dimension_extension) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IFFTSHIFT_MULTI_DIM_EXTENSION,
            BUILTIN_NAME,
        )?;
    }

    match value {
        Value::Tensor(tensor) => {
            let dims = compute_ifftshift_dims(&tensor.shape, dims_arg)?;
            Ok(ifftshift_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::ComplexTensor(ct) => {
            let dims = compute_ifftshift_dims(&ct.shape, dims_arg)?;
            Ok(ifftshift_complex_tensor(ct, &dims).map(Value::ComplexTensor)?)
        }
        Value::LogicalArray(array) => {
            let dims = compute_ifftshift_dims(&array.shape, dims_arg)?;
            Ok(ifftshift_logical(array, &dims).map(Value::LogicalArray)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|source| {
                ifftshift_error_with_detail(
                    &IFFTSHIFT_ERROR_INTERNAL,
                    format!("complex tensor construction failed: {source}"),
                )
            })?;
            let dims = compute_ifftshift_dims(&tensor.shape, dims_arg)?;
            Ok(ifftshift_complex_tensor(tensor, &dims).map(|result| {
                if result.materialize_f64().len() == 1 {
                    let (r, i) = result.materialize_f64()[0];
                    Value::Complex(r, i)
                } else {
                    Value::ComplexTensor(result)
                }
            })?)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value).map_err(|detail| {
                ifftshift_error_with_detail(
                    &IFFTSHIFT_ERROR_INVALID_INPUT,
                    format!("scalar/tensor conversion failed: {detail}"),
                )
            })?;
            let dims = compute_ifftshift_dims(&tensor.shape, dims_arg)?;
            Ok(ifftshift_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::GpuTensor(handle) => {
            let dims = compute_ifftshift_dims(&handle.shape, dims_arg)?;
            Ok(ifftshift_gpu(handle, &dims).await?)
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => {
            Err(ifftshift_error(&IFFTSHIFT_ERROR_INVALID_INPUT))
        }
        Value::Symbolic(_) | Value::SymbolicArray(_) => {
            Err(ifftshift_error(&IFFTSHIFT_ERROR_INVALID_INPUT))
        }
        Value::Struct(_)
        | Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::SparseTensor(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::OutputList(_) => Err(ifftshift_error(&IFFTSHIFT_ERROR_UNSUPPORTED_INPUT)),
    }
}

fn ifftshift_uses_multi_dimension_extension(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.len() != 1,
        Value::LogicalArray(array) => array.data.len() != 1,
        _ => false,
    }
}

fn ifftshift_tensor(tensor: Tensor, dims: &[usize]) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor.into_numeric_storage().map_err(|source| {
        ifftshift_error_with_detail(
            &IFFTSHIFT_ERROR_INTERNAL,
            format!("invalid tensor storage: {source}"),
        )
    })?;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Ifft);
    let rotated =
        apply_shift_numeric_storage(BUILTIN_NAME, storage, &plan.ext_shape, &plan.positive)?;
    Tensor::from_numeric_storage(rotated, shape).map_err(|source| {
        ifftshift_error_with_detail(
            &IFFTSHIFT_ERROR_INTERNAL,
            format!("tensor reconstruction failed: {source}"),
        )
    })
}

fn ifftshift_complex_tensor(tensor: ComplexTensor, dims: &[usize]) -> BuiltinResult<ComplexTensor> {
    let shape = tensor.shape.clone();
    let storage = tensor.into_complex_storage();
    let plan = build_shift_plan(&shape, dims, ShiftKind::Ifft);
    if storage.is_empty() || plan.is_noop() {
        return ComplexTensor::from_complex_storage(storage, shape).map_err(|source| {
            ifftshift_error_with_detail(
                &IFFTSHIFT_ERROR_INTERNAL,
                format!("complex tensor reconstruction failed: {source}"),
            )
        });
    }
    let rotated = match storage {
        ComplexStorage::F64(values) => ComplexStorage::F64(apply_shift(
            BUILTIN_NAME,
            &values,
            &plan.ext_shape,
            &plan.positive,
        )?),
        ComplexStorage::F32(values) => ComplexStorage::F32(apply_shift(
            BUILTIN_NAME,
            &values,
            &plan.ext_shape,
            &plan.positive,
        )?),
        ComplexStorage::Integer(storage) => ComplexStorage::Integer(
            storage
                .reorder(|values| {
                    apply_shift(BUILTIN_NAME, values, &plan.ext_shape, &plan.positive)
                        .map_err(|error| error.to_string())
                })
                .map_err(|source| {
                    ifftshift_error_with_detail(
                        &IFFTSHIFT_ERROR_INTERNAL,
                        format!("complex integer shift failed: {source}"),
                    )
                })?,
        ),
    };
    ComplexTensor::from_complex_storage(rotated, shape).map_err(|source| {
        ifftshift_error_with_detail(
            &IFFTSHIFT_ERROR_INTERNAL,
            format!("complex tensor reconstruction failed: {source}"),
        )
    })
}

fn ifftshift_logical(array: LogicalArray, dims: &[usize]) -> BuiltinResult<LogicalArray> {
    let LogicalArray { data, shape } = array;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Ifft);
    if data.is_empty() || plan.is_noop() {
        return LogicalArray::new(data, shape).map_err(|source| {
            ifftshift_error_with_detail(
                &IFFTSHIFT_ERROR_INTERNAL,
                format!("logical array reconstruction failed: {source}"),
            )
        });
    }
    let rotated = apply_shift(BUILTIN_NAME, &data, &plan.ext_shape, &plan.positive)?;
    LogicalArray::new(rotated, shape).map_err(|source| {
        ifftshift_error_with_detail(
            &IFFTSHIFT_ERROR_INTERNAL,
            format!("logical array reconstruction failed: {source}"),
        )
    })
}

async fn ifftshift_gpu(handle: GpuTensorHandle, dims: &[usize]) -> BuiltinResult<Value> {
    let plan = build_shift_plan(&handle.shape, dims, ShiftKind::Ifft);
    if plan.is_noop() {
        return Ok(Value::GpuTensor(handle));
    }

    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        ifftshift_terminal_error("no acceleration provider owns the input handle")
    })?;
    // Reshape hooks may mutate provider registry metadata in place. Synthetic
    // trailing dimensions therefore use the exact owner-preserving host path.
    if plan.ext_shape == handle.shape {
        let input_metadata = gpu_metadata_snapshot(&handle);
        match provider.circshift(&handle, &plan.provider) {
            Ok(out) => {
                if same_gpu_handle(&out, &handle) {
                    restore_gpu_metadata(&handle, input_metadata);
                    return Err(ifftshift_terminal_error(
                        "provider circshift aliased its input",
                    ));
                }
                if !valid_ifftshift_gpu_output(provider, &out, &handle, input_metadata) {
                    free_rejected_provider_fft_output(provider, &out, &[&handle]);
                    return Err(ifftshift_terminal_error(
                        "provider circshift returned malformed metadata",
                    ));
                }
                return Ok(Value::GpuTensor(out));
            }
            Err(error) if ifftshift_provider_unsupported(&error, "circshift") => {}
            Err(error) => {
                return Err(ifftshift_terminal_error(format!(
                    "provider circshift failed: {error}"
                )));
            }
        }
    }

    ifftshift_gpu_fallback(handle, dims).await
}

async fn ifftshift_gpu_fallback(handle: GpuTensorHandle, dims: &[usize]) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        ifftshift_terminal_error("no acceleration provider owns the input handle")
    })?;
    let host = gpu_helpers::download_value_preserving_residency_async(provider, &handle)
        .await
        .map_err(|error| {
            ifftshift_terminal_error(format!("owner-preserving download failed: {error}"))
        })?;
    let shifted = match host {
        Value::Tensor(tensor) => Value::Tensor(ifftshift_tensor(tensor, dims)?),
        Value::ComplexTensor(tensor) => {
            Value::ComplexTensor(ifftshift_complex_tensor(tensor, dims)?)
        }
        Value::LogicalArray(array) => Value::LogicalArray(ifftshift_logical(array, dims)?),
        other => {
            return Err(ifftshift_terminal_error(format!(
                "unexpected downloaded value {other:?}"
            )));
        }
    };
    restore_ifftshift_gpu_value(provider, &handle, shifted)
}

fn ifftshift_provider_unsupported(error: &anyhow::Error, operation: &str) -> bool {
    let expected = format!("{operation} not supported by provider");
    error.chain().any(|cause| cause.to_string() == expected)
}

fn valid_ifftshift_gpu_output(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    expected: super::common::GpuMetadataSnapshot,
) -> bool {
    output.shape == input.shape
        && output.device_id == input.device_id
        && !same_gpu_handle(output, input)
        && gpu_metadata_snapshot(output) == expected
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn restore_ifftshift_gpu_value(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    value: Value,
) -> BuiltinResult<Value> {
    let input_metadata = gpu_metadata_snapshot(input);
    let output = match &value {
        Value::Tensor(tensor) => {
            if tensor.integer_storage().is_none() {
                let required = match tensor.numeric_dtype() {
                    runmat_builtins::NumericDType::F32 => {
                        runmat_accelerate_api::ProviderPrecision::F32
                    }
                    _ => runmat_accelerate_api::ProviderPrecision::F64,
                };
                if provider.precision() != required {
                    return Ok(value);
                }
            }
            gpu_helpers::upload_tensor(provider, tensor).map_err(|error| {
                ifftshift_terminal_error(format!("failed to restore owner result: {error}"))
            })?
        }
        Value::ComplexTensor(tensor) => {
            let required = match tensor.numeric_dtype() {
                runmat_builtins::NumericDType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
                _ => runmat_accelerate_api::ProviderPrecision::F64,
            };
            if provider.precision() != required {
                return Ok(value);
            }
            gpu_helpers::upload_complex_tensor(provider, tensor).map_err(|error| {
                ifftshift_terminal_error(format!("failed to restore owner result: {error}"))
            })?
        }
        Value::LogicalArray(array) => {
            let tensor = tensor::logical_to_tensor(array).map_err(|error| {
                ifftshift_terminal_error(format!("logical restoration failed: {error}"))
            })?;
            let output = gpu_helpers::upload_tensor(provider, &tensor).map_err(|error| {
                ifftshift_terminal_error(format!("failed to restore owner result: {error}"))
            })?;
            runmat_accelerate_api::set_handle_logical(&output, true);
            output
        }
        _ => {
            return Err(ifftshift_terminal_error(
                "unexpected host result during restoration",
            ));
        }
    };
    if same_gpu_handle(&output, input) {
        restore_gpu_metadata(input, input_metadata);
        return Err(ifftshift_terminal_error(
            "provider upload aliased the protected input",
        ));
    }
    if !valid_ifftshift_gpu_output(provider, &output, input, input_metadata) {
        free_rejected_provider_fft_output(provider, &output, &[input]);
        return Err(ifftshift_terminal_error(
            "provider upload returned malformed metadata",
        ));
    }
    Ok(Value::GpuTensor(output))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::common::{apply_shift, build_shift_plan, ShiftKind};
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{
        builtin_function_by_name, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage,
        LogicalArray, ResolveContext, Tensor, Type,
    };

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    #[test]
    fn ifftshift_type_preserves_tensor_shape() {
        let out = ifftshift_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(5)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(5)])
            }
        );
    }

    #[test]
    fn ifftshift_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("ifftshift builtin");
        let descriptor = builtin.descriptor.expect("ifftshift descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = ifftshift(X)"));
        assert!(labels.contains(&"Y = ifftshift(X, DIM)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.IFFTSHIFT.INVALID_DIMS"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_even_length_vector() {
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
        let result = ifftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![8, 1]);
                assert_eq!(
                    out.materialize_f64(),
                    vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_odd_length_vector() {
        let tensor = Tensor::new((1..=5).map(|v| v as f64).collect(), vec![5, 1]).unwrap();
        let result = ifftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![5, 1]);
                assert_eq!(out.materialize_f64(), vec![3.0, 4.0, 5.0, 1.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_inverts_fftshift() {
        let tensor = Tensor::new((0..7).map(|v| v as f64).collect(), vec![7, 1]).unwrap();
        let fft_plan = build_shift_plan(&tensor.shape, &[0], ShiftKind::Fft);
        let fft_data = apply_shift(
            "ifftshift",
            &tensor.materialize_f64(),
            &fft_plan.ext_shape,
            &fft_plan.positive,
        )
        .expect("apply_shift");
        let fft_tensor = Tensor::new(fft_data, tensor.shape.clone()).unwrap();

        let restored =
            ifftshift_builtin(Value::Tensor(fft_tensor), Vec::new()).expect("ifftshift restore");
        match restored {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![7, 1]);
                assert_eq!(out.materialize_f64(), tensor.materialize_f64());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_dimension_subset() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let dims = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let result =
            ifftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.materialize_f64(), vec![2.0, 5.0, 3.0, 6.0, 1.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_logical_mask_dimensions() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let mask = LogicalArray::new(vec![0, 1], vec![2, 1]).unwrap();
        let result = ifftshift_builtin(Value::Tensor(tensor), vec![Value::LogicalArray(mask)])
            .expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.materialize_f64(), vec![2.0, 5.0, 3.0, 6.0, 1.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_empty_dimension_vector_noop() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let empty_dims = Tensor::new(vec![], vec![0, 1]).unwrap();
        let result = ifftshift_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Tensor(empty_dims)],
        )
        .expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, tensor.shape);
                assert_eq!(out.materialize_f64(), tensor.materialize_f64());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_dimension_zero_error() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = ifftshift_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))])
            .unwrap_err();
        assert_eq!(
            error_identifier(&err),
            IFFTSHIFT_ERROR_INVALID_DIMS.identifier
        );
        assert!(
            error_message(err).contains(IFFTSHIFT_ERROR_INVALID_DIMS.message),
            "unexpected error"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_logical_array_supported() {
        let logical = LogicalArray::new(vec![1, 0, 0, 0], vec![4, 1]).unwrap();
        let result =
            ifftshift_builtin(Value::LogicalArray(logical), Vec::new()).expect("ifftshift");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![4, 1]);
                assert_eq!(out.data, vec![0, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_rejects_non_numeric_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err =
            ifftshift_builtin(Value::Tensor(tensor), vec![Value::from("invalid")]).unwrap_err();
        assert_eq!(
            error_identifier(&err),
            IFFTSHIFT_ERROR_INVALID_DIMS.identifier
        );
        assert!(
            error_message(err).contains(IFFTSHIFT_ERROR_INVALID_DIMS.message),
            "unexpected error"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                ifftshift_builtin(Value::GpuTensor(handle), Vec::new()).expect("ifftshift");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![8, 1]);
            assert_eq!(
                gathered.materialize_f64(),
                vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_complex_tensor() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
            vec![4, 1],
        )
        .unwrap();
        let result = ifftshift_builtin(Value::ComplexTensor(tensor), Vec::new()).unwrap();
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![4, 1]);
                assert_eq!(
                    out.materialize_f64(),
                    vec![(2.0, 2.0), (3.0, 3.0), (0.0, 0.0), (1.0, 1.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn ifftshift_preserves_typed_complex_integer_storage_at_i64_boundaries() {
        let tensor = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I64(vec![i64::MIN, 2, 3, 4, i64::MAX]),
                IntegerStorage::I64(vec![-10, -20, -30, -40, -50]),
            )
            .expect("storage"),
            vec![5, 1],
        )
        .expect("tensor");

        let Value::ComplexTensor(result) =
            ifftshift_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("ifftshift")
        else {
            panic!("expected complex tensor");
        };
        let storage = result.integer_storage().expect("exact integer storage");
        assert_eq!(
            storage.real,
            IntegerStorage::I64(vec![3, 4, i64::MAX, i64::MIN, 2])
        );
        assert_eq!(
            storage.imag,
            IntegerStorage::I64(vec![-30, -40, -50, -10, -20])
        );
    }

    #[test]
    fn ifftshift_preserves_typed_real_integer_storage_exactly_at_i64_boundaries() {
        let tensor = Tensor::new_integer(
            IntegerStorage::I64(vec![i64::MIN, 2, 3, 4, i64::MAX]),
            vec![5, 1],
        )
        .expect("tensor");

        let Value::Tensor(result) =
            ifftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("ifftshift")
        else {
            panic!("expected tensor");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::I64(vec![3, 4, i64::MAX, i64::MIN, 2]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_matrix_rows_only_via_int() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = ifftshift_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))])
            .expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.materialize_f64(), vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ifftshift_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
        let cpu = ifftshift_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("cpu");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let handle = provider.upload(&view).expect("upload");

        let gpu_value =
            ifftshift_builtin(Value::GpuTensor(handle), Vec::new()).expect("ifftshift gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");

        match cpu {
            Value::Tensor(host) => {
                assert_eq!(gathered.shape, host.shape);
                assert_eq!(gathered.materialize_f64(), host.materialize_f64());
            }
            other => panic!("expected tensor cpu result, got {other:?}"),
        }
    }

    fn ifftshift_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::ifftshift_builtin(value, rest))
    }
}
