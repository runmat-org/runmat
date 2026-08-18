//! MATLAB-compatible `logical` builtin with GPU-aware semantics for RunMat.

use log::trace;
use runmat_accelerate_api::{self, AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, LogicalArray, ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{
    gpu_helpers,
    shape::{canonical_scalar_shape, normalize_scalar_shape},
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
    },
    tensor,
};
use crate::builtins::logical::type_resolvers::logical_like;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

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

const LOGICAL_STRING_ARRAY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "logical-string-array-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "logical with string-array input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LogicalStringArrayInputExtension"),
};
const LOGICAL_SYMBOLIC_CONSTANT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "logical-symbolic-constant-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "logical with a symbolic numeric constant is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LogicalSymbolicConstantInputExtension"),
    };
pub const LOGICAL_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    LOGICAL_STRING_ARRAY_EXTENSION,
    LOGICAL_SYMBOLIC_CONSTANT_EXTENSION,
];

const LOGICAL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight real integer classes convert elementwise without floating materialization; zero becomes false and every nonzero value becomes true.",
    }];
pub const LOGICAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tf = logical(integer_A)",
        inputs: &LOGICAL_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Host conversion reads authoritative integer storage exactly; resident conversion uses a validated owning-provider path or exact gather and class-preserving restoration.",
    }];

const LOGICAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical-converted result.",
}];

const LOGICAL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to convert.",
}];

const LOGICAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = logical(A)",
    inputs: &LOGICAL_INPUTS,
    outputs: &LOGICAL_OUTPUT,
}];

const LOGICAL_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOGICAL.TOO_MANY_INPUTS",
    identifier: Some("RunMat:logical:TooManyInputs"),
    when: "More than one input argument is provided.",
    message: "logical: too many input arguments",
};

const LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOGICAL.CONVERSION_NOT_POSSIBLE",
    identifier: Some("RunMat:logical:ConversionNotPossible"),
    when: "Input type cannot be converted to logical.",
    message: "logical: conversion to logical is not possible for this input type",
};

const LOGICAL_ERROR_GPU_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOGICAL.GPU_GATHER_FAILED",
    identifier: Some("RunMat:logical:GpuGatherFailed"),
    when: "GPU input gather fails during host fallback.",
    message: "logical: failed to gather gpuArray input",
};

const LOGICAL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOGICAL.INTERNAL",
    identifier: Some("RunMat:logical:InternalError"),
    when: "Internal logical buffer materialization fails.",
    message: "logical: internal conversion error",
};

const LOGICAL_ERRORS: [BuiltinErrorDescriptor; 4] = [
    LOGICAL_ERROR_TOO_MANY_INPUTS,
    LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE,
    LOGICAL_ERROR_GPU_GATHER_FAILED,
    LOGICAL_ERROR_INTERNAL,
];

pub const LOGICAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOGICAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOGICAL_ERRORS,
};

fn logical_type(args: &[Type], _context: &ResolveContext) -> Type {
    args.first().map(logical_like).unwrap_or(Type::logical())
}

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
    category = "logical",
    summary = "Convert scalars, arrays, and gpuArray values to logical outputs.",
    keywords = "logical,boolean,gpuArray,mask,conversion",
    accel = "unary",
    type_resolver(logical_type),
    descriptor(crate::builtins::logical::ops::LOGICAL_DESCRIPTOR),
    extensions(crate::builtins::logical::ops::LOGICAL_EXTENSIONS),
    integer_capabilities(crate::builtins::logical::ops::LOGICAL_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::logical::ops"
)]
async fn logical_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(logical_error_with_message(
            LOGICAL_ERROR_TOO_MANY_INPUTS.message,
            &LOGICAL_ERROR_TOO_MANY_INPUTS,
        ));
    }
    convert_value_to_logical(value).await
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
            let Some(value) = expr.numeric_constant_value() else {
                return Err(conversion_error("sym"));
            };
            crate::compatibility::ensure_builtin_extension_enabled(
                &LOGICAL_SYMBOLIC_CONSTANT_EXTENSION,
                BUILTIN_NAME,
            )?;
            if value.is_nan() {
                Err(conversion_error("NaN"))
            } else {
                Ok(Value::Bool(value != 0.0))
            }
        }
        Value::Cell(_) => Err(conversion_error("cell")),
        Value::Struct(_) => Err(conversion_error("struct")),
        Value::Object(obj) => Err(conversion_error(&obj.class_name)),
        Value::HandleObject(handle) => Err(conversion_error(&handle.class_name)),
        Value::Listener(_) => Err(conversion_error("event.listener")),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Err(conversion_error("function_handle")),
        Value::ClassRef(_) => Err(conversion_error("meta.class")),
        Value::MException(_) => Err(conversion_error("MException")),
        Value::OutputList(_) => Err(conversion_error("OutputList")),
    }
}

fn logical_from_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    if tensor.integer_storage().is_none()
        && tensor.materialize_f64().iter().any(|value| value.is_nan())
    {
        return Err(conversion_error("NaN"));
    }
    let buffer = LogicalBuffer::from_real_tensor(&tensor);
    logical_buffer_to_host(buffer)
}

fn logical_from_sparse_tensor(sparse: runmat_builtins::SparseTensor) -> BuiltinResult<Value> {
    if sparse.is_logical() {
        return Ok(Value::SparseTensor(sparse));
    }
    if sparse.integer_storage().is_none()
        && (0..sparse.nnz()).any(|index| {
            sparse
                .numeric_value_at(index)
                .is_some_and(|value| match value {
                    runmat_builtins::NumericScalar::F64(value) => value.is_nan(),
                    runmat_builtins::NumericScalar::F32(value) => value.is_nan(),
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
    runmat_builtins::SparseTensor::new_logical(sparse.rows, sparse.cols, col_ptrs, row_indices)
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
    let buffer = LogicalBuffer::from_real_tensor(&tensor);
    logical_buffer_to_gpu(
        buffer,
        provider,
        runmat_accelerate_api::handle_is_explicit(&handle),
    )
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
        gpu_helpers::free_unprotected_exact_owner(&mask, &[source]);
        return Err(logical_error_with_message(
            "logical: provider returned a malformed NaN mask",
            &LOGICAL_ERROR_INTERNAL,
        ));
    }
    let maximum = provider.reduce_max(&mask).await.map_err(|error| {
        gpu_helpers::free_unprotected_exact_owner(&mask, &[source]);
        logical_error_with_message(error.to_string(), &LOGICAL_ERROR_INTERNAL)
    })?;
    let maximum_valid = !gpu_helpers::same_gpu_handle(&maximum, source)
        && !gpu_helpers::same_gpu_handle(&maximum, &mask)
        && maximum.shape.iter().product::<usize>() == 1
        && maximum.device_id == source.device_id
        && gpu_helpers::exact_provider_for_handle(&maximum)
            .is_some_and(|owner| std::ptr::eq(owner, provider));
    if !maximum_valid {
        gpu_helpers::free_unprotected_exact_owner(&maximum, &[source, &mask]);
        gpu_helpers::free_unprotected_exact_owner(&mask, &[source]);
        return Err(logical_error_with_message(
            "logical: provider returned a malformed NaN reduction",
            &LOGICAL_ERROR_INTERNAL,
        ));
    }
    let downloaded = provider.download(&maximum).await.map_err(|error| {
        gpu_helpers::free_unprotected_exact_owner(&maximum, &[source, &mask]);
        gpu_helpers::free_unprotected_exact_owner(&mask, &[source]);
        logical_error_with_message(error.to_string(), &LOGICAL_ERROR_INTERNAL)
    })?;
    gpu_helpers::free_unprotected_exact_owner(&maximum, &[source, &mask]);
    gpu_helpers::free_unprotected_exact_owner(&mask, &[source]);
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
    explicit: bool,
) -> BuiltinResult<Value> {
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
    } else {
        if explicit {
            Err(logical_error_with_message(
                "logical: no exact owner for explicit gpuArray input",
                &LOGICAL_ERROR_GPU_GATHER_FAILED,
            ))
        } else {
            logical_buffer_to_host(buffer)
        }
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
        gpu_helpers::free_unprotected_exact_owner(&zeros, &[input]);
        return None;
    }
    let result = provider
        .elem_ne(input, &zeros)
        .await
        .ok()
        .and_then(|output| {
            if valid_logical_gpu_output(&output, input, provider) {
                Some(output)
            } else {
                gpu_helpers::free_unprotected_exact_owner(&output, &[input, &zeros]);
                None
            }
        });
    let _ = provider.free(&zeros);
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
            .data
            .iter()
            .map(|&ch| if (ch as u32) != 0 { 1 } else { 0 })
            .collect();
        let original_shape = vec![chars.rows, chars.cols];
        let shape = canonical_shape(&original_shape, bits.len());
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
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, MException,
        ObjectInstance, SparseTensor, StructValue, SymbolicExpr,
    };

    fn logical_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::logical_builtin(value, rest))
    }

    fn assert_error_message(err: &crate::RuntimeError, expected: &str) {
        assert_eq!(err.message(), expected);
    }

    fn assert_error_contains(err: &crate::RuntimeError, expected: &str) {
        assert!(
            err.message().contains(expected),
            "unexpected error: {}",
            err.message()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_scalar_num() {
        let result = logical_builtin(Value::Num(5.0), Vec::new()).expect("logical");
        assert_eq!(result, Value::Bool(true));

        let zero_result = logical_builtin(Value::Num(0.0), Vec::new()).expect("logical");
        assert_eq!(zero_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_converts_symbolic_constants() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let nonzero = logical_builtin(Value::Symbolic(SymbolicExpr::constant(2.0)), Vec::new())
            .expect("logical");
        assert_eq!(nonzero, Value::Bool(true));

        let zero = logical_builtin(Value::Symbolic(SymbolicExpr::constant(0.0)), Vec::new())
            .expect("logical");
        assert_eq!(zero, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_rejects_symbolic_variables() {
        let err = logical_builtin(Value::Symbolic(SymbolicExpr::variable("x")), Vec::new())
            .expect_err("symbolic variable should not convert");

        assert_eq!(
            err.identifier(),
            LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE.identifier
        );
        assert!(err.message().contains("logical from sym"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_rejects_nan() {
        let tensor = Tensor::new(vec![0.0, f64::NAN, -0.0], vec![1, 3]).unwrap();
        let error = logical_builtin(Value::Tensor(tensor), Vec::new())
            .expect_err("NaN conversion must fail");
        assert!(error.message().contains("NaN"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_tensor_matrix() {
        let tensor = Tensor::new(vec![0.0, 2.0, -3.0, 0.0], vec![2, 2]).unwrap();
        let result = logical_builtin(Value::Tensor(tensor), Vec::new()).expect("logical");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 2]);
                assert_eq!(array.data, vec![0, 1, 1, 0]);
            }
            other => panic!("expected logical array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_sparse_tensor_preserves_sparse_storage() {
        let sparse = SparseTensor::new(3, 2, vec![0, 1, 2], vec![1, 2], vec![4.0, -1.0]).unwrap();
        let result = logical_builtin(Value::SparseTensor(sparse), Vec::new()).expect("logical");
        match result {
            Value::SparseTensor(sparse) => {
                assert!(sparse.is_logical());
                assert_eq!(sparse.shape(), vec![3, 2]);
                assert_eq!(sparse.col_ptrs, vec![0, 1, 2]);
                assert_eq!(sparse.row_indices, vec![1, 2]);
                assert_eq!(
                    sparse.to_dense_logical().expect("dense logical").data,
                    vec![0, 1, 0, 0, 0, 1]
                );
            }
            other => panic!("expected logical sparse tensor, got {other:?}"),
        }
    }

    #[test]
    fn logical_sparse_nan_rejects_like_dense_nan() {
        let sparse = SparseTensor::new(2, 1, vec![0, 1], vec![0], vec![f64::NAN]).unwrap();
        let error = logical_builtin(Value::SparseTensor(sparse), Vec::new())
            .expect_err("sparse NaN must reject");
        assert_eq!(
            error.identifier(),
            Some("RunMat:logical:ConversionNotPossible")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_rejects_complex_conversion() {
        let complex =
            ComplexTensor::new(vec![(0.0, 0.0), (1.0, 0.0), (0.0, 2.0)], vec![3, 1]).unwrap();
        let error = logical_builtin(Value::ComplexTensor(complex), Vec::new())
            .expect_err("complex conversion must fail");
        assert!(error.message().contains("complex"));
    }

    #[test]
    fn logical_rejects_typed_complex_integer_components() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![0, u64::MAX, 0]),
            IntegerStorage::U64(vec![0, 0, 1_u64 << 63]),
        )
        .expect("matching components");
        let tensor = ComplexTensor::new_integer(storage, vec![3, 1]).expect("typed complex");

        let error = logical_builtin(Value::ComplexTensor(tensor), Vec::new())
            .expect_err("complex conversion must fail");
        assert!(error.message().contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_char_array_conversion() {
        let chars = CharArray::new(vec!['A', '\0', 'C'], 1, 3).unwrap();
        let result = logical_builtin(Value::CharArray(chars), Vec::new()).expect("logical");
        match result {
            Value::LogicalArray(array) => assert_eq!(array.data, vec![1, 0, 1]),
            other => panic!("expected logical array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_string_error() {
        let err = logical_builtin(Value::String("runmat".to_string()), Vec::new()).unwrap_err();
        assert_error_message(
            &err,
            "logical: conversion to logical from string is not possible",
        );
        assert_eq!(
            err.identifier(),
            LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_struct_error() {
        let mut st = StructValue::new();
        st.insert("field", Value::Num(1.0));
        let err = logical_builtin(Value::Struct(st), Vec::new()).unwrap_err();
        assert_error_contains(&err, "struct");
        assert_eq!(
            err.identifier(),
            LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_cell_error() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell creation");
        let err = logical_builtin(Value::Cell(cell), Vec::new()).unwrap_err();
        assert_error_message(
            &err,
            "logical: conversion to logical from cell is not possible",
        );
        assert_eq!(
            err.identifier(),
            LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_function_handle_error() {
        let err = logical_builtin(Value::FunctionHandle("foo".into()), Vec::new()).unwrap_err();
        assert_error_message(
            &err,
            "logical: conversion to logical from function_handle is not possible",
        );
        assert_eq!(
            err.identifier(),
            LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_object_error() {
        let obj = ObjectInstance::new("DemoClass".to_string());
        let err = logical_builtin(Value::Object(obj), Vec::new()).unwrap_err();
        assert_error_contains(&err, "DemoClass");
        assert_eq!(
            err.identifier(),
            LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_mexception_error() {
        let mex = MException::new("id:logical".into(), "message".into());
        let err = logical_builtin(Value::MException(mex), Vec::new()).unwrap_err();
        assert_error_message(
            &err,
            "logical: conversion to logical from MException is not possible",
        );
        assert_eq!(
            err.identifier(),
            LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_too_many_inputs_error() {
        let err = logical_builtin(Value::Bool(true), vec![Value::Bool(false)]).unwrap_err();
        assert_error_message(&err, LOGICAL_ERROR_TOO_MANY_INPUTS.message);
        assert_eq!(err.identifier(), LOGICAL_ERROR_TOO_MANY_INPUTS.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, -2.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                logical_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("logical");
            let gathered = test_support::gather(result.clone()).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![0.0, 1.0, 1.0]);
            if let Value::GpuTensor(out) = result {
                assert!(runmat_accelerate_api::handle_is_logical(&out));
            } else {
                panic!("expected gpu tensor output");
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_gpu_passthrough_for_logical_handle() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            runmat_accelerate_api::set_handle_logical(&handle, true);
            let result =
                logical_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("logical");
            match result {
                Value::GpuTensor(out) => assert_eq!(out, handle),
                other => panic!("expected gpu tensor, got {:?}", other),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_bool_and_logical_inputs_passthrough() {
        let res_bool = logical_builtin(Value::Bool(true), Vec::new()).expect("logical");
        assert_eq!(res_bool, Value::Bool(true));

        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let res_array =
            logical_builtin(Value::LogicalArray(logical.clone()), Vec::new()).expect("logical");
        assert_eq!(res_array, Value::LogicalArray(logical));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_empty_tensor_preserves_shape() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = logical_builtin(Value::Tensor(tensor), Vec::new()).expect("logical");
        match result {
            Value::LogicalArray(array) => {
                assert!(array.data.is_empty());
                assert_eq!(array.shape, vec![0, 3]);
            }
            other => panic!("expected logical array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_integer_scalar() {
        let res = logical_builtin(Value::Int(IntValue::I32(0)), Vec::new()).expect("logical");
        assert_eq!(res, Value::Bool(false));

        let res_nonzero =
            logical_builtin(Value::Int(IntValue::I32(-5)), Vec::new()).expect("logical");
        assert_eq!(res_nonzero, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn logical_wgpu_matches_cpu_conversion() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };

        let tensor = Tensor::new(vec![0.0, 2.0, -3.0, 1.0], vec![2, 2]).unwrap();
        let cpu = logical_builtin(Value::Tensor(tensor.clone()), Vec::new()).unwrap();

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = logical_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
        let out_handle = match gpu_value {
            Value::GpuTensor(ref h) => {
                assert!(runmat_accelerate_api::handle_is_logical(h));
                h.clone()
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        };

        let gathered = test_support::gather(Value::GpuTensor(out_handle)).expect("gather");

        let (expected, expected_shape): (Vec<f64>, Vec<usize>) = match cpu {
            Value::LogicalArray(arr) => (
                arr.data
                    .iter()
                    .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                    .collect(),
                arr.shape.clone(),
            ),
            Value::Bool(flag) => (vec![if flag { 1.0 } else { 0.0 }], vec![1, 1]),
            other => panic!("unexpected cpu result {other:?}"),
        };

        assert_eq!(gathered.shape, expected_shape);
        assert_eq!(gathered.materialize_f64(), expected);
    }
}
