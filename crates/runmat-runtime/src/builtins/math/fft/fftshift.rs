//! MATLAB-compatible `fftshift` builtin with GPU-aware semantics for RunMat.
//!
//! `fftshift` recenters zero-frequency components for outputs produced by FFTs.

use super::common::{
    apply_shift, apply_shift_numeric_storage, build_shift_plan, compute_shift_dims,
    gather_gpu_complex_tensor, restore_complex_gpu_result, ShiftKind,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::fft::type_resolvers::fftshift_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexStorage, ComplexTensor, LogicalArray, Tensor, Value};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::fftshift")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fftshift",
    op_kind: GpuOpKind::Custom("fftshift"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("circshift")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Delegates to provider circshift kernels when available; otherwise gathers once and shifts on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::fftshift")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fftshift",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not currently fused; treated as an explicit data shuffling operation.",
};

const BUILTIN_NAME: &str = "fftshift";

const FFTSHIFT_MULTI_DIM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fftshift-multi-dimension-selector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fftshift numeric dimension vectors and logical masks are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FftshiftMultiDimensionExtension"),
};
pub const FFTSHIFT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [FFTSHIFT_MULTI_DIM_EXTENSION];

const FFTSHIFT_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All integer classes are reordered exactly with class, shape, and complexity preserved.",
}];
const FFTSHIFT_DIM_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "DIM",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes:
        "The documented DIM form is one positive scalar parsed from authoritative integer storage.",
}];
const FFTSHIFT_MULTI_DIM_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "DIMS",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "RunMat-only numeric vectors are decoded exactly; logical mask vectors share the same independent extension gate.",
    }];
pub const FFTSHIFT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "Y = fftshift(integer_X)", inputs: &FFTSHIFT_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "Real and typed-complex integer values are reordered exactly with class preserved; resident fallback re-uploads to the owning provider." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = fftshift(X, integer_DIM)", inputs: &FFTSHIFT_DIM_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "All eight documented integer scalar classes select one dimension exactly." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = fftshift(X, integer_DIMS)", inputs: &FFTSHIFT_MULTI_DIM_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat-only multi-axis vector/mask selection is independently gated from documented scalar DIM semantics." },
];

const FFTSHIFT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Shifted array with the same size and type family as X.",
}];

const FFTSHIFT_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, complex, logical, or gpuArray input.",
}];

const FFTSHIFT_INPUTS_DIMS: [BuiltinParamDescriptor; 2] = [
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
        default: None,
        description: "Positive scalar dimension; vectors and logical masks are RunMat extensions.",
    },
];

const FFTSHIFT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = fftshift(X)",
        inputs: &FFTSHIFT_INPUTS_CORE,
        outputs: &FFTSHIFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = fftshift(X, DIM)",
        inputs: &FFTSHIFT_INPUTS_DIMS,
        outputs: &FFTSHIFT_OUTPUT,
    },
];

const FFTSHIFT_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.ARG_COUNT",
    identifier: Some("RunMat:fftshift:ArgCount"),
    when: "More than two input arguments are supplied.",
    message: "fftshift: invalid argument count",
};

const FFTSHIFT_ERROR_INVALID_DIMS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.INVALID_DIMS",
    identifier: Some("RunMat:fftshift:InvalidDimensions"),
    when: "DIM argument is malformed or out of range.",
    message: "fftshift: invalid dimension argument",
};

const FFTSHIFT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.INVALID_INPUT",
    identifier: Some("RunMat:fftshift:InvalidInput"),
    when: "X is not a supported numeric/logical input type.",
    message: "fftshift: expected numeric or logical input",
};

const FFTSHIFT_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:fftshift:UnsupportedInput"),
    when: "X is an unsupported object/cell/function/meta runtime type.",
    message: "fftshift: unsupported input type",
};

const FFTSHIFT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTSHIFT.INTERNAL",
    identifier: Some("RunMat:fftshift:Internal"),
    when: "Shifting, tensor reconstruction, or GPU transfer operations fail.",
    message: "fftshift: internal error",
};

const FFTSHIFT_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FFTSHIFT_ERROR_ARG_COUNT,
    FFTSHIFT_ERROR_INVALID_DIMS,
    FFTSHIFT_ERROR_INVALID_INPUT,
    FFTSHIFT_ERROR_UNSUPPORTED_INPUT,
    FFTSHIFT_ERROR_INTERNAL,
];

pub const FFTSHIFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FFTSHIFT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FFTSHIFT_ERRORS,
};

fn fftshift_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    fftshift_error_with_message(error.message, error)
}

fn fftshift_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fftshift_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fftshift_error_with_source(
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

fn fftshift_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn compute_fftshift_dims(shape: &[usize], dims_arg: Option<&Value>) -> BuiltinResult<Vec<usize>> {
    compute_shift_dims(shape, dims_arg, BUILTIN_NAME).map_err(|source| {
        fftshift_error_with_source(
            &FFTSHIFT_ERROR_INVALID_DIMS,
            "dimension parsing failed",
            source,
        )
    })
}

#[runtime_builtin(
    name = "fftshift",
    category = "math/fft",
    summary = "Shift zero-frequency components to spectrum centers.",
    keywords = "fftshift,fourier transform,frequency centering,spectrum,gpu",
    accel = "custom",
    type_resolver(fftshift_type),
    descriptor(crate::builtins::math::fft::fftshift::FFTSHIFT_DESCRIPTOR),
    extensions(crate::builtins::math::fft::fftshift::FFTSHIFT_EXTENSIONS),
    integer_capabilities(crate::builtins::math::fft::fftshift::FFTSHIFT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::fft::fftshift"
)]
async fn fftshift_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(fftshift_error(&FFTSHIFT_ERROR_ARG_COUNT));
    }
    let dims_arg = rest.first();
    if dims_arg.is_some_and(fftshift_uses_multi_dimension_extension) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FFTSHIFT_MULTI_DIM_EXTENSION,
            BUILTIN_NAME,
        )?;
    }

    match value {
        Value::Tensor(tensor) => {
            let dims = compute_fftshift_dims(&tensor.shape, dims_arg)?;
            Ok(fftshift_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::ComplexTensor(ct) => {
            let dims = compute_fftshift_dims(&ct.shape, dims_arg)?;
            Ok(fftshift_complex_tensor(ct, &dims).map(Value::ComplexTensor)?)
        }
        Value::LogicalArray(array) => {
            let dims = compute_fftshift_dims(&array.shape, dims_arg)?;
            Ok(fftshift_logical(array, &dims).map(Value::LogicalArray)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|source| {
                fftshift_error_with_detail(
                    &FFTSHIFT_ERROR_INTERNAL,
                    format!("complex tensor construction failed: {source}"),
                )
            })?;
            let dims = compute_fftshift_dims(&tensor.shape, dims_arg)?;
            Ok(fftshift_complex_tensor(tensor, &dims).map(|result| {
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
                fftshift_error_with_detail(
                    &FFTSHIFT_ERROR_INVALID_INPUT,
                    format!("scalar/tensor conversion failed: {detail}"),
                )
            })?;
            let dims = compute_fftshift_dims(&tensor.shape, dims_arg)?;
            Ok(fftshift_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::GpuTensor(handle) => {
            let dims = compute_fftshift_dims(&handle.shape, dims_arg)?;
            Ok(fftshift_gpu(handle, &dims).await?)
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => {
            Err(fftshift_error(&FFTSHIFT_ERROR_INVALID_INPUT))
        }
        Value::Symbolic(_) | Value::SymbolicArray(_) => {
            Err(fftshift_error(&FFTSHIFT_ERROR_INVALID_INPUT))
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
        | Value::Foreign(_)
        | Value::OutputList(_) => Err(fftshift_error(&FFTSHIFT_ERROR_UNSUPPORTED_INPUT)),
    }
}

fn fftshift_uses_multi_dimension_extension(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.len() != 1,
        Value::LogicalArray(array) => array.data.len() != 1,
        _ => false,
    }
}

fn fftshift_tensor(tensor: Tensor, dims: &[usize]) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor.into_numeric_storage().map_err(|source| {
        fftshift_error_with_detail(
            &FFTSHIFT_ERROR_INTERNAL,
            format!("invalid tensor storage: {source}"),
        )
    })?;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Fft);
    let rotated =
        apply_shift_numeric_storage(BUILTIN_NAME, storage, &plan.ext_shape, &plan.positive)?;
    Tensor::from_numeric_storage(rotated, shape).map_err(|source| {
        fftshift_error_with_detail(
            &FFTSHIFT_ERROR_INTERNAL,
            format!("tensor reconstruction failed: {source}"),
        )
    })
}

fn fftshift_complex_tensor(tensor: ComplexTensor, dims: &[usize]) -> BuiltinResult<ComplexTensor> {
    let shape = tensor.shape.clone();
    let storage = tensor.into_complex_storage();
    let plan = build_shift_plan(&shape, dims, ShiftKind::Fft);
    if storage.is_empty() || plan.is_noop() {
        return ComplexTensor::from_complex_storage(storage, shape).map_err(|source| {
            fftshift_error_with_detail(
                &FFTSHIFT_ERROR_INTERNAL,
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
                    fftshift_error_with_detail(
                        &FFTSHIFT_ERROR_INTERNAL,
                        format!("complex integer shift failed: {source}"),
                    )
                })?,
        ),
    };
    ComplexTensor::from_complex_storage(rotated, shape).map_err(|source| {
        fftshift_error_with_detail(
            &FFTSHIFT_ERROR_INTERNAL,
            format!("complex tensor reconstruction failed: {source}"),
        )
    })
}

fn fftshift_logical(array: LogicalArray, dims: &[usize]) -> BuiltinResult<LogicalArray> {
    let LogicalArray { data, shape } = array;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Fft);
    if data.is_empty() || plan.is_noop() {
        return LogicalArray::new(data, shape).map_err(|source| {
            fftshift_error_with_detail(
                &FFTSHIFT_ERROR_INTERNAL,
                format!("logical array reconstruction failed: {source}"),
            )
        });
    }
    let rotated = apply_shift(BUILTIN_NAME, &data, &plan.ext_shape, &plan.positive)?;
    LogicalArray::new(rotated, shape).map_err(|source| {
        fftshift_error_with_detail(
            &FFTSHIFT_ERROR_INTERNAL,
            format!("logical array reconstruction failed: {source}"),
        )
    })
}

async fn fftshift_gpu(handle: GpuTensorHandle, dims: &[usize]) -> BuiltinResult<Value> {
    let plan = build_shift_plan(&handle.shape, dims, ShiftKind::Fft);
    if plan.is_noop() {
        return Ok(Value::GpuTensor(handle));
    }

    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        // Provider reshape may mutate a handle's registry entry in place. Only use
        // the native path when no synthetic trailing dimensions are required.
        if plan.ext_shape == handle.shape {
            if let Ok(out) = provider.circshift(&handle, &plan.provider) {
                return Ok(gpu_helpers::resident_gpu_value(out));
            }
        }
    }

    fftshift_gpu_fallback(handle, dims).await
}

async fn fftshift_gpu_fallback(handle: GpuTensorHandle, dims: &[usize]) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        let host = gather_gpu_complex_tensor(&handle, BUILTIN_NAME)
            .await
            .map_err(|source| {
                fftshift_error_with_source(&FFTSHIFT_ERROR_INTERNAL, "gpu gather failed", source)
            })?;
        let shifted = fftshift_complex_tensor(host, dims)?;
        return restore_complex_gpu_result(&handle, &shifted, BUILTIN_NAME);
    }
    let was_logical = runmat_accelerate_api::handle_is_logical(&handle);
    let host_tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|source| {
            fftshift_error_with_source(&FFTSHIFT_ERROR_INTERNAL, "gpu gather failed", source)
        })?;
    let shifted = fftshift_tensor(host_tensor, dims)?;
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        return gpu_helpers::upload_tensor(provider, &shifted)
            .map(|output| {
                if was_logical {
                    gpu_helpers::logical_gpu_value(output)
                } else {
                    gpu_helpers::resident_gpu_value(output)
                }
            })
            .map_err(|source| {
                fftshift_error_with_detail(
                    &FFTSHIFT_ERROR_INTERNAL,
                    format!("gpu upload failed: {source}"),
                )
            });
    }
    Ok(tensor::tensor_into_value(shifted))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, ResolveContext, Type};
    use runmat_value::{
        ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, Tensor,
    };

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    #[test]
    fn fftshift_type_preserves_tensor_shape() {
        let out = fftshift_type(
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
    fn fftshift_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("fftshift builtin");
        let descriptor = builtin.descriptor.expect("fftshift descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = fftshift(X)"));
        assert!(labels.contains(&"Y = fftshift(X, DIM)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.FFTSHIFT.INVALID_DIMS"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_even_length_vector() {
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("fftshift");
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
    fn fftshift_odd_length_vector() {
        let tensor = Tensor::new((1..=5).map(|v| v as f64).collect(), vec![5, 1]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![5, 1]);
                assert_eq!(out.materialize_f64(), vec![4.0, 5.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_matrix_rows_only() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))])
            .expect("fftshift");
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
    fn fftshift_matrix_columns_only_via_vector_dims() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let dims = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let result =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.materialize_f64(), vec![3.0, 6.0, 1.0, 4.0, 2.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn fftshift_integer_contract_separates_scalar_dim_from_multi_axis_extension() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("fftshift registration");
        assert_eq!(builtin.integer_capabilities.len(), 3);
        assert_eq!(builtin.extensions.len(), 1);

        let scalar_logical = fftshift_builtin(
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
            vec![Value::Bool(true)],
        )
        .expect("documented logical scalar DIM");
        assert!(matches!(
            scalar_logical,
            Value::Int(runmat_value::IntValue::U64(u64::MAX))
        ));

        let dims =
            Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![1, 2]), vec![1, 2]).unwrap();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = fftshift_builtin(Value::Num(1.0), vec![Value::Tensor(dims)])
            .expect_err("multi-axis selector must gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FftshiftMultiDimensionExtension")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_matrix_rows_only_logical_mask() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let mask = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), vec![Value::LogicalArray(mask)])
            .expect("fftshift");
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
    fn fftshift_matrix_all_dims() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.materialize_f64(), vec![6.0, 3.0, 4.0, 1.0, 5.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_with_empty_dimension_vector_noop() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let dims = Tensor::new(Vec::new(), vec![0, 1]).unwrap();
        let original = tensor.clone();
        let result =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, original.shape);
                assert_eq!(out.materialize_f64(), original.materialize_f64());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_dimension_beyond_rank_is_ignored() {
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![2, 4]).unwrap();
        let dims = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let original = tensor.clone();
        let result =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, original.shape);
                assert_eq!(out.materialize_f64(), original.materialize_f64());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_logical_array_input_supported() {
        let logical = LogicalArray::new(vec![1, 0, 0, 0], vec![4, 1]).unwrap();
        let result = fftshift_builtin(Value::LogicalArray(logical), Vec::new()).expect("fftshift");
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
    fn fftshift_complex_tensor() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
            vec![4, 1],
        )
        .unwrap();
        let result = fftshift_builtin(Value::ComplexTensor(tensor), Vec::new()).unwrap();
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
    fn fftshift_preserves_typed_complex_integer_storage_at_u64_boundaries() {
        let tensor = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![u64::MAX, 2, 3, 4, 1_u64 << 63]),
                IntegerStorage::U64(vec![10, 20, 30, 40, 50]),
            )
            .expect("storage"),
            vec![5, 1],
        )
        .expect("tensor");

        let Value::ComplexTensor(result) =
            fftshift_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("fftshift")
        else {
            panic!("expected complex tensor");
        };
        let storage = result.integer_storage().expect("exact integer storage");
        assert_eq!(
            storage.real,
            IntegerStorage::U64(vec![4, 1_u64 << 63, u64::MAX, 2, 3])
        );
        assert_eq!(storage.imag, IntegerStorage::U64(vec![40, 50, 10, 20, 30]));
    }

    #[test]
    fn fftshift_preserves_typed_real_integer_storage_exactly_at_u64_boundaries() {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 2, 3, 4, 1_u64 << 63]),
            vec![5, 1],
        )
        .expect("tensor");

        let Value::Tensor(result) =
            fftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("fftshift")
        else {
            panic!("expected tensor");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![4, 1_u64 << 63, u64::MAX, 2, 3]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_complex_scalar_passthrough() {
        let result = fftshift_builtin(Value::Complex(1.0, -2.0), Vec::new()).expect("fftshift");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.0);
                assert_eq!(im, -2.0);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_rejects_zero_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = fftshift_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))])
            .unwrap_err();
        assert_eq!(
            error_identifier(&err),
            FFTSHIFT_ERROR_INVALID_DIMS.identifier
        );
        assert!(
            error_message(err).contains(FFTSHIFT_ERROR_INVALID_DIMS.message),
            "unexpected error"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_rejects_non_integer_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = fftshift_builtin(Value::Tensor(tensor), vec![Value::Num(1.5)]).unwrap_err();
        assert_eq!(
            error_identifier(&err),
            FFTSHIFT_ERROR_INVALID_DIMS.identifier
        );
        assert!(
            error_message(err).contains(FFTSHIFT_ERROR_INVALID_DIMS.message),
            "unexpected error"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_rejects_non_numeric_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::from("invalid")]).unwrap_err();
        assert_eq!(
            error_identifier(&err),
            FFTSHIFT_ERROR_INVALID_DIMS.identifier
        );
        assert!(
            error_message(err).contains(FFTSHIFT_ERROR_INVALID_DIMS.message),
            "unexpected error"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_rejects_non_vector_dimension_tensor() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).unwrap_err();
        assert_eq!(
            error_identifier(&err),
            FFTSHIFT_ERROR_INVALID_DIMS.identifier
        );
        assert!(
            error_message(err).contains(FFTSHIFT_ERROR_INVALID_DIMS.message),
            "unexpected error"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = fftshift_builtin(Value::GpuTensor(handle), Vec::new()).expect("fftshift");
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
    fn fftshift_gpu_with_explicit_dims() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let dims = Value::Int(IntValue::I32(1));
            let result = fftshift_builtin(Value::GpuTensor(handle), vec![dims]).expect("fftshift");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 3]);
            assert_eq!(
                gathered.materialize_f64(),
                vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fftshift_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
        let cpu =
            fftshift_tensor(tensor.clone(), &(0..tensor.shape.len()).collect::<Vec<_>>()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = fftshift_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.materialize_f64(), cpu.materialize_f64());
    }

    fn fftshift_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::fftshift_builtin(value, rest))
    }
}
