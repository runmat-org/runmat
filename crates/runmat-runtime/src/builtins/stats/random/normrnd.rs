use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, NumericDType, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::common::random_args::extract_dims;
use crate::builtins::common::tensor;
use crate::builtins::common::{gpu_helpers, random};

const BUILTIN_NAME: &str = "normrnd";

const INTEGER_MU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "normrnd-integer-mu",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "normrnd with a typed-integer mean parameter is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormrndIntegerMuExtension"),
};
const INTEGER_SIGMA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "normrnd-integer-sigma",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "normrnd with a typed-integer standard deviation is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormrndIntegerSigmaExtension"),
};
const INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "normrnd-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "normrnd with typed-integer size arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormrndIntegerSizeExtension"),
};
const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "normrnd-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "normrnd with logical numeric inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormrndLogicalInputExtension"),
};
const RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "normrnd-resident-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "normrnd with explicit gpuArray size controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormrndResidentSizeExtension"),
};

pub const EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    INTEGER_MU_EXTENSION,
    INTEGER_SIGMA_EXTENSION,
    INTEGER_SIZE_EXTENSION,
    LOGICAL_INPUT_EXTENSION,
    RESIDENT_SIZE_EXTENSION,
];

const INTEGER_PARAMETER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "mu",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer means are independently gated and cross one exact binary64 boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "sigma",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer standard deviations are independently gated and cross one exact binary64 boundary.",
    },
];
const INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "sz, sz1, ...",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Current MATLAB documents single and double size arguments; native integer size controls are a gated RunMat extension parsed structurally.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "r = normrnd(integer_mu, integer_sigma, ___)",
        inputs: &INTEGER_PARAMETER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer distribution parameters are RunMat-only; output is double unless another documented data parameter is single. Fallback restores through the exact owner when it can preserve the required class, otherwise automatic residency may remain host and explicit residency errors.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "r = normrnd(mu, sigma, integer_sz)",
        inputs: &INTEGER_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer extents are parsed exactly without selecting output class or residency; explicit resident size controls are separately gated.",
    },
];

const NORMRND_OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random sample array from normal distribution.",
}];

const NORMRND_INPUTS_MU_SIGMA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mean parameter.",
    },
    BuiltinParamDescriptor {
        name: "sigma",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard deviation parameter (must be >= 0).",
    },
];

const NORMRND_INPUTS_MU_SIGMA_SZ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mean parameter.",
    },
    BuiltinParamDescriptor {
        name: "sigma",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard deviation parameter (must be >= 0).",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Size scalar or size vector argument.",
    },
];

const NORMRND_INPUTS_MU_SIGMA_DIMS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mean parameter.",
    },
    BuiltinParamDescriptor {
        name: "sigma",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard deviation parameter (must be >= 0).",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension extents for output shape.",
    },
];

const NORMRND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = normrnd(mu, sigma)",
        inputs: &NORMRND_INPUTS_MU_SIGMA,
        outputs: &NORMRND_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = normrnd(mu, sigma, sz)",
        inputs: &NORMRND_INPUTS_MU_SIGMA_SZ,
        outputs: &NORMRND_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = normrnd(mu, sigma, sz1, sz2, ...)",
        inputs: &NORMRND_INPUTS_MU_SIGMA_DIMS,
        outputs: &NORMRND_OUTPUT_R,
    },
];

const NORMRND_ERROR_SIGMA_MUST_BE_NONNEGATIVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORMRND.SIGMA_MUST_BE_NONNEGATIVE",
    identifier: Some("RunMat:normrnd:SigmaMustBeNonnegative"),
    when: "sigma is negative.",
    message: "normrnd: sigma must be non-negative",
};

const NORMRND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORMRND.INVALID_ARGUMENT",
    identifier: Some("RunMat:normrnd:InvalidArgument"),
    when: "Input parameters or size arguments are missing or malformed.",
    message: "normrnd: invalid argument",
};

const NORMRND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORMRND.INTERNAL",
    identifier: Some("RunMat:normrnd:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "normrnd: internal operation failed",
};

const NORMRND_ERRORS: [BuiltinErrorDescriptor; 3] = [
    NORMRND_ERROR_SIGMA_MUST_BE_NONNEGATIVE,
    NORMRND_ERROR_INVALID_ARGUMENT,
    NORMRND_ERROR_INTERNAL,
];

pub const NORMRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NORMRND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NORMRND_ERRORS,
};

fn normrnd_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn normrnd_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    normrnd_error_with(error, error.message)
}

fn normrnd_internal_error(message: impl Into<String>) -> crate::RuntimeError {
    normrnd_error_with(&NORMRND_ERROR_INTERNAL, message)
}

fn normrnd_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    if args.len() <= 2 {
        Type::Num
    } else {
        Type::Unknown
    }
}

#[runtime_builtin(
    name = "normrnd",
    category = "stats/random",
    summary = "Normally-distributed random numbers with mean mu and standard deviation sigma.",
    keywords = "normrnd,normal,gaussian,random,distribution,statistics",
    type_resolver(normrnd_type),
    descriptor(crate::builtins::stats::random::normrnd::NORMRND_DESCRIPTOR),
    extensions(crate::builtins::stats::random::normrnd::EXTENSIONS),
    integer_capabilities(crate::builtins::stats::random::normrnd::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::random::normrnd"
)]
async fn normrnd_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    ensure_extensions(&args)?;
    let output = NormrndOutputPlan::inspect(&args)?;
    let (mu, sigma, shape) = parse_args(args).await?;
    if sigma < 0.0 {
        return Err(normrnd_error(&NORMRND_ERROR_SIGMA_MUST_BE_NONNEGATIVE));
    }
    if let Some(value) = try_gpu_normrnd(&output, mu, sigma, &shape)? {
        return Ok(value);
    }
    let len = tensor::element_count(&shape);
    let data = random::generate_normal_scaled(mu, sigma, len, "normrnd")?;
    output.finish(data, shape)
}

struct NormrndOutputPlan {
    single: bool,
    source: Option<runmat_accelerate_api::GpuTensorHandle>,
}

impl NormrndOutputPlan {
    fn inspect(args: &[Value]) -> crate::BuiltinResult<Self> {
        let data = args.iter().take(2);
        let single = data.clone().any(|value| {
            matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
                || matches!(value, Value::GpuTensor(handle)
                    if runmat_accelerate_api::handle_integer_type(handle).is_none()
                        && !runmat_accelerate_api::handle_is_logical(handle)
                        && runmat_accelerate_api::handle_precision(handle)
                            == Some(runmat_accelerate_api::ProviderPrecision::F32))
        });
        let source = gpu_helpers::select_resident_output_source(
            args.iter().take(2).filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
            BUILTIN_NAME,
        )?;
        Ok(Self { single, source })
    }

    fn host_value(&self, data: Vec<f64>, shape: Vec<usize>) -> crate::BuiltinResult<Value> {
        if self.single {
            return Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
                .map(Value::Tensor)
                .map_err(|err| normrnd_internal_error(format!("normrnd: {err}")));
        }
        Tensor::new(data, shape)
            .map(tensor::tensor_into_value)
            .map_err(|err| normrnd_internal_error(format!("normrnd: {err}")))
    }

    fn finish(&self, data: Vec<f64>, shape: Vec<usize>) -> crate::BuiltinResult<Value> {
        let host = self.host_value(data, shape)?;
        match &self.source {
            Some(source) => {
                let restored =
                    gpu_helpers::restore_class_preserving_value(source, host, BUILTIN_NAME)?;
                if runmat_accelerate_api::handle_is_explicit(source)
                    && !matches!(restored, Value::GpuTensor(_))
                {
                    return Err(normrnd_internal_error(
                        "normrnd: provider cannot preserve explicit gpuArray output residency",
                    ));
                }
                Ok(restored)
            }
            None => Ok(host),
        }
    }
}

fn ensure_extensions(args: &[Value]) -> crate::BuiltinResult<()> {
    if args.first().is_some_and(is_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_MU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args.get(1).is_some_and(is_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_SIGMA_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args.iter().skip(2).any(is_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_SIZE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args.iter().any(is_logical) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args.iter().skip(2).any(|value| {
        matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(&RESIDENT_SIZE_EXTENSION, BUILTIN_NAME)?;
    }
    Ok(())
}

fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

async fn parse_args(args: Vec<Value>) -> crate::BuiltinResult<(f64, f64, Vec<usize>)> {
    if args.len() < 2 {
        return Err(normrnd_error_with(
            &NORMRND_ERROR_INVALID_ARGUMENT,
            "normrnd: requires at least two arguments (mu, sigma)",
        ));
    }
    let mu = scalar_f64(&args[0]).await?;
    let sigma = scalar_f64(&args[1]).await?;
    let shape = parse_shape_args(&args[2..]).await?;
    Ok((mu, sigma, shape))
}

async fn scalar_f64(value: &Value) -> crate::BuiltinResult<f64> {
    let gathered = crate::gather_if_needed_async(value).await.map_err(|err| {
        normrnd_error_with(&NORMRND_ERROR_INVALID_ARGUMENT, format!("normrnd: {err}"))
    })?;
    let scalar = match &gathered {
        Value::Int(value) => Some(exact_integer_as_f64(value).ok_or_else(|| {
            normrnd_error_with(
                &NORMRND_ERROR_INVALID_ARGUMENT,
                "normrnd: integer parameter must be exactly representable as double",
            )
        })?),
        Value::Tensor(tensor)
            if tensor::is_scalar_tensor(tensor) && tensor.integer_storage().is_some() =>
        {
            let integer = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
                .ok_or_else(|| {
                    normrnd_error_with(
                        &NORMRND_ERROR_INTERNAL,
                        "normrnd: scalar integer storage is inconsistent with its shape",
                    )
                })?;
            Some(exact_integer_as_f64(&integer).ok_or_else(|| {
                normrnd_error_with(
                    &NORMRND_ERROR_INVALID_ARGUMENT,
                    "normrnd: integer parameter must be exactly representable as double",
                )
            })?)
        }
        other => tensor::scalar_f64_from_value_async(other)
            .await
            .map_err(|err| {
                normrnd_error_with(&NORMRND_ERROR_INVALID_ARGUMENT, format!("normrnd: {err}"))
            })?,
    };
    scalar.ok_or_else(|| {
        normrnd_error_with(
            &NORMRND_ERROR_INVALID_ARGUMENT,
            format!("normrnd: expected scalar parameter, got {value:?}"),
        )
    })
}

fn exact_integer_as_f64(value: &IntValue) -> Option<f64> {
    const MAX_EXACT_INTEGER: u64 = 1 << 53;
    match value {
        IntValue::I8(value) => Some(*value as f64),
        IntValue::I16(value) => Some(*value as f64),
        IntValue::I32(value) => Some(*value as f64),
        IntValue::I64(value) if value.unsigned_abs() <= MAX_EXACT_INTEGER => Some(*value as f64),
        IntValue::U8(value) => Some(*value as f64),
        IntValue::U16(value) => Some(*value as f64),
        IntValue::U32(value) => Some(*value as f64),
        IntValue::U64(value) if *value <= MAX_EXACT_INTEGER => Some(*value as f64),
        _ => None,
    }
}

async fn parse_shape_args(rest: &[Value]) -> crate::BuiltinResult<Vec<usize>> {
    if rest.is_empty() {
        return Ok(vec![1, 1]);
    }
    let mut dims: Vec<usize> = Vec::new();
    for arg in rest {
        match extract_dims(arg, "normrnd").await? {
            Some(d) => dims.extend(d),
            None => {
                return Err(normrnd_error_with(
                    &NORMRND_ERROR_INVALID_ARGUMENT,
                    format!("normrnd: invalid size argument: {arg:?}"),
                ))
            }
        }
    }
    Ok(normalize_dims(dims))
}

fn normalize_dims(dims: Vec<usize>) -> Vec<usize> {
    if dims.is_empty() {
        vec![0, 0]
    } else if dims.len() == 1 {
        vec![dims[0], dims[0]]
    } else {
        dims
    }
}

fn try_gpu_normrnd(
    output: &NormrndOutputPlan,
    mu: f64,
    sigma: f64,
    shape: &[usize],
) -> crate::BuiltinResult<Option<Value>> {
    let Some(source) = &output.source else {
        return Ok(None);
    };
    let Some(provider) = gpu_helpers::exact_provider_for_handle(source) else {
        return Err(normrnd_internal_error(
            "normrnd: resident input has no owning provider",
        ));
    };
    let expected_precision = if output.single {
        runmat_accelerate_api::ProviderPrecision::F32
    } else {
        runmat_accelerate_api::ProviderPrecision::F64
    };
    if provider.precision() != expected_precision {
        return Ok(None);
    }
    let source_metadata = gpu_helpers::snapshot_handle_metadata(source);
    let provider_result = provider.random_normrnd(mu, sigma, shape);
    gpu_helpers::restore_handle_metadata(source, &source_metadata);
    match provider_result {
        Ok(mut handle) => {
            let valid = !gpu_helpers::same_gpu_handle(source, &handle)
                && handle.shape == shape
                && handle.device_id == provider.device_id()
                && gpu_helpers::exact_provider_for_handle(&handle)
                    .is_some_and(|owner| std::ptr::eq(owner, provider))
                && runmat_accelerate_api::handle_storage(&handle)
                    == runmat_accelerate_api::GpuTensorStorage::Real
                && runmat_accelerate_api::handle_precision(&handle) == Some(expected_precision)
                && runmat_accelerate_api::handle_integer_type(&handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(&handle)
                && gpu_helpers::gpu_class_metadata_matches(
                    &handle,
                    Some(expected_precision),
                    None,
                    false,
                );
            if !valid {
                gpu_helpers::free_unprotected_exact_owner(&handle, &[source]);
                return Err(normrnd_internal_error(
                    "normrnd: provider returned a malformed, aliased, or foreign random result",
                ));
            }
            let len = tensor::element_count(shape);
            // Box-Muller emits two normals per two uniform samples.
            let uniform_count = len.saturating_add(1) / 2 * 2;
            random::skip_uniform(uniform_count, "normrnd")?;
            runmat_accelerate_api::set_handle_provenance(
                &mut handle,
                runmat_accelerate_api::handle_provenance(source)
                    .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic),
            );
            Ok(Some(gpu_helpers::resident_gpu_value(handle)))
        }
        Err(error)
            if error
                .chain()
                .any(|cause| cause.to_string() == "random_normrnd not supported by provider") =>
        {
            Ok(None)
        }
        Err(error) => Err(build_runtime_error(format!(
            "normrnd: provider random generation failed: {error}"
        ))
        .with_builtin(BUILTIN_NAME)
        .with_identifier("RunMat:normrnd:Internal")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::random;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn reset() {
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        tensor
    }

    #[test]
    fn normrnd_scalar_deterministic() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let result =
            block_on(normrnd_builtin(vec![Value::Num(0.0), Value::Num(1.0)])).expect("normrnd");
        let expected = random::expected_normal_scaled_sequence(0.0, 1.0, 1)[0];
        match result {
            Value::Num(v) => assert!((v - expected).abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn normrnd_matrix_dims() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let args = vec![
            Value::Num(5.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ];
        let result = block_on(normrnd_builtin(args)).expect("normrnd");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![3, 4]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn normrnd_size_vec() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let size = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let args = vec![Value::Num(0.0), Value::Num(1.0), Value::Tensor(size)];
        let result = block_on(normrnd_builtin(args)).expect("normrnd");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![3, 4]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn normrnd_reads_typed_integer_parameters_and_size_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let mu = poisoned_int_tensor(IntegerStorage::I16(vec![5]), vec![1, 1]);
        let sigma = poisoned_int_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]);
        let size = poisoned_int_tensor(IntegerStorage::U64(vec![2, 3]), vec![1, 2]);
        let result = block_on(normrnd_builtin(vec![
            Value::Tensor(mu),
            Value::Tensor(sigma),
            Value::Tensor(size),
        ]))
        .expect("normrnd");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![2, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn normrnd_integer_roles_gate_before_conversion_and_wide_values_reject() {
        let integer = Value::Int(IntValue::I16(0));
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(normrnd_builtin(vec![integer.clone(), Value::Num(1.0)]))
                .expect_err("integer mu must gate");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:NormrndIntegerMuExtension")
            );
        }
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let wide = poisoned_int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]);
        let error = block_on(normrnd_builtin(vec![Value::Tensor(wide), Value::Num(1.0)]))
            .expect_err("wide integer mu must reject");
        assert!(error.message.contains("exactly representable"));
    }

    #[test]
    fn normrnd_single_data_parameter_selects_single_output() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let mu = Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap());
        let out = block_on(normrnd_builtin(vec![mu, Value::Num(1.0)])).unwrap();
        let Value::Tensor(out) = out else {
            panic!("expected single tensor output");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::F32);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn normrnd_wgpu_integer_parameter_preserves_class_and_explicit_intent() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("wgpu provider");
        let mu = Tensor::new_integer(IntegerStorage::I16(vec![0]), vec![1, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &mu).expect("integer upload");
        let out = block_on(normrnd_builtin(vec![
            Value::GpuTensor(handle.clone()),
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        ]))
        .expect("normrnd");
        let Value::Tensor(host) = out else {
            panic!("F32 owner cannot relabel required double output");
        };
        assert_eq!(host.numeric_dtype(), NumericDType::F64);
        assert_eq!(host.shape, vec![2, 2]);
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let error = block_on(normrnd_builtin(vec![
            Value::GpuTensor(handle),
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        ]))
        .expect_err("explicit output class mismatch must reject");
        assert!(error.message.contains("cannot preserve explicit gpuArray"));
    }

    #[test]
    fn normrnd_rejects_negative_sigma() {
        let args = vec![Value::Num(0.0), Value::Num(-1.0)];
        let err = block_on(normrnd_builtin(args)).expect_err("negative sigma should error");
        assert_eq!(
            err.identifier(),
            NORMRND_ERROR_SIGMA_MUST_BE_NONNEGATIVE.identifier
        );
    }

    #[test]
    fn normrnd_distribution_mean_and_std() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let mu = 5.0_f64;
        let sigma = 2.0_f64;
        let n = 50_000_usize;
        let args = vec![
            Value::Num(mu),
            Value::Num(sigma),
            Value::Num(n as f64),
            Value::Num(1.0),
        ];
        let result = block_on(normrnd_builtin(args)).expect("normrnd");
        let data = match result {
            Value::Tensor(t) => t.materialize_f64(),
            other => panic!("expected tensor, got {other:?}"),
        };
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();
        assert!(
            (mean - mu).abs() / sigma.max(1.0) < 0.05,
            "sample mean {mean:.4} not within 5% tolerance of mu={mu}"
        );
        assert!(
            (std_dev - sigma).abs() / sigma < 0.05,
            "sample std {std_dev:.4} not within 5% of sigma={sigma}"
        );
    }
}
