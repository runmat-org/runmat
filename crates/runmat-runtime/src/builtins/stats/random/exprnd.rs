use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
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
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random;
use crate::builtins::common::tensor;
use crate::{gather_if_needed_async, BuiltinResult};

const BUILTIN_NAME: &str = "exprnd";

const EXPRND_INTEGER_MU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "exprnd-integer-mean",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "exprnd with a typed-integer mean is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExprndIntegerMeanExtension"),
};

const EXPRND_INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "exprnd-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "exprnd with typed-integer size arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExprndIntegerSizeExtension"),
};

pub const EXPRND_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [EXPRND_INTEGER_MU_EXTENSION, EXPRND_INTEGER_SIZE_EXTENSION];

const EXPRND_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "mu",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer means are gated and enter the floating exponential computation domain only after positivity validation.",
    },
    BuiltinIntegerInputCapability {
        name: "sz",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer size controls are gated and parsed exactly before allocation.",
    },
];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "r = exprnd(mu, sz1, sz2, ...)",
        inputs: &EXPRND_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed-integer mean and size arguments are separate RunMat-only extensions; resident mean fallback restores output only to the exact source owner and only when that owner physically supports the required output precision.",
    }];

const EXPRND_OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random sample array from exponential distribution.",
}];

const EXPRND_INPUTS_MU: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mu",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Exponential mean parameter (must be > 0).",
}];

const EXPRND_INPUTS_MU_SZ: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exponential mean parameter (must be > 0).",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Size scalar or size vector argument.",
    },
];

const EXPRND_INPUTS_MU_DIMS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exponential mean parameter (must be > 0).",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension extents for output shape.",
    },
];

const EXPRND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = exprnd(mu)",
        inputs: &EXPRND_INPUTS_MU,
        outputs: &EXPRND_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = exprnd(mu, sz)",
        inputs: &EXPRND_INPUTS_MU_SZ,
        outputs: &EXPRND_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = exprnd(mu, sz1, sz2, ...)",
        inputs: &EXPRND_INPUTS_MU_DIMS,
        outputs: &EXPRND_OUTPUT_R,
    },
];

const EXPRND_ERROR_MU_MUST_BE_POSITIVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXPRND.MU_MUST_BE_POSITIVE",
    identifier: Some("RunMat:exprnd:MuMustBePositive"),
    when: "mu is zero or negative.",
    message: "exprnd: mu must be greater than zero",
};

const EXPRND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXPRND.INVALID_ARGUMENT",
    identifier: Some("RunMat:exprnd:InvalidArgument"),
    when: "Input parameters or size arguments are missing or malformed.",
    message: "exprnd: invalid argument",
};

const EXPRND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXPRND.INTERNAL",
    identifier: Some("RunMat:exprnd:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "exprnd: internal operation failed",
};

const EXPRND_ERRORS: [BuiltinErrorDescriptor; 3] = [
    EXPRND_ERROR_MU_MUST_BE_POSITIVE,
    EXPRND_ERROR_INVALID_ARGUMENT,
    EXPRND_ERROR_INTERNAL,
];

pub const EXPRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EXPRND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EXPRND_ERRORS,
};

fn exprnd_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn exprnd_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    exprnd_error_with(error, error.message)
}

fn exprnd_internal_error(message: impl Into<String>) -> crate::RuntimeError {
    exprnd_error_with(&EXPRND_ERROR_INTERNAL, message)
}

fn exprnd_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let _ = args;
    Type::Unknown
}

#[runtime_builtin(
    name = "exprnd",
    category = "stats/random",
    summary = "Generate exponentially distributed random samples with mean `mu`.",
    keywords = "exprnd,exponential,random,distribution,statistics",
    type_resolver(exprnd_type),
    descriptor(crate::builtins::stats::random::exprnd::EXPRND_DESCRIPTOR),
    extensions(crate::builtins::stats::random::exprnd::EXPRND_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::random::exprnd::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::random::exprnd"
)]
async fn exprnd_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    ensure_output_arity()?;
    let integer_mu = args.first().is_some_and(is_typed_integer_value);
    let integer_size = args.iter().skip(1).any(is_typed_integer_value);
    if integer_mu {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXPRND_INTEGER_MU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if integer_size {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXPRND_INTEGER_SIZE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    reject_unsupported_classes(&args)?;
    let parsed = parse_args(args).await?;
    if parsed.mu.iter().any(|mu| !mu.is_finite() || *mu <= 0.0) {
        return Err(exprnd_error(&EXPRND_ERROR_MU_MUST_BE_POSITIVE));
    }
    if parsed.mu.len() == 1
        && parsed.output_precision == OutputPrecision::Double
        && parsed.gpu_source.is_none()
    {
        if let Some(value) = try_gpu_exponential(parsed.mu[0], &parsed.shape)? {
            return Ok(value);
        }
    }
    let len = tensor::element_count(&parsed.shape);
    let data = random::generate_exponential_array(&parsed.mu, len, BUILTIN_NAME)?;
    build_output(data, parsed)
}

fn ensure_output_arity() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: too many output arguments",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputPrecision {
    Double,
    Single,
}

struct ParsedExprnd {
    mu: Vec<f64>,
    shape: Vec<usize>,
    output_precision: OutputPrecision,
    gpu_source: Option<GpuTensorHandle>,
}

fn output_precision(value: &Value) -> OutputPrecision {
    match value {
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            OutputPrecision::Single
        }
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(ProviderPrecision::F32) =>
        {
            OutputPrecision::Single
        }
        _ => OutputPrecision::Double,
    }
}

fn build_output(data: Vec<f64>, parsed: ParsedExprnd) -> BuiltinResult<Value> {
    let output = match parsed.output_precision {
        OutputPrecision::Double => Tensor::new(data, parsed.shape),
        OutputPrecision::Single => Tensor::from_f32(
            data.into_iter().map(|value| value as f32).collect(),
            parsed.shape,
        ),
    }
    .map_err(|e| exprnd_internal_error(format!("exprnd: {e}")))?;
    if let Some(source) = parsed.gpu_source {
        let Some(provider) = runmat_accelerate_api::provider_for_handle(&source)
            .filter(|owner| owner.device_id() == source.device_id)
        else {
            return Ok(exprnd_host_output(output, parsed.output_precision));
        };
        let required_precision = match parsed.output_precision {
            OutputPrecision::Double => ProviderPrecision::F64,
            OutputPrecision::Single => ProviderPrecision::F32,
        };
        if provider.precision() != required_precision {
            return Ok(exprnd_host_output(output, parsed.output_precision));
        }
        let handle = gpu_helpers::upload_tensor(provider, &output)
            .map_err(|err| exprnd_internal_error(format!("exprnd: {err}")))?;
        if !valid_exprnd_gpu_output(
            &handle,
            &source,
            provider,
            &output.shape,
            required_precision,
        ) {
            free_exprnd_gpu_output(&handle, &source);
            return Err(exprnd_internal_error(
                "exprnd: provider upload returned malformed resident output",
            ));
        }
        return Ok(gpu_helpers::resident_gpu_value(handle));
    }
    Ok(exprnd_host_output(output, parsed.output_precision))
}

fn exprnd_host_output(output: Tensor, precision: OutputPrecision) -> Value {
    if precision == OutputPrecision::Double {
        tensor::tensor_into_value(output)
    } else {
        Value::Tensor(output)
    }
}

async fn parse_args(args: Vec<Value>) -> crate::BuiltinResult<ParsedExprnd> {
    if args.is_empty() {
        return Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: requires at least one argument (mu)",
        ));
    }
    let output_precision = output_precision(&args[0]);
    let gpu_source = match &args[0] {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let gathered = gather_if_needed_async(&args[0]).await.map_err(|err| {
        exprnd_error_with(&EXPRND_ERROR_INVALID_ARGUMENT, format!("exprnd: {err}"))
    })?;
    let mu_tensor = value_to_mu_tensor(gathered)?;
    let mu_shape = normalize_shape(mu_tensor.shape.clone());
    let mu = tensor::tensor_into_values_f64(mu_tensor);
    let shape = if args.len() == 1 {
        mu_shape.clone()
    } else {
        parse_shape_args(&args[1..]).await?
    };
    if mu.len() != 1 && mu_shape != shape {
        return Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: requested size must match nonscalar mu",
        ));
    }
    Ok(ParsedExprnd {
        mu,
        shape,
        output_precision,
        gpu_source,
    })
}

async fn parse_shape_args(rest: &[Value]) -> crate::BuiltinResult<Vec<usize>> {
    let mut dims = Vec::new();
    for value in rest {
        let gathered = gather_if_needed_async(value).await.map_err(|err| {
            exprnd_error_with(&EXPRND_ERROR_INVALID_ARGUMENT, format!("exprnd: {err}"))
        })?;
        dims.extend(dimensions_from_host_value(&gathered)?);
    }
    Ok(normalize_dims(dims))
}

fn normalize_dims(dims: Vec<usize>) -> Vec<usize> {
    if dims.is_empty() {
        vec![0, 0]
    } else if dims.len() == 1 {
        vec![dims[0], dims[0]]
    } else {
        normalize_shape(dims)
    }
}

fn normalize_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        shape = vec![1, 1];
    } else if shape.len() == 1 {
        shape.push(1);
    }
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn value_to_mu_tensor(value: Value) -> BuiltinResult<Tensor> {
    let result = match value {
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1]),
        Value::Int(value) => Tensor::new(vec![value.to_f64()], vec![1, 1]),
        Value::Tensor(tensor) => return Ok(tensor),
        other => {
            return Err(exprnd_error_with(
                &EXPRND_ERROR_INVALID_ARGUMENT,
                format!("exprnd: expected real single or double mu, got {other:?}"),
            ))
        }
    };
    result.map_err(|err| exprnd_internal_error(format!("exprnd: {err}")))
}

fn reject_unsupported_classes(args: &[Value]) -> BuiltinResult<()> {
    let Some(mu) = args.first() else {
        return Ok(());
    };
    let invalid = |value: &Value| {
        matches!(
            value,
            Value::Bool(_)
                | Value::LogicalArray(_)
                | Value::Complex(_, _)
                | Value::ComplexTensor(_)
        ) || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle) || runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved)
    };
    if invalid(mu) {
        return Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: mu must be a real single or double array",
        ));
    }
    if args.iter().skip(1).any(invalid) {
        return Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: size arguments must be real integer-valued arrays",
        ));
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn dimensions_from_host_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Num(value) => Ok(vec![floating_dimension(*value)?]),
        Value::Int(value) => Ok(vec![integer_dimension(value)?]),
        Value::Tensor(tensor) => {
            let vector = tensor.len() <= 1
                || tensor.shape.len() == 1
                || tensor.shape.first() == Some(&1)
                || tensor.shape.get(1) == Some(&1);
            if !vector {
                return Err(exprnd_error_with(
                    &EXPRND_ERROR_INVALID_ARGUMENT,
                    "exprnd: size must be a scalar or vector",
                ));
            }
            if let Some(storage) = tensor.integer_storage() {
                return (0..storage.len())
                    .map(|index| {
                        integer_dimension(&storage.value_at(index).expect("integer size value"))
                    })
                    .collect();
            }
            tensor
                .materialize_f64()
                .iter()
                .map(|value| floating_dimension(*value))
                .collect()
        }
        other => Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            format!("exprnd: invalid size argument: {other:?}"),
        )),
    }
}

fn integer_dimension(value: &IntValue) -> BuiltinResult<usize> {
    if value.to_f64() < 0.0 {
        return Ok(0);
    }
    value.try_to_usize().ok_or_else(|| {
        exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: size is outside the supported range",
        )
    })
}

fn floating_dimension(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: dimensions must be finite integers",
        ));
    }
    if value <= 0.0 {
        return Ok(0);
    }
    if value >= usize::MAX as f64 {
        return Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: size is outside the supported range",
        ));
    }
    Ok(value as usize)
}

fn try_gpu_exponential(mu: f64, shape: &[usize]) -> crate::BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };
    if provider.precision() != runmat_accelerate_api::ProviderPrecision::F64 {
        return Ok(None);
    }
    match provider.random_exponential(mu, shape) {
        Ok(handle) if valid_new_exprnd_gpu_output(&handle, provider, shape) => {
            let len = tensor::element_count(shape);
            random::skip_uniform(len, "exprnd")?;
            Ok(Some(Value::GpuTensor(handle)))
        }
        Ok(handle) => {
            free_exprnd_gpu_output_without_input(&handle);
            Err(exprnd_internal_error(
                "exprnd: provider random_exponential returned malformed output",
            ))
        }
        Err(err) if err.to_string().contains("random_exponential not supported") => Ok(None),
        Err(err) => Err(exprnd_internal_error(format!(
            "exprnd: provider random_exponential failed: {err}"
        ))),
    }
}

fn valid_new_exprnd_gpu_output(
    output: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
) -> bool {
    output.shape == shape
        && output.device_id == provider.device_id()
        && runmat_accelerate_api::handle_precision(output) == Some(ProviderPrecision::F64)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn valid_exprnd_gpu_output(
    output: &GpuTensorHandle,
    source: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
    precision: ProviderPrecision,
) -> bool {
    output.shape == shape
        && output.device_id == source.device_id
        && !(output.device_id == source.device_id && output.buffer_id == source.buffer_id)
        && runmat_accelerate_api::handle_precision(output) == Some(precision)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn free_exprnd_gpu_output(output: &GpuTensorHandle, source: &GpuTensorHandle) {
    if output.device_id == source.device_id && output.buffer_id == source.buffer_id {
        return;
    }
    free_exprnd_gpu_output_without_input(output);
}

fn free_exprnd_gpu_output_without_input(output: &GpuTensorHandle) {
    if let Some(owner) = runmat_accelerate_api::provider_for_handle(output)
        .filter(|owner| owner.device_id() == output.device_id)
    {
        let _ = owner.free(output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::random;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    struct MalformedRandomProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider,
    }

    impl runmat_accelerate_api::AccelProvider for MalformedRandomProvider {
        fn upload(
            &self,
            host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<GpuTensorHandle> {
            self.inner.upload(host)
        }

        fn download<'a>(
            &'a self,
            handle: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            self.inner.download(handle)
        }

        fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
            self.inner.free(handle)
        }

        fn device_info(&self) -> String {
            self.inner.device_info()
        }

        fn device_id(&self) -> u32 {
            self.inner.device_id()
        }

        fn precision(&self) -> ProviderPrecision {
            ProviderPrecision::F64
        }

        fn random_exponential(
            &self,
            _mu: f64,
            _shape: &[usize],
        ) -> anyhow::Result<GpuTensorHandle> {
            self.inner.upload(&runmat_accelerate_api::HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
        }
    }

    fn reset() {
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        tensor
    }

    #[test]
    fn exprnd_scalar_deterministic() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let result = block_on(exprnd_builtin(vec![Value::Num(2.0)])).expect("exprnd");
        let expected = random::expected_exponential_sequence(2.0, 1)[0];
        match result {
            Value::Num(v) => {
                assert!(v > 0.0);
                assert!((v - expected).abs() < 1e-12);
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn exprnd_matrix_dims() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let args = vec![Value::Num(1.0), Value::Num(3.0), Value::Num(4.0)];
        let result = block_on(exprnd_builtin(args)).expect("exprnd");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 4]);
                assert!(t.materialize_f64().iter().all(|&v| v > 0.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn exprnd_size_vec() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let size = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let args = vec![Value::Num(1.0), Value::Tensor(size)];
        let result = block_on(exprnd_builtin(args)).expect("exprnd");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![3, 4]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn exprnd_reads_typed_integer_mu_and_size_exactly() {
        let _guard = random::test_lock().lock().unwrap();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        reset();
        let mu = poisoned_int_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]);
        let size = poisoned_int_tensor(IntegerStorage::U64(vec![2, 3]), vec![1, 2]);
        let result =
            block_on(exprnd_builtin(vec![Value::Tensor(mu), Value::Tensor(size)])).expect("exprnd");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert!(t.materialize_f64().iter().all(|&v| v > 0.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn exprnd_rejects_negative_mu() {
        let args = vec![Value::Num(-1.0)];
        let err = block_on(exprnd_builtin(args)).expect_err("negative mu should error");
        assert_eq!(
            err.identifier(),
            EXPRND_ERROR_MU_MUST_BE_POSITIVE.identifier
        );
    }

    #[test]
    fn exprnd_rejects_zero_mu() {
        let args = vec![Value::Num(0.0)];
        let err = block_on(exprnd_builtin(args)).expect_err("zero mu should error");
        assert_eq!(
            err.identifier(),
            EXPRND_ERROR_MU_MUST_BE_POSITIVE.identifier
        );
    }

    #[test]
    fn exprnd_distribution_mean() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let mu = 3.0_f64;
        let n = 50_000_usize;
        let args = vec![Value::Num(mu), Value::Num(n as f64), Value::Num(1.0)];
        let result = block_on(exprnd_builtin(args)).expect("exprnd");
        let data = match result {
            Value::Tensor(t) => t.materialize_f64(),
            other => panic!("expected tensor, got {other:?}"),
        };
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        assert!(
            (mean - mu).abs() / mu < 0.05,
            "sample mean {mean:.4} not within 5% of mu={mu}"
        );
    }

    #[test]
    fn exprnd_array_mu_preserves_shape_and_native_single() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let mu = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = block_on(exprnd_builtin(vec![Value::Tensor(mu)])).expect("array mu");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor output");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        assert!(tensor.materialize_f64().iter().all(|value| *value > 0.0));
    }

    #[test]
    fn exprnd_array_mu_requires_matching_explicit_size() {
        let mu = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let error = block_on(exprnd_builtin(vec![
            Value::Tensor(mu),
            Value::Num(3.0),
            Value::Num(1.0),
        ]))
        .expect_err("shape mismatch");
        assert_eq!(error.identifier(), EXPRND_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn exprnd_normalizes_empty_and_trailing_singleton_sizes() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let before = random::snapshot().unwrap();
        let empty = block_on(exprnd_builtin(vec![
            Value::Num(1.0),
            Value::Num(-2.0),
            Value::Num(3.0),
        ]))
        .expect("negative size produces empty");
        let Value::Tensor(empty) = empty else {
            panic!("expected empty tensor");
        };
        assert_eq!(empty.shape, vec![0, 3]);
        assert_eq!(before.state, random::snapshot().unwrap().state);

        let output = block_on(exprnd_builtin(vec![
            Value::Num(1.0),
            Value::Num(3.0),
            Value::Num(1.0),
            Value::Num(1.0),
        ]))
        .expect("trailing singleton dimensions");
        let Value::Tensor(output) = output else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![3, 1]);
    }

    #[test]
    fn exprnd_rejects_logical_complex_and_excess_outputs_before_rng() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        for value in [Value::Bool(true), Value::Complex(1.0, 0.0)] {
            let before = random::snapshot().unwrap();
            let error = block_on(exprnd_builtin(vec![value])).expect_err("invalid class");
            assert_eq!(error.identifier(), EXPRND_ERROR_INVALID_ARGUMENT.identifier);
            assert_eq!(before.state, random::snapshot().unwrap().state);
        }
        let before = random::snapshot().unwrap();
        let _outputs = crate::output_count::push_output_count(Some(2));
        let error = block_on(exprnd_builtin(vec![Value::Num(1.0)])).expect_err("excess outputs");
        assert_eq!(error.identifier(), EXPRND_ERROR_INVALID_ARGUMENT.identifier);
        assert_eq!(before.state, random::snapshot().unwrap().state);
    }

    #[test]
    fn exprnd_zero_output_evaluation_still_advances_rng_once() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let before = random::snapshot().unwrap().state;
        let _outputs = crate::output_count::push_output_count(Some(0));
        let _ = block_on(exprnd_builtin(vec![Value::Num(1.0)])).expect("zero-output call");
        assert_ne!(before, random::snapshot().unwrap().state);
    }

    #[test]
    fn exprnd_rejects_malformed_provider_output_before_advancing_rng() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let _accel_guard = crate::builtins::common::test_support::accel_test_lock();
        let provider: &'static dyn runmat_accelerate_api::AccelProvider =
            Box::leak(Box::new(MalformedRandomProvider {
                inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            }));
        let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let before = random::snapshot().expect("rng snapshot").state;
        let error = block_on(exprnd_builtin(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        ]))
        .expect_err("malformed provider result");
        assert!(error.message().contains("malformed output"));
        assert_eq!(before, random::snapshot().expect("rng snapshot").state);
    }

    #[test]
    fn exprnd_integer_extensions_follow_compatibility_mode() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(exprnd_builtin(vec![Value::Int(
            runmat_builtins::IntValue::U16(2),
        )]))
        .expect_err("integer mu is gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:ExprndIntegerMeanExtension")
        );
        let error = block_on(exprnd_builtin(vec![
            Value::Num(1.0),
            Value::Int(runmat_builtins::IntValue::U16(2)),
        ]))
        .expect_err("integer size is gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:ExprndIntegerSizeExtension")
        );
    }

    #[test]
    fn exprnd_runmat_extension_accepts_all_integer_mean_classes() {
        let _guard = random::test_lock().lock().unwrap();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = vec![
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ];
        for storage in storages {
            reset();
            let mu = poisoned_int_tensor(storage, vec![1, 1]);
            let sample =
                block_on(exprnd_builtin(vec![Value::Tensor(mu)])).expect("integer mean extension");
            assert!(matches!(sample, Value::Num(value) if value > 0.0));
        }
    }

    #[test]
    fn exprnd_resident_single_mean_stays_host_when_owner_is_physically_f64() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let mu = Tensor::from_f32(vec![2.0, 3.0], vec![2, 1]).expect("single mu");
            let input = gpu_helpers::upload_tensor(provider, &mu).expect("upload single mu");
            let output =
                block_on(exprnd_builtin(vec![Value::GpuTensor(input)])).expect("resident exprnd");
            let Value::Tensor(output) = output else {
                panic!("expected host single output");
            };
            assert_eq!(output.shape, vec![2, 1]);
            assert_eq!(output.numeric_dtype(), NumericDType::F32);
            assert!(output.materialize_f64().iter().all(|value| *value > 0.0));
        });
    }
}
