use super::*;
use super::*;
normal_descriptor!("binocdf", BINOCDF_SIGNATURES);

const INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "binocdf-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "binocdf with typed-integer evaluation points is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BinocdfIntegerXExtension"),
};

const INTEGER_N_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "binocdf-integer-trials",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "binocdf with typed-integer trial counts is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BinocdfIntegerTrialsExtension"),
};

const INTEGER_P_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "binocdf-integer-probability",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "binocdf with typed-integer probabilities is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BinocdfIntegerProbabilityExtension"),
};

const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "binocdf-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "binocdf with logical numeric inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BinocdfLogicalInputExtension"),
};

pub const EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    INTEGER_X_EXTENSION,
    INTEGER_N_EXTENSION,
    INTEGER_P_EXTENSION,
    LOGICAL_INPUT_EXTENSION,
];

const INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer evaluation points are gated by binocdf-integer-x and must be exactly representable at the floating binomial-CDF boundary.",
    }];

const INTEGER_N_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer trial counts are gated by binocdf-integer-trials and must be exactly representable at the documented floating computation boundary.",
    }];

const INTEGER_P_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "p",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer probabilities are gated by binocdf-integer-probability; only values in the ordinary probability domain can produce non-NaN results.",
    }];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = binocdf(x, n, p) with integer x",
        inputs: &INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only typed-integer x values produce double probabilities unless a documented single input makes the output single; resident inputs use host fallback and the result is re-uploaded to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = binocdf(x, n, p) with integer n",
        inputs: &INTEGER_N_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only typed-integer n values produce double probabilities unless a documented single input makes the output single, after an exact-representability check at the binary64 boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = binocdf(x, n, p) with integer p",
        inputs: &INTEGER_P_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only typed-integer p values produce double probabilities unless a documented single input makes the output single; resident inputs use host fallback and preserve output residency.",
    },
];

#[runtime_builtin(
    name = "binocdf",
    category = "stats/summary",
    summary = "Evaluate the binomial cumulative distribution function.",
    keywords = "binocdf,binomial,cdf,upper tail,statistics,distribution",
    type_resolver(super::normal_type),
    descriptor(self::DESCRIPTOR),
    extensions(self::EXTENSIONS),
    integer_capabilities(self::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::distributions::binocdf"
)]
pub(crate) async fn binocdf_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = parse_args(value, rest).await?;
    let data = args
        .x
        .iter()
        .zip(args.a.iter())
        .zip(args.b.iter())
        .map(|((x, n), p)| super::binocdf_scalar(*x, *n, *p, args.upper))
        .collect();
    build_output(data, args.shape, args.output_precision, args.gpu_source)
}

#[derive(Clone, Copy)]
enum OutputPrecision {
    Double,
    Single,
}

struct BinocdfArgs {
    x: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    shape: Vec<usize>,
    upper: bool,
    output_precision: OutputPrecision,
    gpu_source: Option<GpuTensorHandle>,
}

async fn parse_args(value: Value, mut rest: Vec<Value>) -> BuiltinResult<BinocdfArgs> {
    let upper = rest
        .last()
        .and_then(crate::builtins::common::random_args::keyword_of)
        .is_some_and(|keyword| keyword.eq_ignore_ascii_case("upper"));
    if upper {
        rest.pop();
    }
    if rest.len() != 2 {
        return Err(super::normal_error(
            "binocdf",
            "binocdf: expected x, n, and p arguments",
        ));
    }
    let inputs = [&value, &rest[0], &rest[1]];
    ensure_extensions(inputs)?;
    let output_precision = if inputs.iter().any(|value| is_single_value(value)) {
        OutputPrecision::Single
    } else {
        OutputPrecision::Double
    };
    let gpu_source = inputs.iter().find_map(|value| match value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    });
    let x = gather_numeric(value).await?;
    let n = gather_numeric(rest.remove(0)).await?;
    let p = gather_numeric(rest.remove(0)).await?;
    ensure_exact_integer_boundary(&x, "x")?;
    ensure_exact_integer_boundary(&n, "n")?;
    ensure_exact_integer_boundary(&p, "p")?;
    let (mut broadcasted, shape) = super::broadcast_tensors("binocdf", &[&x, &n, &p])?;
    Ok(BinocdfArgs {
        x: broadcasted.remove(0),
        a: broadcasted.remove(0),
        b: broadcasted.remove(0),
        shape,
        upper,
        output_precision,
        gpu_source,
    })
}

fn ensure_extensions(inputs: [&Value; 3]) -> BuiltinResult<()> {
    for (value, extension) in inputs.into_iter().zip([
        &INTEGER_X_EXTENSION,
        &INTEGER_N_EXTENSION,
        &INTEGER_P_EXTENSION,
    ]) {
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(extension, "binocdf")?;
        }
    }
    if inputs.into_iter().any(is_logical_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOGICAL_INPUT_EXTENSION,
            "binocdf",
        )?;
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn is_single_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle) == Some(ProviderPrecision::F32))
}

async fn gather_numeric(value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| super::normal_error("binocdf", format!("binocdf: {err}")))?;
    tensor::value_into_tensor_for("binocdf", gathered)
        .map_err(|err| super::normal_error("binocdf", format!("binocdf: {err}")))
}

fn ensure_exact_integer_boundary(tensor: &Tensor, name: &str) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none() {
        return Ok(());
    }
    const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
    for index in 0..tensor.len() {
        let exact = match tensor.numeric_value_at(index) {
            Some(NumericScalar::I8(value)) => i128::from(value),
            Some(NumericScalar::I16(value)) => i128::from(value),
            Some(NumericScalar::I32(value)) => i128::from(value),
            Some(NumericScalar::I64(value)) => i128::from(value),
            Some(NumericScalar::U8(value)) => i128::from(value),
            Some(NumericScalar::U16(value)) => i128::from(value),
            Some(NumericScalar::U32(value)) => i128::from(value),
            Some(NumericScalar::U64(value)) => i128::from(value),
            _ => continue,
        };
        if !(-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&exact) {
            return Err(super::normal_error(
                "binocdf",
                format!("binocdf: integer {name} values must be exactly representable as double"),
            ));
        }
    }
    Ok(())
}

fn build_output(
    data: Vec<f64>,
    shape: Vec<usize>,
    precision: OutputPrecision,
    gpu_source: Option<GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let tensor = match precision {
        OutputPrecision::Double => Tensor::new(data, shape),
        OutputPrecision::Single => {
            Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
        }
    }
    .map_err(|err| super::normal_error("binocdf", format!("binocdf: {err}")))?;
    if let Some(source) = gpu_source {
        let provider = runmat_accelerate_api::provider_for_handle(&source).ok_or_else(|| {
            super::normal_error(
                "binocdf",
                "binocdf: no acceleration provider registered for GPU output",
            )
        })?;
        let mut handle = gpu_helpers::upload_tensor_protecting_typed(provider, &tensor, &[&source])
            .map_err(|err| {
                crate::runtime_error::with_provider_execution_source(
                    super::normal_error("binocdf", format!("binocdf: {err}")),
                    runmat_accelerate_api::ProviderExecutionError::from_source(err),
                )
            })?;
        gpu_helpers::propagate_output_provenance(&mut handle, [&source])?;
        return gpu_helpers::resident_gpu_value(handle);
    }
    match precision {
        OutputPrecision::Double => Ok(tensor::tensor_into_value(tensor)),
        OutputPrecision::Single => Ok(Value::Tensor(tensor)),
    }
}
