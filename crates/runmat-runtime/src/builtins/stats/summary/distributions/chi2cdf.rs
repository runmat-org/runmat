use super::*;
use super::*;
normal_descriptor!("chi2cdf", CHI2CDF_SIGNATURES);

const INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "chi2cdf-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "chi2cdf with typed-integer evaluation points is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Chi2cdfIntegerXExtension"),
};
const INTEGER_NU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "chi2cdf-integer-degrees-of-freedom",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "chi2cdf with typed-integer degrees of freedom is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Chi2cdfIntegerDegreesOfFreedomExtension"),
};
const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "chi2cdf-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "chi2cdf with logical numeric input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Chi2cdfLogicalInputExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    INTEGER_X_EXTENSION,
    INTEGER_NU_EXTENSION,
    LOGICAL_INPUT_EXTENSION,
];

const INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer evaluation points are outside the documented single/double input domain and cross a checked floating boundary only in RunMat mode.",
    }];
const INTEGER_NU_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "nu",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer degrees of freedom are outside the documented single/double input domain and cross a checked floating boundary only in RunMat mode.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "p = chi2cdf(integer_x, nu, ...)",
        inputs: &INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "RunMat-only integer x is checked for exact binary64 representation before CDF evaluation; documented single input selects single output and documented resident calls return to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "p = chi2cdf(x, integer_nu, ...)",
        inputs: &INTEGER_NU_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "RunMat-only integer nu is checked for exact binary64 representation before CDF evaluation; documented single input selects single output and documented resident calls return to the owning provider.",
    },
];

#[runtime_builtin(
    name = "chi2cdf",
    category = "stats/summary",
    summary = "Evaluate the chi-square cumulative distribution function.",
    keywords = "chi2cdf,chi-square,cdf,upper tail,statistics,distribution",
    type_resolver(super::normal_type),
    descriptor(self::DESCRIPTOR),
    extensions(self::EXTENSIONS),
    integer_capabilities(self::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::distributions::chi2cdf"
)]
pub(crate) async fn chi2cdf_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = parse_args(value, rest).await?;
    let data = args
        .x
        .iter()
        .zip(args.nu.iter())
        .map(|(x, nu)| super::chi2cdf_scalar(*x, *nu, args.upper))
        .collect();
    build_output(data, args.shape, args.precision, args.gpu_source)
}

#[derive(Clone, Copy)]
enum OutputPrecision {
    Double,
    Single,
}

struct Args {
    x: Vec<f64>,
    nu: Vec<f64>,
    shape: Vec<usize>,
    upper: bool,
    precision: OutputPrecision,
    gpu_source: Option<GpuTensorHandle>,
}

async fn parse_args(value: Value, mut rest: Vec<Value>) -> BuiltinResult<Args> {
    let upper = rest
        .last()
        .and_then(crate::builtins::common::random_args::keyword_of)
        .is_some_and(|keyword| keyword.eq_ignore_ascii_case("upper"));
    if upper {
        rest.pop();
    }
    if rest.len() != 1 {
        return Err(super::normal_error(
            "chi2cdf",
            "chi2cdf: expected x and nu, optionally followed by 'upper'",
        ));
    }
    let nu_value = rest.pop().expect("one nu argument");
    ensure_extensions(&value, &nu_value)?;
    let precision = if is_single(&value) || is_single(&nu_value) {
        OutputPrecision::Single
    } else {
        OutputPrecision::Double
    };
    let gpu_source = [&value, &nu_value]
        .into_iter()
        .find_map(|input| match input {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        });
    let x = gather_numeric(value).await?;
    let nu = gather_numeric(nu_value).await?;
    ensure_exact_integer_boundary(&x, "x")?;
    ensure_exact_integer_boundary(&nu, "nu")?;
    let (x, nu, shape) = same_size_or_scalar(&x, &nu)?;
    Ok(Args {
        x,
        nu,
        shape,
        upper,
        precision,
        gpu_source,
    })
}

fn ensure_extensions(x: &Value, nu: &Value) -> BuiltinResult<()> {
    if is_typed_integer(x) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_X_EXTENSION, "chi2cdf")?;
    }
    if is_typed_integer(nu) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_NU_EXTENSION, "chi2cdf")?;
    }
    if is_logical(x) || is_logical(nu) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOGICAL_INPUT_EXTENSION,
            "chi2cdf",
        )?;
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

fn is_single(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle) == Some(ProviderPrecision::F32))
}

async fn gather_numeric(value: Value) -> BuiltinResult<Tensor> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(|error| super::normal_error("chi2cdf", format!("chi2cdf: {error}")))?;
    tensor::value_into_tensor_for("chi2cdf", host)
        .map_err(|error| super::normal_error("chi2cdf", format!("chi2cdf: {error}")))
}

fn ensure_exact_integer_boundary(tensor: &Tensor, label: &str) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none() {
        return Ok(());
    }
    const MAX_EXACT: i128 = 1_i128 << 53;
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
        if !(-MAX_EXACT..=MAX_EXACT).contains(&exact) {
            return Err(super::normal_error(
                "chi2cdf",
                format!("chi2cdf: integer {label} values must be exactly representable as double"),
            ));
        }
    }
    Ok(())
}

fn same_size_or_scalar(x: &Tensor, nu: &Tensor) -> BuiltinResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let x_scalar = x.len() == 1;
    let nu_scalar = nu.len() == 1;
    let shape = if x_scalar && !nu_scalar {
        nu.shape.clone()
    } else if nu_scalar || x.shape == nu.shape {
        x.shape.clone()
    } else {
        return Err(super::normal_error(
            "chi2cdf",
            "chi2cdf: x and nu must have the same size unless one input is scalar",
        ));
    };
    let len = shape.iter().try_fold(1usize, |count, extent| {
        count.checked_mul(*extent).ok_or_else(|| {
            super::normal_error("chi2cdf", "chi2cdf: output shape exceeds platform limits")
        })
    })?;
    let x_values = tensor::tensor_values_f64(x);
    let nu_values = tensor::tensor_values_f64(nu);
    let x = if x_scalar {
        vec![x_values[0]; len]
    } else {
        x_values
    };
    let nu = if nu_scalar {
        vec![nu_values[0]; len]
    } else {
        nu_values
    };
    Ok((x, nu, shape))
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
    .map_err(|error| super::normal_error("chi2cdf", format!("chi2cdf: {error}")))?;
    if let Some(source) = gpu_source {
        let provider = runmat_accelerate_api::provider_for_handle(&source).ok_or_else(|| {
            super::normal_error(
                "chi2cdf",
                "chi2cdf: no acceleration provider registered for GPU output",
            )
        })?;
        let mut handle = gpu_helpers::upload_tensor_protecting_typed(provider, &tensor, &[&source])
            .map_err(|error| {
                crate::runtime_error::with_provider_execution_source(
                    super::normal_error("chi2cdf", format!("chi2cdf: {error}")),
                    runmat_accelerate_api::ProviderExecutionError::from_source(error),
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
