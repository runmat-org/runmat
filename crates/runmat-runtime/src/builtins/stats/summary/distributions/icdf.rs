use super::*;
use super::*;

const INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "icdf-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "icdf with typed-integer probability or parameter input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IcdfIntegerInputExtension"),
};
const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "icdf-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "icdf with logical probability or parameter input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IcdfLogicalInputExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [INTEGER_INPUT_EXTENSION, LOGICAL_INPUT_EXTENSION];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "p",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer probability input is gated before provider access and checked for exact representation at the floating inverse-CDF boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "distribution parameters",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer named-distribution parameters are gated independently from logical input and checked before floating evaluation.",
    },
];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "x = icdf(name_or_fitted_distribution, p, integer_parameters?)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "RunMat-only integer/logical inputs enter the same checked floating boundary for named and fitted-distribution-object forms. The implemented named-family subset preserves documented single/double class selection and restores resident output to the exact owner only when that owner can physically preserve the required precision; this metadata does not claim unsupported distribution families are implemented.",
    }];

const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.icdf.INVALID_ARGUMENT",
        identifier: Some("RunMat:icdf:InvalidArgument"),
        when: ERROR_INVALID_ARGUMENT.when,
        message: "icdf: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.icdf.INTERNAL",
        identifier: Some("RunMat:icdf:Internal"),
        when: ERROR_INTERNAL.when,
        message: "icdf: internal error",
    },
];

pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ICDF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[derive(Clone, Copy)]
enum IcdfDistribution {
    Normal,
    StudentT,
    Weibull,
    ChiSquare,
    Binomial,
    Gamma,
    Exponential,
    Poisson,
    Uniform,
    Lognormal,
    Beta,
    F,
}

#[runtime_builtin(
    name = "icdf",
    category = "stats/summary",
    summary = "Evaluate inverse cumulative distribution functions for supported named families.",
    keywords = "icdf,inverse,cdf,normal,student t,weibull,binomial,chi-square,gamma,exponential,poisson,uniform,lognormal,beta,f,statistics",
    type_resolver(super::icdf_type),
    descriptor(self::DESCRIPTOR),
    extensions(self::EXTENSIONS),
    integer_capabilities(self::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::distributions::icdf"
)]
pub(crate) async fn icdf_builtin(name: Value, p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if matches!(name, Value::Object(_)) {
        if !rest.is_empty() {
            return Err(super::normal_error(
                "icdf",
                "icdf: fitted distribution object form accepts exactly two inputs",
            ));
        }
        let admission = IcdfAdmission::new(&p, &[])?;
        let p = gather_input(p).await?;
        ensure_exact_integer_boundary(&p, "p")?;
        let output =
            crate::builtins::stats::summary::fitdist::icdf_probability_distribution(name, p)
                .await?;
        return admission.finish(output);
    }

    let distribution = parse_distribution_name(&name)?;
    let admission = IcdfAdmission::new(&p, &rest)?;
    let p = gather_input(p).await?;
    let mut gathered_rest = Vec::with_capacity(rest.len());
    for value in rest {
        gathered_rest.push(gather_input(value).await?);
    }
    for (label, value) in
        std::iter::once(("p", &p)).chain(gathered_rest.iter().map(|value| ("parameter", value)))
    {
        ensure_exact_integer_boundary(value, label)?;
    }
    let output = match distribution {
        IcdfDistribution::Normal => normal_icdf(p, gathered_rest).await,
        IcdfDistribution::StudentT => student_t_icdf(p, gathered_rest).await,
        IcdfDistribution::Weibull => weibull_icdf(p, gathered_rest).await,
        IcdfDistribution::ChiSquare => chi_square_icdf(p, gathered_rest).await,
        IcdfDistribution::Binomial => binomial_icdf(p, gathered_rest).await,
        IcdfDistribution::Gamma => gamma_icdf(p, gathered_rest).await,
        IcdfDistribution::Exponential => exponential_icdf(p, gathered_rest).await,
        IcdfDistribution::Poisson => poisson_icdf(p, gathered_rest).await,
        IcdfDistribution::Uniform => uniform_icdf(p, gathered_rest).await,
        IcdfDistribution::Lognormal => lognormal_icdf(p, gathered_rest).await,
        IcdfDistribution::Beta => beta_icdf(p, gathered_rest).await,
        IcdfDistribution::F => f_icdf(p, gathered_rest).await,
    }?;
    admission.finish(output)
}

#[derive(Clone, Copy)]
enum OutputPrecision {
    Double,
    Single,
}

struct IcdfAdmission {
    precision: OutputPrecision,
    gpu_source: Option<GpuTensorHandle>,
}

impl IcdfAdmission {
    fn new(p: &Value, rest: &[Value]) -> BuiltinResult<Self> {
        let values = std::iter::once(p).chain(rest.iter());
        if values.clone().any(is_typed_integer_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTEGER_INPUT_EXTENSION,
                "icdf",
            )?;
        }
        if values.clone().any(is_logical_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &LOGICAL_INPUT_EXTENSION,
                "icdf",
            )?;
        }
        let precision = if values.clone().any(is_single_value) {
            OutputPrecision::Single
        } else {
            OutputPrecision::Double
        };
        let mut gpu_source: Option<GpuTensorHandle> = None;
        let mut owner: Option<&'static dyn runmat_accelerate_api::AccelProvider> = None;
        for handle in values.filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        }) {
            let current = gpu_helpers::exact_provider_for_handle(handle).ok_or_else(|| {
                super::normal_error("icdf", "icdf: resident input owner is unavailable")
            })?;
            if owner
                .is_some_and(|owner| !runmat_accelerate_api::same_provider_instance(owner, current))
            {
                return Err(super::normal_error(
                    "icdf",
                    "icdf: resident inputs must share one exact provider",
                ));
            }
            owner = Some(current);
            gpu_source.get_or_insert_with(|| handle.clone());
        }
        Ok(Self {
            precision,
            gpu_source,
        })
    }

    fn finish(self, output: Value) -> BuiltinResult<Value> {
        if self.gpu_source.is_none() && matches!(self.precision, OutputPrecision::Double) {
            return Ok(output);
        }
        let tensor = match output {
            Value::Num(value) => Tensor::new(vec![value], vec![1, 1]),
            Value::Tensor(tensor) => Ok(tensor),
            other => {
                return Err(super::normal_error(
                    "icdf",
                    format!("icdf: unexpected numeric output {other:?}"),
                ))
            }
        }
        .map_err(|error| super::normal_error("icdf", format!("icdf: {error}")))?;
        let tensor = match self.precision {
            OutputPrecision::Double => tensor,
            OutputPrecision::Single => Tensor::from_f32(
                tensor
                    .materialize_f64()
                    .into_iter()
                    .map(|value| value as f32)
                    .collect(),
                tensor.shape,
            )
            .map_err(|error| super::normal_error("icdf", format!("icdf: {error}")))?,
        };
        let Some(source) = self.gpu_source else {
            return Ok(Value::Tensor(tensor));
        };
        gpu_helpers::restore_class_preserving_value(&source, Value::Tensor(tensor), "icdf")
            .map_err(|error| super::normal_error("icdf", format!("icdf: {error}")))
    }
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

async fn gather_input(value: Value) -> BuiltinResult<Value> {
    let Value::GpuTensor(handle) = &value else {
        return Ok(value);
    };
    let owner = gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| super::normal_error("icdf", "icdf: resident input owner is unavailable"))?;
    gpu_helpers::download_value_preserving_residency_async(owner, handle)
        .await
        .map_err(|error| super::normal_error("icdf", format!("icdf: {error}")))
}

fn ensure_exact_integer_boundary(value: &Value, label: &str) -> BuiltinResult<()> {
    let tensor = match value {
        Value::Int(integer) => {
            let exact = match integer {
                runmat_value::IntValue::I8(value) => i128::from(*value),
                runmat_value::IntValue::I16(value) => i128::from(*value),
                runmat_value::IntValue::I32(value) => i128::from(*value),
                runmat_value::IntValue::I64(value) => i128::from(*value),
                runmat_value::IntValue::U8(value) => i128::from(*value),
                runmat_value::IntValue::U16(value) => i128::from(*value),
                runmat_value::IntValue::U32(value) => i128::from(*value),
                runmat_value::IntValue::U64(value) => i128::from(*value),
            };
            return ensure_exact_i128(exact, label);
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => tensor,
        _ => return Ok(()),
    };
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
        ensure_exact_i128(exact, label)?;
    }
    Ok(())
}

fn ensure_exact_i128(exact: i128, label: &str) -> BuiltinResult<()> {
    const MAX_EXACT: i128 = 1_i128 << 53;
    if !(-MAX_EXACT..=MAX_EXACT).contains(&exact) {
        return Err(super::normal_error(
            "icdf",
            format!("icdf: integer {label} values must be exactly representable as double"),
        ));
    }
    Ok(())
}

fn parse_distribution_name(value: &Value) -> BuiltinResult<IcdfDistribution> {
    let Some(keyword) = crate::builtins::common::random_args::keyword_of(value) else {
        return Err(super::normal_error(
            "icdf",
            "icdf: distribution name must be a string scalar",
        ));
    };
    let normalized = keyword
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    match normalized.as_str() {
        "normal" | "norm" | "gaussian" => Ok(IcdfDistribution::Normal),
        "t" | "tdistribution" | "studentt" | "student" => Ok(IcdfDistribution::StudentT),
        "weibull" | "wbl" => Ok(IcdfDistribution::Weibull),
        "chisquare" | "chi2" | "chisquared" => Ok(IcdfDistribution::ChiSquare),
        "binomial" | "bino" => Ok(IcdfDistribution::Binomial),
        "gamma" | "gam" => Ok(IcdfDistribution::Gamma),
        "exponential" | "exp" => Ok(IcdfDistribution::Exponential),
        "poisson" | "poiss" => Ok(IcdfDistribution::Poisson),
        "uniform" | "unif" => Ok(IcdfDistribution::Uniform),
        "lognormal" | "logn" => Ok(IcdfDistribution::Lognormal),
        "beta" => Ok(IcdfDistribution::Beta),
        "f" | "fdist" | "fdistribution" => Ok(IcdfDistribution::F),
        _ => Err(super::normal_error(
            "icdf",
            format!("icdf: unsupported distribution '{keyword}'"),
        )),
    }
}

async fn normal_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = super::normal_args("icdf", p, rest, false).await?;
    let data = args
        .x
        .iter()
        .zip(args.mu.iter())
        .zip(args.sigma.iter())
        .map(|((p, mu), sigma)| super::norminv_scalar(*p, *mu, *sigma))
        .collect();
    super::finish(args.shape, data)
}

async fn student_t_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = super::t_args("icdf", p, rest).await?;
    let data = args
        .x
        .iter()
        .zip(args.nu.iter())
        .map(|(p, nu)| distribution_math::student_t_inv(*p, *nu))
        .collect();
    super::finish(args.shape, data)
}

async fn weibull_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = super::three_args("icdf", p, rest, Some((1.0, 1.0))).await?;
    let data = args
        .x
        .iter()
        .zip(args.a.iter())
        .zip(args.b.iter())
        .map(|((p, a), b)| super::wblinv_scalar(*p, *a, *b))
        .collect();
    super::finish(args.shape, data)
}

async fn chi_square_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = super::t_args("icdf", p, rest).await?;
    let data = args
        .x
        .iter()
        .zip(args.nu.iter())
        .map(|(p, nu)| super::chi2inv_scalar(*p, *nu))
        .collect();
    super::finish(args.shape, data)
}

async fn binomial_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = super::three_args("icdf", p, rest, None).await?;
    let data = args
        .x
        .iter()
        .zip(args.a.iter())
        .zip(args.b.iter())
        .map(|((probability, n), success_probability)| {
            super::binoinv_scalar(*probability, *n, *success_probability)
        })
        .collect();
    super::finish(args.shape, data)
}

async fn gamma_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = super::three_args("icdf", p, rest, None).await?;
    let data = args
        .x
        .iter()
        .zip(args.a.iter())
        .zip(args.b.iter())
        .map(|((p, shape), scale)| super::gaminv_scalar(*p, *shape, *scale))
        .collect();
    super::finish(args.shape, data)
}

async fn exponential_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = one_param_args("Exponential", p, rest).await?;
    let data = args
        .x
        .iter()
        .zip(args.param.iter())
        .map(|(p, mu)| super::expinv_scalar(*p, *mu))
        .collect();
    super::finish(args.shape, data)
}

async fn poisson_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = one_param_args("Poisson", p, rest).await?;
    let data = args
        .x
        .iter()
        .zip(args.param.iter())
        .map(|(p, lambda)| super::poissinv_scalar(*p, *lambda))
        .collect();
    super::finish(args.shape, data)
}

async fn uniform_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = super::three_args("icdf", p, rest, Some((0.0, 1.0))).await?;
    let data = args
        .x
        .iter()
        .zip(args.a.iter())
        .zip(args.b.iter())
        .map(|((p, a), b)| super::unifinv_scalar(*p, *a, *b))
        .collect();
    super::finish(args.shape, data)
}

async fn lognormal_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = super::normal_args("icdf", p, rest, false).await?;
    let data = args
        .x
        .iter()
        .zip(args.mu.iter())
        .zip(args.sigma.iter())
        .map(|((p, mu), sigma)| super::logninv_scalar(*p, *mu, *sigma))
        .collect();
    super::finish(args.shape, data)
}

async fn beta_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = super::three_args("icdf", p, rest, None).await?;
    let data = args
        .x
        .iter()
        .zip(args.a.iter())
        .zip(args.b.iter())
        .map(|((p, a), b)| super::betainv_scalar(*p, *a, *b))
        .collect();
    super::finish(args.shape, data)
}

async fn f_icdf(p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = super::three_args("icdf", p, rest, None).await?;
    let data = args
        .x
        .iter()
        .zip(args.a.iter())
        .zip(args.b.iter())
        .map(|((p, v1), v2)| super::finv_scalar(*p, *v1, *v2))
        .collect();
    super::finish(args.shape, data)
}

struct OneParamArgs {
    x: Vec<f64>,
    param: Vec<f64>,
    shape: Vec<usize>,
}

async fn one_param_args(
    distribution: &str,
    p: Value,
    rest: Vec<Value>,
) -> BuiltinResult<OneParamArgs> {
    if rest.len() != 1 {
        return Err(super::normal_error(
            "icdf",
            format!("icdf: {distribution} distribution expects one parameter"),
        ));
    }
    let x = super::value_to_tensor("icdf", p).await?;
    let param = super::value_to_tensor("icdf", rest[0].clone()).await?;
    let (x, param, shape) = super::broadcast_pair("icdf", &x, &param)?;
    Ok(OneParamArgs { x, param, shape })
}
