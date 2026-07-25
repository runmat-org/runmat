//! Fitted probability distribution objects and object-aware distribution methods.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ObjectInstance, ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast;
use crate::builtins::common::random;
use crate::builtins::common::random_args::{extract_dims, keyword_of};
use crate::builtins::common::tensor;
use crate::builtins::math::elementwise::gammaln::gammaln_nonnegative_scalar;
use crate::builtins::stats::summary::distribution_math;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const FITDIST_NAME: &str = "fitdist";
const PDF_NAME: &str = "pdf";
const CDF_NAME: &str = "cdf";
const RANDOM_NAME: &str = "random";
const PROBABILITY_DISTRIBUTION_CLASS: &str = "ProbabilityDistribution";
const MIN_POSITIVE: f64 = 1.0e-12;

const OUTPUT_PD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "pd",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Fitted probability distribution object.",
};

const OUTPUT_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Distribution function values.",
};

const OUTPUT_R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random samples.",
};

const INPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample data or evaluation points.",
};

const INPUT_DIST: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "distname",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Distribution name.",
};

const INPUT_PD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "pd",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "ProbabilityDistribution object returned by fitdist.",
};

const INPUT_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Probability values.",
};

const INPUT_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "NameValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options.",
};

const INPUT_SZ: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Output size.",
};

const FITDIST_INPUTS: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_DIST];
const FITDIST_INPUTS_OPTIONS: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_DIST, INPUT_OPTIONS];
const FITDIST_OUTPUTS: [BuiltinParamDescriptor; 1] = [OUTPUT_PD];
const PDF_INPUTS: [BuiltinParamDescriptor; 2] = [INPUT_PD, INPUT_X];
const PDF_NAME_INPUTS: [BuiltinParamDescriptor; 3] = [INPUT_DIST, INPUT_X, INPUT_OPTIONS];
const CDF_INPUTS: [BuiltinParamDescriptor; 2] = [INPUT_PD, INPUT_X];
const CDF_NAME_INPUTS: [BuiltinParamDescriptor; 3] = [INPUT_DIST, INPUT_X, INPUT_OPTIONS];
const ICDF_INPUTS: [BuiltinParamDescriptor; 2] = [INPUT_PD, INPUT_P];
const RANDOM_INPUTS: [BuiltinParamDescriptor; 1] = [INPUT_PD];
const RANDOM_INPUTS_SIZE: [BuiltinParamDescriptor; 2] = [INPUT_PD, INPUT_SZ];
const RANDOM_NAME_INPUTS: [BuiltinParamDescriptor; 2] = [INPUT_DIST, INPUT_OPTIONS];
const DIST_OUTPUTS: [BuiltinParamDescriptor; 1] = [OUTPUT_Y];
const RANDOM_OUTPUTS: [BuiltinParamDescriptor; 1] = [OUTPUT_R];

const FITDIST_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "pd = fitdist(x, distname)",
        inputs: &FITDIST_INPUTS,
        outputs: &FITDIST_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "pd = fitdist(x, distname, Name, Value)",
        inputs: &FITDIST_INPUTS_OPTIONS,
        outputs: &FITDIST_OUTPUTS,
    },
];

const PDF_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "y = pdf(pd, x)",
        inputs: &PDF_INPUTS,
        outputs: &DIST_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "y = pdf(distname, x, params)",
        inputs: &PDF_NAME_INPUTS,
        outputs: &DIST_OUTPUTS,
    },
];

const CDF_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "p = cdf(pd, x)",
        inputs: &CDF_INPUTS,
        outputs: &DIST_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "p = cdf(distname, x, params)",
        inputs: &CDF_NAME_INPUTS,
        outputs: &DIST_OUTPUTS,
    },
];

pub(crate) const ICDF_OBJECT_SIGNATURE: BuiltinSignatureDescriptor = BuiltinSignatureDescriptor {
    label: "x = icdf(pd, p)",
    inputs: &ICDF_INPUTS,
    outputs: &DIST_OUTPUTS,
};

const RANDOM_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = random(pd)",
        inputs: &RANDOM_INPUTS,
        outputs: &RANDOM_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "r = random(pd, sz)",
        inputs: &RANDOM_INPUTS_SIZE,
        outputs: &RANDOM_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "r = random(distname, params, sz)",
        inputs: &RANDOM_NAME_INPUTS,
        outputs: &RANDOM_OUTPUTS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITDIST.INVALID_ARGUMENT",
    identifier: Some("RunMat:fitdist:InvalidArgument"),
    when: "Sample data, distribution name, options, or evaluation inputs are malformed.",
    message: "fitdist: invalid argument",
};

const ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITDIST.NUMERICAL",
    identifier: Some("RunMat:fitdist:Numerical"),
    when: "Distribution parameter estimation fails to converge or is ill-conditioned.",
    message: "fitdist: numerical failure",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITDIST.INTERNAL",
    identifier: Some("RunMat:fitdist:Internal"),
    when: "RunMat cannot construct distribution outputs.",
    message: "fitdist: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_NUMERICAL, ERROR_INTERNAL];

pub const FITDIST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FITDIST_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const PDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PDF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const CDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CDF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const RANDOM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RANDOM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DistributionKind {
    Normal,
    Exponential,
    Lognormal,
    Gamma,
    Weibull,
    Poisson,
}

impl DistributionKind {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Exponential => "Exponential",
            Self::Lognormal => "Lognormal",
            Self::Gamma => "Gamma",
            Self::Weibull => "Weibull",
            Self::Poisson => "Poisson",
        }
    }

    fn parameter_names(self) -> &'static [&'static str] {
        match self {
            Self::Normal => &["mu", "sigma"],
            Self::Exponential => &["mu"],
            Self::Lognormal => &["mu", "sigma"],
            Self::Gamma => &["a", "b"],
            Self::Weibull => &["a", "b"],
            Self::Poisson => &["lambda"],
        }
    }
}

#[derive(Clone, Debug)]
struct FittedDistribution {
    kind: DistributionKind,
    parameters: Vec<f64>,
    nlogl: f64,
    observations: f64,
}

#[derive(Clone, Debug)]
struct WeightedSample {
    values: Vec<f64>,
    weights: Vec<f64>,
    total_weight: f64,
}

#[derive(Default)]
struct FitOptions {
    frequency: Option<Vec<f64>>,
}

fn fitdist_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn distribution_eval_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.get(1) {
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn random_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    if args.len() <= 1 {
        Type::Num
    } else {
        Type::Tensor { shape: None }
    }
}

fn error_for(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if builtin == FITDIST_NAME {
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
    } else {
        let suffix = if std::ptr::eq(descriptor, &ERROR_INTERNAL) {
            "Internal"
        } else if std::ptr::eq(descriptor, &ERROR_NUMERICAL) {
            "Numerical"
        } else {
            "InvalidArgument"
        };
        builder = builder.with_identifier(format!("RunMat:{builtin}:{suffix}"));
    }
    builder.build()
}

fn error(descriptor: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(FITDIST_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error(&ERROR_INVALID_ARGUMENT, message)
}

fn invalid_for(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    error_for(builtin, &ERROR_INVALID_ARGUMENT, message)
}

fn numerical(message: impl Into<String>) -> RuntimeError {
    error(&ERROR_NUMERICAL, message)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error(&ERROR_INTERNAL, message)
}

fn internal_for(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    error_for(builtin, &ERROR_INTERNAL, message)
}

#[runtime_builtin(
    name = "fitdist",
    category = "stats/summary",
    summary = "Fit a probability distribution to sample data.",
    keywords = "fitdist,probability distribution,normal,exponential,lognormal,gamma,weibull,poisson,statistics",
    type_resolver(fitdist_type),
    descriptor(crate::builtins::stats::summary::fitdist::FITDIST_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::fitdist"
)]
pub(crate) async fn fitdist_builtin(
    data: Value,
    distname: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let kind = parse_distribution_name(&distname)?;
    let sample_tensor = value_to_tensor(FITDIST_NAME, data).await?;
    let sample = parse_sample(sample_tensor, parse_fit_options(rest).await?)?;
    let fit = fit_distribution(kind, &sample)?;
    Ok(Value::Object(distribution_object(&fit)?))
}

#[runtime_builtin(
    name = "pdf",
    category = "stats/summary",
    summary = "Evaluate a fitted probability distribution density or mass function.",
    keywords = "pdf,fitdist,probability distribution,density,mass,statistics",
    type_resolver(distribution_eval_type),
    descriptor(crate::builtins::stats::summary::fitdist::PDF_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::fitdist"
)]
pub(crate) async fn pdf_builtin(
    distribution: Value,
    x: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    evaluate_distribution_or_name(PDF_NAME, distribution, x, rest, DistributionEvaluation::Pdf)
        .await
}

#[runtime_builtin(
    name = "cdf",
    category = "stats/summary",
    summary = "Evaluate a fitted probability distribution cumulative distribution function.",
    keywords = "cdf,fitdist,probability distribution,cumulative,statistics",
    type_resolver(distribution_eval_type),
    descriptor(crate::builtins::stats::summary::fitdist::CDF_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::fitdist"
)]
pub(crate) async fn cdf_builtin(
    distribution: Value,
    x: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    evaluate_distribution_or_name(CDF_NAME, distribution, x, rest, DistributionEvaluation::Cdf)
        .await
}

#[runtime_builtin(
    name = "random",
    category = "stats/random",
    summary = "Generate random samples from a fitted probability distribution.",
    keywords = "random,fitdist,probability distribution,statistics",
    type_resolver(random_type),
    descriptor(crate::builtins::stats::summary::fitdist::RANDOM_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::fitdist"
)]
pub(crate) async fn random_builtin(distribution: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let (fit, shape) = if matches!(distribution, Value::Object(_)) {
        (
            distribution_from_value(&distribution)?,
            parse_shape_args(&rest).await?,
        )
    } else {
        parse_named_random_args(distribution, rest).await?
    };
    let len = tensor::element_count(&shape);
    let data = random_samples(&fit, len)?;
    finish_for(RANDOM_NAME, shape, data)
}

pub(crate) async fn icdf_probability_distribution(
    distribution: Value,
    p: Value,
) -> BuiltinResult<Value> {
    evaluate_distribution(distribution, p, DistributionEvaluation::Icdf).await
}

#[derive(Clone, Copy)]
enum DistributionEvaluation {
    Pdf,
    Cdf,
    Icdf,
}

async fn evaluate_distribution(
    distribution: Value,
    input: Value,
    mode: DistributionEvaluation,
) -> BuiltinResult<Value> {
    let fit = distribution_from_value(&distribution)?;
    let x = value_to_tensor(
        match mode {
            DistributionEvaluation::Pdf => PDF_NAME,
            DistributionEvaluation::Cdf => CDF_NAME,
            DistributionEvaluation::Icdf => "icdf",
        },
        input,
    )
    .await?;
    let data = x
        .data
        .iter()
        .map(|value| match mode {
            DistributionEvaluation::Pdf => pdf_scalar(&fit, *value),
            DistributionEvaluation::Cdf => cdf_scalar(&fit, *value),
            DistributionEvaluation::Icdf => icdf_scalar(&fit, *value),
        })
        .collect::<Vec<_>>();
    finish_for(
        match mode {
            DistributionEvaluation::Pdf => PDF_NAME,
            DistributionEvaluation::Cdf => CDF_NAME,
            DistributionEvaluation::Icdf => "icdf",
        },
        x.shape,
        data,
    )
}

async fn evaluate_distribution_or_name(
    builtin: &'static str,
    distribution: Value,
    input: Value,
    rest: Vec<Value>,
    mode: DistributionEvaluation,
) -> BuiltinResult<Value> {
    if matches!(distribution, Value::Object(_)) {
        if !rest.is_empty() {
            return Err(invalid_for(
                builtin,
                format!("{builtin}: fitted distribution object form accepts exactly two inputs"),
            ));
        }
        return evaluate_distribution(distribution, input, mode).await;
    }

    let kind = parse_distribution_name_for(builtin, &distribution)?;
    let params = parse_named_eval_parameters(builtin, kind, rest).await?;
    let fit = FittedDistribution {
        kind,
        parameters: params,
        nlogl: f64::NAN,
        observations: f64::NAN,
    };
    let x = value_to_tensor(builtin, input).await?;
    let data = x
        .data
        .iter()
        .map(|value| match mode {
            DistributionEvaluation::Pdf => pdf_scalar(&fit, *value),
            DistributionEvaluation::Cdf => cdf_scalar(&fit, *value),
            DistributionEvaluation::Icdf => icdf_scalar(&fit, *value),
        })
        .collect::<Vec<_>>();
    finish_for(builtin, x.shape, data)
}

async fn parse_named_eval_parameters(
    builtin: &'static str,
    kind: DistributionKind,
    rest: Vec<Value>,
) -> BuiltinResult<Vec<f64>> {
    let tensors = parse_parameter_tensors(builtin, kind, rest).await?;
    let (values, _shape) = broadcast_tensors_for(builtin, &tensors.iter().collect::<Vec<_>>())?;
    if values.iter().all(|values| values.len() == 1) {
        return Ok(values.into_iter().map(|values| values[0]).collect());
    }
    Err(invalid_for(
        builtin,
        format!("{builtin}: named distribution parameters must be scalar for this overload"),
    ))
}

async fn parse_named_random_args(
    distribution: Value,
    rest: Vec<Value>,
) -> BuiltinResult<(FittedDistribution, Vec<usize>)> {
    let kind = parse_distribution_name_for(RANDOM_NAME, &distribution)?;
    let parameter_count = kind.parameter_names().len();
    if rest.len() < parameter_count {
        return Err(invalid_for(
            RANDOM_NAME,
            format!(
                "random: {} distribution requires {} parameter argument(s)",
                kind.canonical_name(),
                parameter_count
            ),
        ));
    }
    let parameter_values = rest[..parameter_count].to_vec();
    let shape_args = rest[parameter_count..].to_vec();
    let params = parse_named_eval_parameters(RANDOM_NAME, kind, parameter_values).await?;
    let fit = FittedDistribution {
        kind,
        parameters: params,
        nlogl: f64::NAN,
        observations: f64::NAN,
    };
    let shape = parse_shape_args(&shape_args).await?;
    Ok((fit, shape))
}

async fn parse_parameter_tensors(
    builtin: &'static str,
    kind: DistributionKind,
    rest: Vec<Value>,
) -> BuiltinResult<Vec<Tensor>> {
    match kind {
        DistributionKind::Normal | DistributionKind::Lognormal => match rest.as_slice() {
            [] => Ok(vec![scalar_tensor(0.0), scalar_tensor(1.0)]),
            [mu] => Ok(vec![
                value_to_tensor(builtin, mu.clone()).await?,
                scalar_tensor(1.0),
            ]),
            [mu, sigma] => Ok(vec![
                value_to_tensor(builtin, mu.clone()).await?,
                value_to_tensor(builtin, sigma.clone()).await?,
            ]),
            _ => Err(invalid_for(
                builtin,
                format!(
                    "{builtin}: {} distribution expects x, x and mu, or x, mu, sigma",
                    kind.canonical_name()
                ),
            )),
        },
        DistributionKind::Exponential | DistributionKind::Poisson => {
            if rest.len() != 1 {
                return Err(invalid_for(
                    builtin,
                    format!(
                        "{builtin}: {} distribution expects one parameter",
                        kind.canonical_name()
                    ),
                ));
            }
            Ok(vec![value_to_tensor(builtin, rest[0].clone()).await?])
        }
        DistributionKind::Gamma | DistributionKind::Weibull => {
            if rest.len() != 2 {
                return Err(invalid_for(
                    builtin,
                    format!(
                        "{builtin}: {} distribution expects two parameters",
                        kind.canonical_name()
                    ),
                ));
            }
            Ok(vec![
                value_to_tensor(builtin, rest[0].clone()).await?,
                value_to_tensor(builtin, rest[1].clone()).await?,
            ])
        }
    }
}

async fn parse_fit_options(rest: Vec<Value>) -> BuiltinResult<FitOptions> {
    if !rest.len().is_multiple_of(2) {
        return Err(invalid("fitdist: name-value options must be paired"));
    }
    let mut options = FitOptions::default();
    for pair in rest.chunks_exact(2) {
        let name = keyword_of(&pair[0])
            .ok_or_else(|| invalid("fitdist: option names must be text"))?
            .to_ascii_lowercase();
        match name.as_str() {
            "frequency" | "freq" => {
                let tensor = value_to_tensor(FITDIST_NAME, pair[1].clone()).await?;
                options.frequency = Some(tensor.data);
            }
            "censoring" | "censor" => {
                return Err(invalid(
                    "fitdist: Censoring is not supported for fitted distributions yet",
                ))
            }
            "options" | "by" => {
                return Err(invalid(format!(
                    "fitdist: option '{name}' is not supported yet"
                )))
            }
            other => return Err(invalid(format!("fitdist: unknown option '{other}'"))),
        }
    }
    Ok(options)
}

async fn value_to_tensor(name: &'static str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_for(name, format!("{name}: {err}")))?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|_| invalid_for(name, format!("{name}: expected numeric input")))?;
    tensor::integer_tensor_to_f64(tensor).map_err(|err| invalid_for(name, format!("{name}: {err}")))
}

fn scalar_tensor(value: f64) -> Tensor {
    Tensor::new(vec![value], vec![1, 1]).expect("scalar tensor shape is valid")
}

fn broadcast_tensors_for(
    builtin: &'static str,
    inputs: &[&Tensor],
) -> BuiltinResult<(Vec<Vec<f64>>, Vec<usize>)> {
    let Some(first) = inputs.first() else {
        return Ok((Vec::new(), vec![1, 1]));
    };
    let mut shape = first.shape.clone();
    for tensor in inputs.iter().skip(1) {
        shape = broadcast::broadcast_shapes(builtin, &shape, &tensor.shape)
            .map_err(|err| invalid_for(builtin, err))?;
    }
    let mut values = Vec::with_capacity(inputs.len());
    for tensor in inputs {
        values.push(broadcast_tensor_to(builtin, tensor, &shape)?);
    }
    Ok((values, shape))
}

fn broadcast_tensor_to(
    builtin: &'static str,
    tensor: &Tensor,
    out_shape: &[usize],
) -> BuiltinResult<Vec<f64>> {
    let len = tensor::element_count(out_shape);
    if len == 0 {
        return Ok(Vec::new());
    }
    let in_shape = align_shape(&tensor.shape, out_shape.len());
    let strides = broadcast::compute_strides(&in_shape);
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        let source_idx = broadcast::broadcast_index(idx, out_shape, &in_shape, &strides);
        let Some(value) = tensor.data.get(source_idx) else {
            return Err(invalid_for(
                builtin,
                format!("{builtin}: tensor data does not match tensor shape"),
            ));
        };
        out.push(*value);
    }
    Ok(out)
}

fn align_shape(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut aligned = Vec::with_capacity(rank);
    aligned.extend(std::iter::repeat_n(1, rank.saturating_sub(shape.len())));
    aligned.extend_from_slice(shape);
    aligned
}

fn parse_sample(tensor: Tensor, options: FitOptions) -> BuiltinResult<WeightedSample> {
    if tensor.shape.iter().copied().filter(|dim| *dim > 1).count() > 1 {
        return Err(invalid("fitdist: data must be a vector"));
    }
    let frequency = options
        .frequency
        .unwrap_or_else(|| vec![1.0; tensor.data.len()]);
    if frequency.len() != tensor.data.len() {
        return Err(invalid(
            "fitdist: Frequency must contain one value per observation",
        ));
    }
    let mut values = Vec::new();
    let mut weights = Vec::new();
    let mut total_weight = 0.0;
    for (value, weight) in tensor.data.iter().copied().zip(frequency) {
        if weight.is_nan() || weight < 0.0 {
            return Err(invalid(
                "fitdist: Frequency values must be nonnegative finite numbers",
            ));
        }
        if !weight.is_finite() {
            return Err(invalid(
                "fitdist: Frequency values must be nonnegative finite numbers",
            ));
        }
        if value.is_nan() || weight == 0.0 {
            continue;
        }
        if !value.is_finite() {
            return Err(invalid("fitdist: data must not contain Inf values"));
        }
        values.push(value);
        weights.push(weight);
        total_weight += weight;
    }
    if values.is_empty() || total_weight <= 0.0 {
        return Err(invalid(
            "fitdist: at least one finite observation is required",
        ));
    }
    Ok(WeightedSample {
        values,
        weights,
        total_weight,
    })
}

fn fit_distribution(
    kind: DistributionKind,
    sample: &WeightedSample,
) -> BuiltinResult<FittedDistribution> {
    let parameters = match kind {
        DistributionKind::Normal => fit_normal(sample)?,
        DistributionKind::Exponential => fit_exponential(sample)?,
        DistributionKind::Lognormal => fit_lognormal(sample)?,
        DistributionKind::Gamma => fit_gamma(sample)?,
        DistributionKind::Weibull => fit_weibull(sample)?,
        DistributionKind::Poisson => fit_poisson(sample)?,
    };
    let fit = FittedDistribution {
        kind,
        nlogl: sample
            .values
            .iter()
            .zip(sample.weights.iter())
            .map(|(value, weight)| -weight * pdf_scalar_raw(kind, &parameters, *value).ln())
            .sum(),
        parameters,
        observations: sample.total_weight,
    };
    Ok(fit)
}

fn fit_normal(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    let mu = weighted_mean(sample);
    let variance = sample
        .values
        .iter()
        .zip(sample.weights.iter())
        .map(|(value, weight)| weight * (value - mu).powi(2))
        .sum::<f64>()
        / sample.total_weight;
    if variance < 0.0 {
        return Err(numerical("fitdist: normal variance is invalid"));
    }
    Ok(vec![mu, variance.sqrt()])
}

fn fit_exponential(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(sample, |value| value >= 0.0, "Exponential")?;
    let mu = weighted_mean(sample);
    if mu <= 0.0 {
        return Err(invalid("fitdist: Exponential data must have positive mean"));
    }
    Ok(vec![mu])
}

fn fit_lognormal(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(sample, |value| value > 0.0, "Lognormal")?;
    let logs = transformed_sample(sample, |value| value.ln());
    fit_normal(&logs)
}

fn fit_gamma(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(sample, |value| value > 0.0, "Gamma")?;
    let mean = weighted_mean(sample);
    let mean_log = sample
        .values
        .iter()
        .zip(sample.weights.iter())
        .map(|(value, weight)| weight * value.ln())
        .sum::<f64>()
        / sample.total_weight;
    let s = mean.ln() - mean_log;
    if s <= 0.0 {
        return Err(numerical(
            "fitdist: Gamma shape is undefined for nearly constant data",
        ));
    }
    let mut shape =
        ((3.0 - s + ((s - 3.0).powi(2) + 24.0 * s).sqrt()) / (12.0 * s)).max(MIN_POSITIVE);
    for _ in 0..64 {
        let f = shape.ln() - digamma(shape) - s;
        let fp = 1.0 / shape - trigamma(shape);
        let step = f / fp;
        let candidate = shape - step;
        if candidate.is_finite() && candidate > 0.0 {
            shape = candidate;
        } else {
            shape *= 0.5;
        }
        if step.abs() <= 1.0e-12 * shape.max(1.0) {
            break;
        }
    }
    if !shape.is_finite() || shape <= 0.0 {
        return Err(numerical("fitdist: Gamma fit did not converge"));
    }
    Ok(vec![shape, mean / shape])
}

fn fit_weibull(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(sample, |value| value > 0.0, "Weibull")?;
    let logs = transformed_sample(sample, |value| value.ln());
    let log_mean = weighted_mean(&logs);
    let log_var = logs
        .values
        .iter()
        .zip(logs.weights.iter())
        .map(|(value, weight)| weight * (value - log_mean).powi(2))
        .sum::<f64>()
        / logs.total_weight;
    let mut shape = (std::f64::consts::PI / (6.0 * log_var.max(1.0e-12)).sqrt()).clamp(0.1, 100.0);
    for _ in 0..80 {
        let (a, b, c) = weibull_sums(sample, shape);
        let g = a / b - log_mean - 1.0 / shape;
        let gp = c / b - (a / b).powi(2) + 1.0 / shape.powi(2);
        let step = g / gp;
        let candidate = shape - step;
        if candidate.is_finite() && candidate > 0.0 {
            shape = candidate;
        } else {
            shape *= 0.5;
        }
        if step.abs() <= 1.0e-11 * shape.max(1.0) {
            break;
        }
    }
    if !shape.is_finite() || shape <= 0.0 {
        return Err(numerical("fitdist: Weibull fit did not converge"));
    }
    let scale = (sample
        .values
        .iter()
        .zip(sample.weights.iter())
        .map(|(value, weight)| weight * value.powf(shape))
        .sum::<f64>()
        / sample.total_weight)
        .powf(1.0 / shape);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(numerical("fitdist: Weibull scale is invalid"));
    }
    Ok(vec![scale, shape])
}

fn fit_poisson(sample: &WeightedSample) -> BuiltinResult<Vec<f64>> {
    require_range(
        sample,
        |value| value >= 0.0 && value.fract() == 0.0,
        "Poisson",
    )?;
    Ok(vec![weighted_mean(sample)])
}

fn require_range(
    sample: &WeightedSample,
    pred: impl Fn(f64) -> bool,
    name: &str,
) -> BuiltinResult<()> {
    if sample.values.iter().copied().all(pred) {
        Ok(())
    } else {
        Err(invalid(format!(
            "fitdist: {name} distribution data are outside the supported range"
        )))
    }
}

fn transformed_sample(sample: &WeightedSample, transform: impl Fn(f64) -> f64) -> WeightedSample {
    WeightedSample {
        values: sample.values.iter().copied().map(transform).collect(),
        weights: sample.weights.clone(),
        total_weight: sample.total_weight,
    }
}

fn weighted_mean(sample: &WeightedSample) -> f64 {
    sample
        .values
        .iter()
        .zip(sample.weights.iter())
        .map(|(value, weight)| value * weight)
        .sum::<f64>()
        / sample.total_weight
}

fn weibull_sums(sample: &WeightedSample, shape: f64) -> (f64, f64, f64) {
    let mut weighted_xk_log = 0.0;
    let mut weighted_xk = 0.0;
    let mut weighted_xk_log2 = 0.0;
    for (value, weight) in sample.values.iter().zip(sample.weights.iter()) {
        let log_value = value.ln();
        let xk = value.powf(shape);
        weighted_xk_log += weight * xk * log_value;
        weighted_xk += weight * xk;
        weighted_xk_log2 += weight * xk * log_value * log_value;
    }
    (weighted_xk_log, weighted_xk, weighted_xk_log2)
}

fn distribution_object(fit: &FittedDistribution) -> BuiltinResult<ObjectInstance> {
    let mut object = ObjectInstance::new(PROBABILITY_DISTRIBUTION_CLASS.to_string());
    object.properties.insert(
        "DistributionName".to_string(),
        Value::String(fit.kind.canonical_name().to_string()),
    );
    object.properties.insert(
        "DistName".to_string(),
        Value::String(fit.kind.canonical_name().to_string()),
    );
    object.properties.insert(
        "ParameterNames".to_string(),
        Value::StringArray(string_row(
            fit.kind
                .parameter_names()
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
        )?),
    );
    object.properties.insert(
        "ParameterValues".to_string(),
        Value::Tensor(
            Tensor::new(fit.parameters.clone(), vec![1, fit.parameters.len()])
                .map_err(|err| internal(format!("fitdist: {err}")))?,
        ),
    );
    object.properties.insert(
        "NumParameters".to_string(),
        Value::Num(fit.parameters.len() as f64),
    );
    object
        .properties
        .insert("NumObservations".to_string(), Value::Num(fit.observations));
    object
        .properties
        .insert("NLogL".to_string(), Value::Num(fit.nlogl));
    object
        .properties
        .insert("IsTruncated".to_string(), Value::Bool(false));
    for (name, value) in fit.kind.parameter_names().iter().zip(fit.parameters.iter()) {
        object
            .properties
            .insert((*name).to_string(), Value::Num(*value));
    }
    Ok(object)
}

fn string_row(values: Vec<String>) -> BuiltinResult<StringArray> {
    StringArray::new(values.clone(), vec![1, values.len()])
        .map_err(|err| internal(format!("fitdist: {err}")))
}

fn distribution_from_value(value: &Value) -> BuiltinResult<FittedDistribution> {
    let Value::Object(object) = value else {
        return Err(invalid("fitdist: expected ProbabilityDistribution object"));
    };
    if !object.is_class(PROBABILITY_DISTRIBUTION_CLASS) {
        return Err(invalid(format!(
            "fitdist: expected ProbabilityDistribution object, got {}",
            object.class_name
        )));
    }
    let dist_name = string_property(object, "DistributionName")?;
    let kind = parse_distribution_keyword(&dist_name)?;
    let parameters = numeric_vector_property(object, "ParameterValues")?;
    if parameters.len() != kind.parameter_names().len() {
        return Err(invalid(
            "fitdist: ProbabilityDistribution object has malformed ParameterValues",
        ));
    }
    let nlogl = numeric_scalar_property(object, "NLogL").unwrap_or(f64::NAN);
    let observations = numeric_scalar_property(object, "NumObservations").unwrap_or(f64::NAN);
    Ok(FittedDistribution {
        kind,
        parameters,
        nlogl,
        observations,
    })
}

fn string_property(object: &ObjectInstance, name: &str) -> BuiltinResult<String> {
    match object.properties.get(name) {
        Some(Value::String(value)) => Ok(value.clone()),
        Some(Value::CharArray(chars)) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        _ => Err(invalid(format!(
            "fitdist: ProbabilityDistribution object is missing {name}"
        ))),
    }
}

fn numeric_vector_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<f64>> {
    match object.properties.get(name) {
        Some(Value::Tensor(tensor)) => Ok(tensor.data.clone()),
        Some(Value::Num(value)) => Ok(vec![*value]),
        _ => Err(invalid(format!(
            "fitdist: ProbabilityDistribution object is missing {name}"
        ))),
    }
}

fn numeric_scalar_property(object: &ObjectInstance, name: &str) -> Option<f64> {
    match object.properties.get(name) {
        Some(Value::Num(value)) => Some(*value),
        _ => None,
    }
}

fn parse_distribution_name(value: &Value) -> BuiltinResult<DistributionKind> {
    parse_distribution_name_for(FITDIST_NAME, value)
}

fn parse_distribution_name_for(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<DistributionKind> {
    let keyword = keyword_of(value).ok_or_else(|| {
        invalid_for(
            builtin,
            format!("{builtin}: distribution name must be a string scalar"),
        )
    })?;
    parse_distribution_keyword_for(builtin, &keyword)
}

fn parse_distribution_keyword(keyword: &str) -> BuiltinResult<DistributionKind> {
    parse_distribution_keyword_for(FITDIST_NAME, keyword)
}

fn parse_distribution_keyword_for(
    builtin: &'static str,
    keyword: &str,
) -> BuiltinResult<DistributionKind> {
    let normalized = keyword
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    match normalized.as_str() {
        "normal" | "norm" | "gaussian" => Ok(DistributionKind::Normal),
        "exponential" | "exp" => Ok(DistributionKind::Exponential),
        "lognormal" | "logn" => Ok(DistributionKind::Lognormal),
        "gamma" | "gam" => Ok(DistributionKind::Gamma),
        "weibull" | "wbl" => Ok(DistributionKind::Weibull),
        "poisson" | "poiss" => Ok(DistributionKind::Poisson),
        _ => Err(invalid_for(
            builtin,
            format!("{builtin}: unsupported distribution '{keyword}'"),
        )),
    }
}

fn pdf_scalar(fit: &FittedDistribution, x: f64) -> f64 {
    pdf_scalar_raw(fit.kind, &fit.parameters, x)
}

fn pdf_scalar_raw(kind: DistributionKind, params: &[f64], x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    match kind {
        DistributionKind::Normal => {
            let [mu, sigma] = two(params);
            if sigma <= 0.0 {
                return if x == mu { f64::INFINITY } else { 0.0 };
            }
            distribution_math::standard_normal_pdf((x - mu) / sigma) / sigma
        }
        DistributionKind::Exponential => {
            let mu = params[0];
            if x < 0.0 || mu <= 0.0 {
                0.0
            } else {
                (-x / mu).exp() / mu
            }
        }
        DistributionKind::Lognormal => {
            let [mu, sigma] = two(params);
            if x <= 0.0 || sigma <= 0.0 {
                0.0
            } else {
                distribution_math::standard_normal_pdf((x.ln() - mu) / sigma) / (x * sigma)
            }
        }
        DistributionKind::Gamma => {
            let [shape, scale] = two(params);
            if x < 0.0 || shape <= 0.0 || scale <= 0.0 {
                0.0
            } else if x == 0.0 && shape < 1.0 {
                f64::INFINITY
            } else if x == 0.0 && shape > 1.0 {
                0.0
            } else {
                ((shape - 1.0) * x.ln()
                    - x / scale
                    - gammaln_nonnegative_scalar(shape)
                    - shape * scale.ln())
                .exp()
            }
        }
        DistributionKind::Weibull => {
            let [scale, shape] = two(params);
            if x < 0.0 || scale <= 0.0 || shape <= 0.0 {
                0.0
            } else if x == 0.0 && shape < 1.0 {
                f64::INFINITY
            } else if x == 0.0 && shape > 1.0 {
                0.0
            } else {
                (shape / scale) * (x / scale).powf(shape - 1.0) * (-(x / scale).powf(shape)).exp()
            }
        }
        DistributionKind::Poisson => {
            let lambda = params[0];
            if x < 0.0 || x.fract() != 0.0 || lambda < 0.0 {
                0.0
            } else if lambda == 0.0 {
                if x == 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                (x * lambda.ln() - lambda - gammaln_nonnegative_scalar(x + 1.0)).exp()
            }
        }
    }
}

fn cdf_scalar(fit: &FittedDistribution, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    match fit.kind {
        DistributionKind::Normal => {
            let [mu, sigma] = two(&fit.parameters);
            if sigma <= 0.0 {
                if x < mu {
                    0.0
                } else {
                    1.0
                }
            } else {
                distribution_math::standard_normal_cdf((x - mu) / sigma)
            }
        }
        DistributionKind::Exponential => {
            let mu = fit.parameters[0];
            if x < 0.0 || mu <= 0.0 {
                0.0
            } else {
                1.0 - (-x / mu).exp()
            }
        }
        DistributionKind::Lognormal => {
            let [mu, sigma] = two(&fit.parameters);
            if x <= 0.0 || sigma <= 0.0 {
                0.0
            } else {
                distribution_math::standard_normal_cdf((x.ln() - mu) / sigma)
            }
        }
        DistributionKind::Gamma => {
            let [shape, scale] = two(&fit.parameters);
            if x <= 0.0 || shape <= 0.0 || scale <= 0.0 {
                0.0
            } else {
                distribution_math::regularized_gamma_p(shape, x / scale)
            }
        }
        DistributionKind::Weibull => {
            let [scale, shape] = two(&fit.parameters);
            if x < 0.0 || scale <= 0.0 || shape <= 0.0 {
                0.0
            } else {
                1.0 - (-(x / scale).powf(shape)).exp()
            }
        }
        DistributionKind::Poisson => {
            let lambda = fit.parameters[0];
            if x < 0.0 {
                0.0
            } else if lambda == 0.0 {
                1.0
            } else {
                distribution_math::regularized_gamma_q(x.floor() + 1.0, lambda)
            }
        }
    }
}

fn icdf_scalar(fit: &FittedDistribution, p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    match fit.kind {
        DistributionKind::Normal => {
            let [mu, sigma] = two(&fit.parameters);
            mu + sigma * distribution_math::standard_normal_inv(p)
        }
        DistributionKind::Exponential => {
            let mu = fit.parameters[0];
            if p == 1.0 {
                f64::INFINITY
            } else {
                -mu * (1.0 - p).ln()
            }
        }
        DistributionKind::Lognormal => {
            let [mu, sigma] = two(&fit.parameters);
            (mu + sigma * distribution_math::standard_normal_inv(p)).exp()
        }
        DistributionKind::Gamma => {
            let [shape, scale] = two(&fit.parameters);
            invert_positive(p, shape * scale, |x| {
                distribution_math::regularized_gamma_p(shape, x / scale)
            })
        }
        DistributionKind::Weibull => {
            let [scale, shape] = two(&fit.parameters);
            if p == 1.0 {
                f64::INFINITY
            } else {
                scale * (-(1.0 - p).ln()).powf(1.0 / shape)
            }
        }
        DistributionKind::Poisson => poisson_inv(p, fit.parameters[0]),
    }
}

fn random_samples(fit: &FittedDistribution, len: usize) -> BuiltinResult<Vec<f64>> {
    match fit.kind {
        DistributionKind::Normal => {
            let [mu, sigma] = two(&fit.parameters);
            random::generate_normal_scaled(mu, sigma, len, RANDOM_NAME)
        }
        DistributionKind::Exponential => {
            random::generate_exponential(fit.parameters[0].max(MIN_POSITIVE), len, RANDOM_NAME)
        }
        DistributionKind::Lognormal => {
            let [mu, sigma] = two(&fit.parameters);
            random::generate_normal_scaled(mu, sigma, len, RANDOM_NAME)
                .map(|values| values.into_iter().map(f64::exp).collect())
        }
        DistributionKind::Gamma => {
            let [shape, scale] = two(&fit.parameters);
            random::generate_gamma_shape_scale(&[shape], &[scale], len, RANDOM_NAME)
        }
        DistributionKind::Weibull => {
            let [scale, shape] = two(&fit.parameters);
            random::generate_weibull(&[scale], &[shape], len, RANDOM_NAME)
        }
        DistributionKind::Poisson => {
            let uniforms = random::generate_uniform(len, RANDOM_NAME)?;
            Ok(uniforms
                .into_iter()
                .map(|u| poisson_inv(u, fit.parameters[0]))
                .collect())
        }
    }
}

async fn parse_shape_args(rest: &[Value]) -> BuiltinResult<Vec<usize>> {
    if rest.is_empty() {
        return Ok(vec![1, 1]);
    }
    let mut dims = Vec::new();
    for arg in rest {
        match extract_dims(arg, RANDOM_NAME).await {
            Ok(Some(values)) => dims.extend(values),
            Ok(None) => return Err(invalid("random: invalid size argument")),
            Err(err) => return Err(invalid(err)),
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

fn finish_for(builtin: &'static str, shape: Vec<usize>, data: Vec<f64>) -> BuiltinResult<Value> {
    if shape.iter().copied().product::<usize>() == 1 {
        return Ok(Value::Num(data.first().copied().unwrap_or(0.0)));
    }
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| internal_for(builtin, format!("{builtin}: {err}")))
}

fn two(values: &[f64]) -> [f64; 2] {
    [values[0], values[1]]
}

fn invert_positive(p: f64, initial_hi: f64, cdf: impl Fn(f64) -> f64) -> f64 {
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let mut lo = 0.0;
    let mut hi = initial_hi.max(1.0);
    let mut iter = 0;
    while cdf(hi) < p {
        hi *= 2.0;
        iter += 1;
        if !hi.is_finite() || iter > 2048 {
            return f64::INFINITY;
        }
    }
    for _ in 0..160 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

fn poisson_inv(p: f64, lambda: f64) -> f64 {
    if p.is_nan() || lambda.is_nan() || lambda < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if lambda == 0.0 || p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let mut k = 0.0;
    loop {
        if distribution_math::regularized_gamma_q(k + 1.0, lambda) >= p {
            return k;
        }
        k += 1.0;
        if k > lambda + 20.0 * lambda.sqrt().max(1.0) + 1000.0 {
            return k;
        }
    }
}

fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 8.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + inv + 0.5 * inv2 + inv2 * inv / 6.0 - inv2 * inv2 * inv / 30.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn vec_tensor(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).unwrap())
    }

    fn int_vec_tensor(storage: IntegerStorage, len: usize) -> Value {
        Value::Tensor(Tensor::new_integer(storage, vec![len, 1]).unwrap())
    }

    fn object(value: Value) -> ObjectInstance {
        match value {
            Value::Object(object) => object,
            other => panic!("expected object, got {other:?}"),
        }
    }

    #[test]
    fn fitdist_normal_fits_and_evaluates_object_methods() {
        let pd = block_on(fitdist_builtin(
            vec_tensor(&[1.0, 2.0, 3.0]),
            Value::String("Normal".into()),
            Vec::new(),
        ))
        .unwrap();
        let object = object(pd.clone());
        assert_eq!(
            object.properties.get("DistributionName"),
            Some(&Value::String("Normal".into()))
        );
        let values = numeric_vector_property(&object, "ParameterValues").unwrap();
        assert!((values[0] - 2.0).abs() < 1.0e-12);
        assert!((values[1] - (2.0_f64 / 3.0).sqrt()).abs() < 1.0e-12);

        let density = block_on(pdf_builtin(pd.clone(), Value::Num(2.0), Vec::new())).unwrap();
        let Value::Num(density) = density else {
            panic!("expected scalar pdf");
        };
        assert!(density > 0.0);

        let p = block_on(cdf_builtin(pd.clone(), Value::Num(2.0), Vec::new())).unwrap();
        assert_eq!(p, Value::Num(0.5));

        let x = block_on(icdf_probability_distribution(pd, Value::Num(0.5))).unwrap();
        assert_eq!(x, Value::Num(2.0));
    }

    #[test]
    fn fitdist_accepts_typed_integer_sample_and_eval_points() {
        let pd = block_on(fitdist_builtin(
            int_vec_tensor(IntegerStorage::I16(vec![1, 2, 3]), 3),
            Value::String("Normal".into()),
            Vec::new(),
        ))
        .unwrap();
        let object = object(pd.clone());
        let values = numeric_vector_property(&object, "ParameterValues").unwrap();
        assert!((values[0] - 2.0).abs() < 1.0e-12);

        let density = block_on(pdf_builtin(
            pd,
            int_vec_tensor(IntegerStorage::U16(vec![2, 3]), 2),
            Vec::new(),
        ))
        .unwrap();
        match density {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert!(tensor.data.iter().all(|value| value.is_finite()));
            }
            other => panic!("expected tensor density, got {other:?}"),
        }
    }

    #[test]
    fn fitdist_frequency_and_range_validation() {
        let pd = object(
            block_on(fitdist_builtin(
                vec_tensor(&[1.0, 2.0]),
                Value::String("Exponential".into()),
                vec![
                    Value::String("Frequency".into()),
                    Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
                ],
            ))
            .unwrap(),
        );
        let values = numeric_vector_property(&pd, "ParameterValues").unwrap();
        assert!((values[0] - 1.75).abs() < 1.0e-12);

        let err = block_on(fitdist_builtin(
            vec_tensor(&[-1.0, 2.0]),
            Value::String("Gamma".into()),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message.contains("outside the supported range"));
    }

    #[test]
    fn fitdist_gamma_weibull_and_poisson_smoke() {
        let gamma = block_on(fitdist_builtin(
            vec_tensor(&[1.0, 2.0, 4.0, 8.0]),
            Value::String("Gamma".into()),
            Vec::new(),
        ))
        .unwrap();
        assert!(matches!(
            block_on(cdf_builtin(gamma, Value::Num(2.0), Vec::new())).unwrap(),
            Value::Num(value) if value.is_finite()
        ));

        let weibull = block_on(fitdist_builtin(
            vec_tensor(&[1.0, 2.0, 3.0, 5.0, 8.0]),
            Value::String("Weibull".into()),
            Vec::new(),
        ))
        .unwrap();
        assert!(matches!(
            block_on(pdf_builtin(weibull, Value::Num(2.0), Vec::new())).unwrap(),
            Value::Num(value) if value.is_finite() && value >= 0.0
        ));

        let poisson = block_on(fitdist_builtin(
            vec_tensor(&[0.0, 1.0, 1.0, 2.0, 3.0]),
            Value::String("Poisson".into()),
            Vec::new(),
        ))
        .unwrap();
        let samples = block_on(random_builtin(poisson, vec![Value::Num(2.0)])).unwrap();
        match samples {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![2, 2]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn generic_pdf_cdf_random_name_overloads_execute() {
        let density = block_on(pdf_builtin(
            Value::String("Normal".into()),
            Value::Num(0.0),
            vec![Value::Num(0.0), Value::Num(1.0)],
        ))
        .unwrap();
        let Value::Num(density) = density else {
            panic!("expected scalar density");
        };
        assert!((density - distribution_math::standard_normal_pdf(0.0)).abs() < 1.0e-12);

        let probability = block_on(cdf_builtin(
            Value::String("Poisson".into()),
            Value::Num(2.0),
            vec![Value::Num(3.0)],
        ))
        .unwrap();
        let Value::Num(probability) = probability else {
            panic!("expected scalar probability");
        };
        assert!(probability > 0.0 && probability < 1.0);

        let samples = block_on(random_builtin(
            Value::String("Weibull".into()),
            vec![
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(2.0),
                Value::Num(3.0),
            ],
        ))
        .unwrap();
        match samples {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![2, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }
}
