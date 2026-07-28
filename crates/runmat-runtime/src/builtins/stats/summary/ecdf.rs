//! Empirical distribution function helpers.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, ResolveContext, StructValue, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{LinePlot, LineStyle};

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::builtins::plotting::op_common::{apply_axes_target, split_leading_axes_handle};
use crate::builtins::plotting::state::{render_active_plot, PlotRenderOptions};
use crate::builtins::stats::summary::distribution_math::standard_normal_inv;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const ECDF_NAME: &str = "ecdf";
const CDFPLOT_NAME: &str = "cdfplot";

const OUTPUT_F: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "f",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empirical function values.",
};

const OUTPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Evaluation points.",
};

const OUTPUT_FLO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "flo",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Lower confidence bounds.",
};

const OUTPUT_FUP: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "fup",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Upper confidence bounds.",
};

const OUTPUT_HANDLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the empirical CDF plot line.",
};

const OUTPUT_STATS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "stats",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Summary statistics structure.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample data vector.",
};

const PARAM_AX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Function, Censoring, Frequency, Alpha, and Bounds options.",
};

const INPUT_Y: [BuiltinParamDescriptor; 1] = [PARAM_Y];
const INPUT_Y_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_Y, PARAM_OPTIONS];
const INPUT_AX_Y: [BuiltinParamDescriptor; 2] = [PARAM_AX, PARAM_Y];
const INPUT_AX_Y_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_AX, PARAM_Y, PARAM_OPTIONS];
const OUTPUT_F_X: [BuiltinParamDescriptor; 2] = [OUTPUT_F, OUTPUT_X];
const OUTPUT_F_X_BOUNDS: [BuiltinParamDescriptor; 4] = [OUTPUT_F, OUTPUT_X, OUTPUT_FLO, OUTPUT_FUP];
const OUTPUT_H_ONLY: [BuiltinParamDescriptor; 1] = [OUTPUT_HANDLE];
const OUTPUT_H_STATS: [BuiltinParamDescriptor; 2] = [OUTPUT_HANDLE, OUTPUT_STATS];

const ECDF_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "[f, x] = ecdf(y)",
        inputs: &INPUT_Y,
        outputs: &OUTPUT_F_X,
    },
    BuiltinSignatureDescriptor {
        label: "[f, x] = ecdf(y, Name, Value)",
        inputs: &INPUT_Y_OPTIONS,
        outputs: &OUTPUT_F_X,
    },
    BuiltinSignatureDescriptor {
        label: "[f, x, flo, fup] = ecdf(___)",
        inputs: &INPUT_Y_OPTIONS,
        outputs: &OUTPUT_F_X_BOUNDS,
    },
    BuiltinSignatureDescriptor {
        label: "ecdf(y)",
        inputs: &INPUT_Y,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "ecdf(y, Name, Value)",
        inputs: &INPUT_Y_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "ecdf(ax, y)",
        inputs: &INPUT_AX_Y,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "ecdf(ax, y, Name, Value)",
        inputs: &INPUT_AX_Y_OPTIONS,
        outputs: &[],
    },
];

const CDFPLOT_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "cdfplot(x)",
        inputs: &INPUT_Y,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "cdfplot(ax, x)",
        inputs: &INPUT_AX_Y,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = cdfplot(___)",
        inputs: &INPUT_Y,
        outputs: &OUTPUT_H_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "[h, stats] = cdfplot(___)",
        inputs: &INPUT_Y,
        outputs: &OUTPUT_H_STATS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ECDF.INVALID_ARGUMENT",
    identifier: Some("RunMat:ecdf:InvalidArgument"),
    when: "Inputs, name-value options, censoring, frequency, or alpha values are malformed.",
    message: "ecdf: invalid argument",
};

const ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ECDF.UNSUPPORTED",
    identifier: Some("RunMat:ecdf:Unsupported"),
    when: "A documented but not yet implemented interval or double-censored form is requested.",
    message: "ecdf: unsupported form",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ECDF.INTERNAL",
    identifier: Some("RunMat:ecdf:Internal"),
    when: "RunMat cannot construct ECDF outputs or plots.",
    message: "ecdf: internal error",
};

const CDFPLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CDFPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:cdfplot:InvalidArgument"),
    when: "Inputs or axes arguments are malformed.",
    message: "cdfplot: invalid argument",
};

const CDFPLOT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CDFPLOT.INTERNAL",
    identifier: Some("RunMat:cdfplot:Internal"),
    when: "RunMat cannot construct summary statistics or plots.",
    message: "cdfplot: internal error",
};

const ECDF_ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_UNSUPPORTED, ERROR_INTERNAL];
const CDFPLOT_ERRORS: [BuiltinErrorDescriptor; 2] =
    [CDFPLOT_ERROR_INVALID_ARGUMENT, CDFPLOT_ERROR_INTERNAL];

pub const ECDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ECDF_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ECDF_ERRORS,
};

pub const CDFPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CDFPLOT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CDFPLOT_ERRORS,
};

fn ecdf_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn ecdf_error(
    builtin: &'static str,
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    let descriptor = if builtin == CDFPLOT_NAME {
        &CDFPLOT_ERROR_INVALID_ARGUMENT
    } else {
        &ERROR_INVALID_ARGUMENT
    };
    ecdf_error(builtin, message, descriptor)
}

fn unsupported(message: impl Into<String>) -> RuntimeError {
    ecdf_error(ECDF_NAME, message, &ERROR_UNSUPPORTED)
}

fn internal_error(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    let descriptor = if builtin == CDFPLOT_NAME {
        &CDFPLOT_ERROR_INTERNAL
    } else {
        &ERROR_INTERNAL
    };
    ecdf_error(builtin, message, descriptor)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FunctionKind {
    Cdf,
    Survivor,
    CumulativeHazard,
}

impl FunctionKind {
    fn y_label(self) -> &'static str {
        match self {
            Self::Cdf => "F(x)",
            Self::Survivor => "S(x)",
            Self::CumulativeHazard => "H(x)",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cdf => "Empirical cdf",
            Self::Survivor => "Empirical survivor function",
            Self::CumulativeHazard => "Empirical cumulative hazard",
        }
    }
}

#[derive(Clone, Debug)]
struct Options {
    function: FunctionKind,
    censoring: Option<Vec<f64>>,
    frequency: Option<Vec<f64>>,
    alpha: f64,
    bounds_on: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            function: FunctionKind::Cdf,
            censoring: None,
            frequency: None,
            alpha: 0.05,
            bounds_on: false,
        }
    }
}

#[derive(Clone, Debug)]
struct Observation {
    time: f64,
    censor: f64,
    frequency: f64,
}

#[derive(Clone, Debug)]
struct EcdfEvaluation {
    f: Vec<f64>,
    x: Vec<f64>,
    flo: Vec<f64>,
    fup: Vec<f64>,
    hazard: Vec<f64>,
    hazard_flo: Vec<f64>,
    hazard_fup: Vec<f64>,
    function: FunctionKind,
    bounds_on: bool,
}

impl EcdfEvaluation {
    fn f_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.f.clone()).map(Value::Tensor)
    }

    fn x_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.x.clone()).map(Value::Tensor)
    }

    fn flo_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.flo.clone()).map(Value::Tensor)
    }

    fn fup_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.fup.clone()).map(Value::Tensor)
    }

    fn outputs_for_count(&self, out_count: usize) -> BuiltinResult<Vec<Value>> {
        let mut outputs = Vec::with_capacity(out_count.min(4));
        if out_count >= 1 {
            outputs.push(self.f_value()?);
        }
        if out_count >= 2 {
            outputs.push(self.x_value()?);
        }
        if out_count >= 3 {
            outputs.push(self.flo_value()?);
        }
        if out_count >= 4 {
            outputs.push(self.fup_value()?);
        }
        Ok(outputs)
    }
}

#[runtime_builtin(
    name = "ecdf",
    category = "stats/summary",
    summary = "Compute or plot an empirical cumulative distribution function.",
    keywords = "ecdf,empirical cdf,cumulative distribution,survivor,kaplan-meier,statistics",
    suppress_auto_output = true,
    type_resolver(ecdf_type),
    descriptor(crate::builtins::stats::summary::ecdf::ECDF_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::ecdf"
)]
pub(crate) async fn ecdf_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (axes_target, args) =
        split_leading_axes_handle(args, ECDF_NAME).map_err(|err| map_error(ECDF_NAME, err))?;
    let mut args = gather_values(args).await?;
    if args.is_empty() {
        return Err(invalid_argument(ECDF_NAME, "ecdf: expected sample data"));
    }
    let y = args.remove(0);
    let options = parse_options(args)?;
    let eval = evaluate_ecdf(y, options)?;

    if is_statement_form_call() {
        apply_axes_target(axes_target, ECDF_NAME).map_err(|err| map_error(ECDF_NAME, err))?;
        plot_ecdf_line(ECDF_NAME, &eval)?;
        return Ok(Value::OutputList(Vec::new()));
    }

    match crate::output_count::current_output_count() {
        Some(0) => {
            apply_axes_target(axes_target, ECDF_NAME).map_err(|err| map_error(ECDF_NAME, err))?;
            plot_ecdf_line(ECDF_NAME, &eval)?;
            Ok(Value::OutputList(Vec::new()))
        }
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            eval.outputs_for_count(out_count)?,
        )),
        None => eval.f_value(),
    }
}

#[runtime_builtin(
    name = "cdfplot",
    category = "stats/summary",
    summary = "Plot an empirical cumulative distribution function and return summary statistics.",
    keywords = "cdfplot,empirical cdf,plot,statistics",
    sink = true,
    suppress_auto_output = true,
    type_resolver(ecdf_type),
    descriptor(crate::builtins::stats::summary::ecdf::CDFPLOT_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::ecdf"
)]
pub(crate) async fn cdfplot_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (axes_target, args) = split_leading_axes_handle(args, CDFPLOT_NAME)
        .map_err(|err| map_error(CDFPLOT_NAME, err))?;
    let mut args = gather_values(args).await?;
    if args.len() != 1 {
        return Err(invalid_argument(
            CDFPLOT_NAME,
            "cdfplot: expected exactly one numeric vector after optional axes handle",
        ));
    }
    let data = numeric_vector(&args[0], CDFPLOT_NAME)?;
    let stats = cdfplot_stats(&data)?;
    let eval = evaluate_ecdf(args.remove(0), Options::default())?;
    apply_axes_target(axes_target, CDFPLOT_NAME).map_err(|err| map_error(CDFPLOT_NAME, err))?;
    let handle = plot_ecdf_line(CDFPLOT_NAME, &eval)?;

    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![Value::Num(handle)])),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![Value::Num(handle), Value::Struct(stats)],
        )),
        None => Ok(Value::Num(handle)),
    }
}

fn is_statement_form_call() -> bool {
    matches!(crate::output_count::current_output_count(), Some(0))
        || (crate::output_context::requested_output_count() == Some(0)
            && crate::output_count::current_output_count().is_none())
}

fn map_error(builtin: &'static str, err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        invalid_argument(builtin, err.message)
    }
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(values.len());
    for value in values {
        gathered.push(gather_if_needed_async(&value).await?);
    }
    Ok(gathered)
}

fn parse_options(rest: Vec<Value>) -> BuiltinResult<Options> {
    if !rest.len().is_multiple_of(2) {
        return Err(invalid_argument(
            ECDF_NAME,
            "ecdf: name-value options must appear in pairs",
        ));
    }
    let mut options = Options::default();
    let mut idx = 0;
    while idx < rest.len() {
        let name = keyword_of(&rest[idx])
            .ok_or_else(|| invalid_argument(ECDF_NAME, "ecdf: option names must be text"))?;
        let value = &rest[idx + 1];
        idx += 2;
        match name.to_ascii_lowercase().as_str() {
            "function" => {
                let text = keyword_of(value).ok_or_else(|| {
                    invalid_argument(ECDF_NAME, "ecdf: Function option must be text")
                })?;
                options.function = match text.to_ascii_lowercase().as_str() {
                    "cdf" => FunctionKind::Cdf,
                    "survivor" => FunctionKind::Survivor,
                    "cumulative hazard" | "cumhazard" | "hazard" => FunctionKind::CumulativeHazard,
                    other => {
                        return Err(invalid_argument(
                            ECDF_NAME,
                            format!("ecdf: unsupported Function value '{other}'"),
                        ))
                    }
                };
            }
            "censoring" => {
                options.censoring = Some(numeric_vector(value, ECDF_NAME)?);
            }
            "frequency" => {
                options.frequency = Some(numeric_vector(value, ECDF_NAME)?);
            }
            "alpha" => {
                let alpha = scalar_number(value).ok_or_else(|| {
                    invalid_argument(ECDF_NAME, "ecdf: Alpha must be a numeric scalar")
                })?;
                if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
                    return Err(invalid_argument(
                        ECDF_NAME,
                        "ecdf: Alpha must be a finite scalar in the open interval (0,1)",
                    ));
                }
                options.alpha = alpha;
            }
            "bounds" => {
                let text = keyword_of(value).ok_or_else(|| {
                    invalid_argument(ECDF_NAME, "ecdf: Bounds option must be text")
                })?;
                options.bounds_on = match text.to_ascii_lowercase().as_str() {
                    "on" => true,
                    "off" => false,
                    other => {
                        return Err(invalid_argument(
                            ECDF_NAME,
                            format!("ecdf: unsupported Bounds value '{other}'"),
                        ))
                    }
                };
            }
            "iterationlimit" | "tolerance" | "icmfrequency" => {
                return Err(unsupported(format!(
                    "ecdf: option '{name}' is only valid for interval or double-censored data"
                )));
            }
            other => {
                return Err(invalid_argument(
                    ECDF_NAME,
                    format!("ecdf: unsupported option '{other}'"),
                ))
            }
        }
    }
    Ok(options)
}

fn evaluate_ecdf(value: Value, options: Options) -> BuiltinResult<EcdfEvaluation> {
    let tensor = tensor_from_value(value, ECDF_NAME)?;
    if tensor.rows() > 1 && tensor.cols() > 1 {
        if tensor.cols() == 2 {
            return Err(unsupported(
                "ecdf: interval-censored two-column input is not supported yet",
            ));
        }
        return Err(invalid_argument(
            ECDF_NAME,
            format!(
                "ecdf: expected a vector or two-column interval matrix, got {}x{} matrix",
                tensor.rows(),
                tensor.cols()
            ),
        ));
    }
    let data = tensor::tensor_values_f64(&tensor);
    if data.is_empty() {
        return Err(invalid_argument(
            ECDF_NAME,
            "ecdf: sample data must be nonempty",
        ));
    }
    let n = data.len();
    let censoring = options.censoring.unwrap_or_else(|| vec![0.0; n]);
    let frequency = options.frequency.unwrap_or_else(|| vec![1.0; n]);
    if censoring.len() != n {
        return Err(invalid_argument(
            ECDF_NAME,
            "ecdf: Censoring vector must have the same number of elements as y",
        ));
    }
    if frequency.len() != n {
        return Err(invalid_argument(
            ECDF_NAME,
            "ecdf: Frequency vector must have the same number of elements as y",
        ));
    }

    let mut observations = Vec::new();
    let mut has_left = false;
    let mut has_right = false;
    for ((time, censor), freq) in data
        .into_iter()
        .zip(censoring.into_iter())
        .zip(frequency.into_iter())
    {
        if time.is_nan() || censor.is_nan() || freq.is_nan() || freq == 0.0 {
            continue;
        }
        if !time.is_finite() {
            return Err(invalid_argument(
                ECDF_NAME,
                "ecdf: sample data must be finite after removing NaN values",
            ));
        }
        if !(freq.is_finite() && freq >= 0.0) {
            return Err(invalid_argument(
                ECDF_NAME,
                "ecdf: Frequency values must be finite nonnegative scalars",
            ));
        }
        let censor = if censor == 0.0 {
            0.0
        } else if censor == 1.0 {
            has_right = true;
            1.0
        } else if censor == -1.0 {
            has_left = true;
            -1.0
        } else {
            return Err(invalid_argument(
                ECDF_NAME,
                "ecdf: Censoring values must be 0, 1, or -1",
            ));
        };
        observations.push(Observation {
            time,
            censor,
            frequency: freq,
        });
    }
    if observations.is_empty() {
        return Err(invalid_argument(
            ECDF_NAME,
            "ecdf: no finite positive-frequency observations remain",
        ));
    }
    if has_left && has_right {
        return Err(unsupported(
            "ecdf: double-censored data with both left and right censoring is not supported yet",
        ));
    }
    if has_left {
        return evaluate_left_censored(&observations, options.alpha)
            .map(|eval| transform_eval(eval, options.function, options.bounds_on));
    }

    observations.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(Ordering::Equal));
    let eval = if has_right {
        evaluate_kaplan_meier(&observations, options.alpha)
    } else {
        evaluate_fully_observed(&observations, options.alpha)
    }?;
    Ok(transform_eval(eval, options.function, options.bounds_on))
}

fn evaluate_fully_observed(
    observations: &[Observation],
    alpha: f64,
) -> BuiltinResult<EcdfEvaluation> {
    let total: f64 = observations.iter().map(|obs| obs.frequency).sum();
    if total <= 0.0 {
        return Err(invalid_argument(
            ECDF_NAME,
            "ecdf: total frequency must be positive",
        ));
    }
    let z = standard_normal_inv(1.0 - alpha / 2.0);
    let mut x = Vec::new();
    let mut f = Vec::new();
    let mut flo = Vec::new();
    let mut fup = Vec::new();
    let mut hazard = Vec::new();
    let mut hazard_flo = Vec::new();
    let mut hazard_fup = Vec::new();
    let first = observations[0].time;
    x.push(first);
    f.push(0.0);
    flo.push(0.0);
    fup.push(0.0);
    hazard.push(0.0);
    hazard_flo.push(0.0);
    hazard_fup.push(0.0);

    let mut cumulative = 0.0;
    let mut cumulative_hazard = 0.0;
    let mut hazard_variance = 0.0;
    let mut idx = 0;
    while idx < observations.len() {
        let time = observations[idx].time;
        let mut count = 0.0;
        while idx < observations.len() && observations[idx].time == time {
            count += observations[idx].frequency;
            idx += 1;
        }
        let risk = total - cumulative;
        cumulative += count;
        let p = (cumulative / total).clamp(0.0, 1.0);
        let se = (p * (1.0 - p) / total).sqrt();
        if risk > 0.0 {
            cumulative_hazard += count / risk;
            hazard_variance += count / (risk * risk);
        }
        let hazard_se = hazard_variance.sqrt();
        x.push(time);
        f.push(p);
        flo.push((p - z * se).clamp(0.0, 1.0));
        fup.push((p + z * se).clamp(0.0, 1.0));
        hazard.push(cumulative_hazard);
        hazard_flo.push((cumulative_hazard - z * hazard_se).max(0.0));
        hazard_fup.push(cumulative_hazard + z * hazard_se);
    }

    Ok(EcdfEvaluation {
        f,
        x,
        flo,
        fup,
        hazard,
        hazard_flo,
        hazard_fup,
        function: FunctionKind::Cdf,
        bounds_on: false,
    })
}

fn evaluate_kaplan_meier(
    observations: &[Observation],
    alpha: f64,
) -> BuiltinResult<EcdfEvaluation> {
    let z = standard_normal_inv(1.0 - alpha / 2.0);
    let total: f64 = observations.iter().map(|obs| obs.frequency).sum();
    if total <= 0.0 {
        return Err(invalid_argument(
            ECDF_NAME,
            "ecdf: total frequency must be positive",
        ));
    }
    let mut x = Vec::new();
    let mut f = Vec::new();
    let mut flo = Vec::new();
    let mut fup = Vec::new();
    let mut hazard = Vec::new();
    let mut hazard_flo = Vec::new();
    let mut hazard_fup = Vec::new();
    let mut risk = total;
    let mut survivor = 1.0;
    let mut greenwood = 0.0;
    let mut cumulative_hazard = 0.0;
    let mut hazard_variance = 0.0;
    let mut idx = 0;
    while idx < observations.len() {
        let time = observations[idx].time;
        let mut events = 0.0;
        let mut censored = 0.0;
        while idx < observations.len() && observations[idx].time == time {
            if observations[idx].censor == 0.0 {
                events += observations[idx].frequency;
            } else {
                censored += observations[idx].frequency;
            }
            idx += 1;
        }
        if events > 0.0 && risk > 0.0 {
            if x.is_empty() {
                x.push(time);
                f.push(0.0);
                flo.push(0.0);
                fup.push(0.0);
                hazard.push(0.0);
                hazard_flo.push(0.0);
                hazard_fup.push(0.0);
            }
            let decrement = (1.0_f64 - events / risk).max(0.0_f64);
            survivor *= decrement;
            if risk > events {
                greenwood += events / (risk * (risk - events));
            }
            cumulative_hazard += events / risk;
            hazard_variance += events / (risk * risk);
            let cdf = (1.0 - survivor).clamp(0.0, 1.0);
            let se = (survivor * survivor * greenwood).sqrt();
            let hazard_se = hazard_variance.sqrt();
            x.push(time);
            f.push(cdf);
            flo.push((cdf - z * se).clamp(0.0, 1.0));
            fup.push((cdf + z * se).clamp(0.0, 1.0));
            hazard.push(cumulative_hazard);
            hazard_flo.push((cumulative_hazard - z * hazard_se).max(0.0));
            hazard_fup.push(cumulative_hazard + z * hazard_se);
        }
        risk -= events + censored;
    }

    Ok(EcdfEvaluation {
        f,
        x,
        flo,
        fup,
        hazard,
        hazard_flo,
        hazard_fup,
        function: FunctionKind::Cdf,
        bounds_on: false,
    })
}

fn evaluate_left_censored(
    observations: &[Observation],
    alpha: f64,
) -> BuiltinResult<EcdfEvaluation> {
    let z = standard_normal_inv(1.0 - alpha / 2.0);
    let total: f64 = observations.iter().map(|obs| obs.frequency).sum();
    if total <= 0.0 {
        return Err(invalid_argument(
            ECDF_NAME,
            "ecdf: total frequency must be positive",
        ));
    }

    let mut descending = observations.to_vec();
    descending.sort_by(|a, b| b.time.partial_cmp(&a.time).unwrap_or(Ordering::Equal));
    let mut risk = total;
    let mut reverse_survivor: f64 = 1.0;
    let mut greenwood: f64 = 0.0;
    let mut points = Vec::new();
    let mut idx = 0;
    while idx < descending.len() {
        let time = descending[idx].time;
        let mut events = 0.0;
        let mut censored = 0.0;
        while idx < descending.len() && descending[idx].time == time {
            if descending[idx].censor == 0.0 {
                events += descending[idx].frequency;
            } else {
                censored += descending[idx].frequency;
            }
            idx += 1;
        }
        if events > 0.0 && risk > 0.0 {
            let cdf = reverse_survivor.clamp(0.0, 1.0);
            let se = (reverse_survivor * reverse_survivor * greenwood).sqrt();
            points.push((
                time,
                cdf,
                (cdf - z * se).clamp(0.0, 1.0),
                (cdf + z * se).clamp(0.0, 1.0),
            ));
            let decrement = (1.0_f64 - events / risk).max(0.0_f64);
            if risk > events {
                greenwood += events / (risk * (risk - events));
            }
            reverse_survivor *= decrement;
        }
        risk -= events + censored;
    }
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut x = Vec::new();
    let mut f = Vec::new();
    let mut flo = Vec::new();
    let mut fup = Vec::new();
    if let Some((first_time, _, _, _)) = points.first().copied() {
        x.push(first_time);
        f.push(0.0);
        flo.push(0.0);
        fup.push(0.0);
    }
    for (time, cdf, lower, upper) in points {
        x.push(time);
        f.push(cdf);
        flo.push(lower);
        fup.push(upper);
    }
    let (hazard, hazard_flo, hazard_fup) = hazard_from_cdf_bounds(&f, &flo, &fup);

    Ok(EcdfEvaluation {
        f,
        x,
        flo,
        fup,
        hazard,
        hazard_flo,
        hazard_fup,
        function: FunctionKind::Cdf,
        bounds_on: false,
    })
}

fn transform_eval(
    mut eval: EcdfEvaluation,
    function: FunctionKind,
    bounds_on: bool,
) -> EcdfEvaluation {
    if function == FunctionKind::Cdf {
        eval.bounds_on = bounds_on;
        return eval;
    }
    match function {
        FunctionKind::Cdf => unreachable!(),
        FunctionKind::Survivor => {
            let f = std::mem::take(&mut eval.f);
            let flo = std::mem::take(&mut eval.flo);
            let fup = std::mem::take(&mut eval.fup);
            eval.f = f.iter().map(|value| 1.0 - value).collect();
            eval.flo = fup.iter().map(|value| 1.0 - value).collect();
            eval.fup = flo.iter().map(|value| 1.0 - value).collect();
        }
        FunctionKind::CumulativeHazard => {
            eval.f = eval.hazard.clone();
            eval.flo = eval.hazard_flo.clone();
            eval.fup = eval.hazard_fup.clone();
        }
    }
    eval.function = function;
    eval.bounds_on = bounds_on;
    eval
}

fn hazard_from_survivor(survivor: f64) -> f64 {
    if survivor <= 0.0 {
        f64::INFINITY
    } else {
        -survivor.ln()
    }
}

fn hazard_from_cdf_bounds(f: &[f64], flo: &[f64], fup: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let hazard = f
        .iter()
        .map(|value| hazard_from_survivor(1.0 - value))
        .collect();
    let hazard_flo = flo
        .iter()
        .map(|value| hazard_from_survivor(1.0 - value))
        .collect();
    let hazard_fup = fup
        .iter()
        .map(|value| hazard_from_survivor(1.0 - value))
        .collect();
    (hazard, hazard_flo, hazard_fup)
}

fn plot_ecdf_line(builtin: &'static str, eval: &EcdfEvaluation) -> BuiltinResult<f64> {
    let (x, y) = step_line_points(&eval.x, &eval.f);
    let label = eval.function.label();
    let mut plots = vec![line_plot(builtin, x, y, label)?];
    if eval.bounds_on {
        let (x, y) = step_line_points(&eval.x, &eval.flo);
        plots.push(line_plot(builtin, x, y, "Lower confidence bound")?);
        let (x, y) = step_line_points(&eval.x, &eval.fup);
        plots.push(line_plot(builtin, x, y, "Upper confidence bound")?);
    }
    let opts = PlotRenderOptions {
        title: if builtin == CDFPLOT_NAME {
            "Empirical CDF"
        } else {
            ""
        },
        x_label: "x",
        y_label: eval.function.y_label(),
        ..Default::default()
    };
    let mut plots_opt = Some(plots);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(builtin, opts, move |figure, axes_index| {
        let plots = plots_opt.take().expect("ecdf plots consumed exactly once");
        for (idx, plot) in plots.into_iter().enumerate() {
            let plot_index = figure.add_line_plot_on_axes(plot, axes_index);
            if idx == 0 {
                *plot_index_slot.borrow_mut() = Some((axes_index, plot_index));
            }
        }
        Ok(())
    });
    let Some((axes_index, plot_index)) = *plot_index_out.borrow() else {
        return render_result
            .map(|_| f64::NAN)
            .map_err(|err| internal_error(builtin, err.message));
    };
    let handle = crate::builtins::plotting::state::register_line_handle(
        figure_handle,
        axes_index,
        plot_index,
    );
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(internal_error(builtin, err.message));
    }
    Ok(handle)
}

fn line_plot(
    builtin: &'static str,
    x: Vec<f64>,
    y: Vec<f64>,
    label: &str,
) -> BuiltinResult<LinePlot> {
    LinePlot::new(x, y)
        .map_err(|err| internal_error(builtin, format!("{builtin}: {err}")))
        .map(|plot| {
            plot.with_label(label).with_style(
                [0.0, 0.4470, 0.7410, 1.0].into(),
                1.5,
                LineStyle::Solid,
            )
        })
}

fn step_line_points(x: &[f64], f: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if x.is_empty() || f.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut xs = Vec::with_capacity(x.len().saturating_mul(2));
    let mut ys = Vec::with_capacity(f.len().saturating_mul(2));
    xs.push(x[0]);
    ys.push(f[0]);
    for idx in 1..x.len().min(f.len()) {
        xs.push(x[idx]);
        ys.push(f[idx - 1]);
        xs.push(x[idx]);
        ys.push(f[idx]);
    }
    (xs, ys)
}

fn cdfplot_stats(data: &[f64]) -> BuiltinResult<StructValue> {
    let mut values = data
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Err(invalid_argument(
            CDFPLOT_NAME,
            "cdfplot: sample data must contain at least one non-NaN value",
        ));
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let median = if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    };
    let std = if n < 2 {
        0.0
    } else {
        let var = values
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f64>()
            / (n - 1) as f64;
        var.sqrt()
    };
    let mut stats = StructValue::new();
    stats
        .fields
        .insert("min".to_string(), Value::Num(values[0]));
    stats
        .fields
        .insert("max".to_string(), Value::Num(values[n - 1]));
    stats.fields.insert("mean".to_string(), Value::Num(mean));
    stats
        .fields
        .insert("median".to_string(), Value::Num(median));
    stats.fields.insert("std".to_string(), Value::Num(std));
    Ok(stats)
}

fn numeric_vector(value: &Value, builtin: &'static str) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor_from_value(value.clone(), builtin)?;
    if tensor.rows() > 1 && tensor.cols() > 1 {
        return Err(invalid_argument(
            builtin,
            format!(
                "{builtin}: expected a vector, got {}x{} matrix",
                tensor.rows(),
                tensor.cols()
            ),
        ));
    }
    Ok(tensor_values_f64(&tensor))
}

fn tensor_from_value(value: Value, builtin: &'static str) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            Tensor::new(tensor_values_f64(&tensor), tensor.shape.clone())
                .map_err(|err| internal_error(builtin, format!("{builtin}: {err}")))
        }
        Value::Tensor(tensor) => Ok(tensor),
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| internal_error(builtin, format!("{builtin}: {err}"))),
        Value::Int(value) => Tensor::new(vec![value.to_f64()], vec![1, 1])
            .map_err(|err| internal_error(builtin, format!("{builtin}: {err}"))),
        Value::Bool(value) => Tensor::new(vec![if value { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|err| internal_error(builtin, format!("{builtin}: {err}"))),
        other => Tensor::try_from(&other)
            .map_err(|err| invalid_argument(builtin, format!("{builtin}: {err}"))),
    }
}

fn scalar_number(value: &Value) -> Option<f64> {
    if let Some(value) = integer_scalar(value) {
        return Some(value.to_f64());
    }
    match value {
        Value::Num(value) => Some(*value),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        _ => None,
    }
}

fn integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn tensor_values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor::tensor_values_f64(tensor)
}

fn column_tensor(data: Vec<f64>) -> BuiltinResult<Tensor> {
    let len = data.len();
    Tensor::new(data, vec![len, 1]).map_err(|err| internal_error(ECDF_NAME, format!("ecdf: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn tensor(data: Vec<f64>) -> Value {
        Value::Tensor(Tensor::new(data.clone(), vec![data.len(), 1]).unwrap())
    }

    fn int_tensor(storage: IntegerStorage, len: usize) -> Value {
        let mut tensor = Tensor::new_integer(storage, vec![len, 1]).unwrap();
        tensor.data.clear();
        Value::Tensor(tensor)
    }

    fn output_tensors(value: Value) -> Vec<Tensor> {
        let Value::OutputList(values) = value else {
            panic!("expected output list");
        };
        values
            .into_iter()
            .map(|value| match value {
                Value::Tensor(tensor) => tensor,
                other => panic!("expected tensor, got {other:?}"),
            })
            .collect()
    }

    fn matrix(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
    }

    #[test]
    fn ecdf_numeric_parsers_read_typed_integer_storage_exactly() {
        let wide = u64::MAX - 1;
        assert_eq!(
            numeric_vector(
                &int_tensor(IntegerStorage::U64(vec![wide, wide - 1]), 2),
                ECDF_NAME,
            )
            .unwrap(),
            vec![
                IntValue::U64(wide).to_f64(),
                IntValue::U64(wide - 1).to_f64()
            ]
        );
        assert_eq!(
            scalar_number(&int_tensor(IntegerStorage::U64(vec![wide]), 1)).unwrap(),
            IntValue::U64(wide).to_f64()
        );
    }

    #[test]
    fn ecdf_accepts_typed_integer_data_frequency_and_censoring() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let value = block_on(ecdf_builtin(vec![
            int_tensor(IntegerStorage::I16(vec![1, 2, 3, 4]), 4),
            Value::from("Frequency"),
            int_tensor(IntegerStorage::U16(vec![2, 1, 1, 1]), 4),
            Value::from("Censoring"),
            int_tensor(IntegerStorage::I8(vec![0, 1, 0, 0]), 4),
        ]))
        .unwrap();
        let outputs = output_tensors(value);
        assert_eq!(outputs[1].data, vec![1.0, 1.0, 3.0, 4.0]);
        assert!((outputs[0].data[1] - 0.4).abs() < 1.0e-12);
        assert!((outputs[0].data[2] - 0.7).abs() < 1.0e-12);
    }

    #[test]
    fn ecdf_alpha_rejects_typed_integer_boundary_value() {
        let err = block_on(ecdf_builtin(vec![
            int_tensor(IntegerStorage::I16(vec![1, 2, 3]), 3),
            Value::from("Alpha"),
            int_tensor(IntegerStorage::U8(vec![1]), 1),
        ]))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:ecdf:InvalidArgument"));
        assert!(err.message().contains("open interval"), "{}", err.message());
    }

    #[test]
    fn ecdf_returns_unique_sorted_points_with_initial_zero() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let value = block_on(ecdf_builtin(vec![tensor(vec![3.0, 1.0, 2.0, 2.0])])).unwrap();
        let outputs = output_tensors(value);
        assert_eq!(outputs[0].data, vec![0.0, 0.25, 0.75, 1.0]);
        assert_eq!(outputs[1].data, vec![1.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn ecdf_supports_frequency_survivor_and_bounds_outputs() {
        let _guard = crate::output_count::push_output_count(Some(4));
        let value = block_on(ecdf_builtin(vec![
            tensor(vec![1.0, 2.0, 3.0]),
            Value::from("Frequency"),
            tensor(vec![2.0, 1.0, 1.0]),
            Value::from("Function"),
            Value::from("survivor"),
            Value::from("Alpha"),
            Value::Num(0.1),
        ]))
        .unwrap();
        let outputs = output_tensors(value);
        assert_eq!(outputs[0].data, vec![1.0, 0.5, 0.25, 0.0]);
        assert_eq!(outputs[1].data, vec![1.0, 1.0, 2.0, 3.0]);
        assert_eq!(outputs[2].shape, vec![4, 1]);
        assert_eq!(outputs[3].shape, vec![4, 1]);
    }

    #[test]
    fn ecdf_supports_right_censoring_with_kaplan_meier_steps() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let value = block_on(ecdf_builtin(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0]),
            Value::from("Censoring"),
            tensor(vec![0.0, 1.0, 0.0, 0.0]),
        ]))
        .unwrap();
        let outputs = output_tensors(value);
        assert_eq!(outputs[1].data, vec![1.0, 1.0, 3.0, 4.0]);
        assert!((outputs[0].data[1] - 0.25).abs() < 1.0e-12);
        assert!((outputs[0].data[2] - 0.625).abs() < 1.0e-12);
        assert!((outputs[0].data[3] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn ecdf_right_censoring_starts_at_first_event_time() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let value = block_on(ecdf_builtin(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0]),
            Value::from("Censoring"),
            tensor(vec![1.0, 0.0, 0.0, 0.0]),
        ]))
        .unwrap();
        let outputs = output_tensors(value);
        assert_eq!(outputs[1].data, vec![2.0, 2.0, 3.0, 4.0]);
        assert!((outputs[0].data[1] - (1.0 / 3.0)).abs() < 1.0e-12);
    }

    #[test]
    fn ecdf_cumulative_hazard_uses_nelson_aalen_steps() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let value = block_on(ecdf_builtin(vec![
            tensor(vec![1.0, 2.0, 3.0]),
            Value::from("Function"),
            Value::from("cumulative hazard"),
        ]))
        .unwrap();
        let outputs = output_tensors(value);
        assert_eq!(outputs[1].data, vec![1.0, 1.0, 2.0, 3.0]);
        let expected = [
            0.0,
            1.0 / 3.0,
            1.0 / 3.0 + 1.0 / 2.0,
            1.0 / 3.0 + 1.0 / 2.0 + 1.0,
        ];
        for (actual, expected) in outputs[0].data.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        assert!(outputs[0].data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn ecdf_rejects_non_vector_non_interval_matrix() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let err = block_on(ecdf_builtin(vec![matrix(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            2,
            3,
        )]))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:ecdf:InvalidArgument"));
    }

    #[test]
    fn ecdf_supports_left_censoring_without_right_censoring() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let value = block_on(ecdf_builtin(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0]),
            Value::from("Censoring"),
            tensor(vec![-1.0, 0.0, 0.0, 0.0]),
        ]))
        .unwrap();
        let outputs = output_tensors(value);
        assert_eq!(outputs[1].data, vec![2.0, 2.0, 3.0, 4.0]);
        assert!(outputs[0].data.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn cdfplot_returns_handle_and_stats() {
        let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
        let _ = crate::builtins::plotting::clear_figure(None);
        let _guard = crate::output_count::push_output_count(Some(2));
        let value = block_on(cdfplot_builtin(vec![tensor(vec![1.0, 2.0, 3.0])])).unwrap();
        let Value::OutputList(values) = value else {
            panic!("expected output list");
        };
        assert!(matches!(values[0], Value::Num(handle) if handle > 0.0));
        let Value::Struct(stats) = &values[1] else {
            panic!("expected stats struct");
        };
        assert!(matches!(stats.fields.get("min"), Some(Value::Num(1.0))));
        assert!(matches!(stats.fields.get("max"), Some(Value::Num(3.0))));
        assert!(matches!(stats.fields.get("mean"), Some(Value::Num(2.0))));
        assert!(matches!(stats.fields.get("median"), Some(Value::Num(2.0))));
    }
}
