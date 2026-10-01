//! Hypothesis-test compatibility helpers.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Ttest2OptionName, Ttest2Tail as Tail, Ttest2Variance as VarType, Type,
    TTEST2_ERROR_INTERNAL, TTEST2_ERROR_INVALID_ARGUMENT, TTEST2_INTEGER_OPTION_EXTENSION,
    TTEST2_INTEGER_SAMPLE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, StructValue, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::stats::summary::distribution_math::{
    student_t_cdf, student_t_cdf_upper, student_t_inv,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "ttest2";
const KSTEST_NAME: &str = "kstest";
const MAX_SUPPORTED_DIM: usize = 1024;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

#[path = "hypothesis/ttest2_options.rs"]
mod ttest2_options;
use ttest2_options::Ttest2OptionPlan;

// kstest has not yet moved to the catalog; keep its output metadata owned
// here rather than coupling that identity to ttest2's catalog package.
const OUTPUT_H: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Hypothesis test decision.",
};
const OUTPUT_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "P-value.",
};

const KSTEST_INTEGER_SAMPLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "kstest-integer-sample",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "kstest with a native-class integer sample is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:KstestIntegerSampleExtension"),
};
const KSTEST_INTEGER_CDF_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "kstest-integer-cdf",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "kstest with a native-class integer CDF table is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:KstestIntegerCdfExtension"),
};
const KSTEST_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "kstest-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "kstest host fallback for explicit gpuArray inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:KstestResidentInputExtension"),
};
pub const KSTEST_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    KSTEST_INTEGER_SAMPLE_EXTENSION,
    KSTEST_INTEGER_CDF_EXTENSION,
    KSTEST_RESIDENT_INPUT_EXTENSION,
];

const KSTEST_INTEGER_SAMPLE_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "x", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Native integer samples are gated before gather and checked for exact binary64 representation." }];
const KSTEST_INTEGER_CDF_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "CDF table",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Native integer CDF tables are gated and checked before interpolation.",
    }];
pub const KSTEST_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "h = kstest(integer_x, ___)", inputs: &KSTEST_INTEGER_SAMPLE_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "h is logical; p, ksstat, and cv are double. Lossy integer conversion rejects." },
    BuiltinIntegerCapabilityDescriptor { form: "h = kstest(x, CDF=integer_table)", inputs: &KSTEST_INTEGER_CDF_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer CDF tables are a checked RunMat extension; automatic residency gathers transparently, while explicit gpuArray fallback remains separately gated." },
];

const KSTEST_PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample data.",
};

const KSTEST_PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Alpha, CDF, and Tail options.",
};

const OUTPUT_KSSTAT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ksstat",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Kolmogorov-Smirnov test statistic.",
};

const OUTPUT_CV: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "cv",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Approximate critical value.",
};

const KSTEST_INPUTS_X: [BuiltinParamDescriptor; 1] = [KSTEST_PARAM_X];
const KSTEST_INPUTS_OPTIONS: [BuiltinParamDescriptor; 2] = [KSTEST_PARAM_X, KSTEST_PARAM_OPTIONS];
const KSTEST_OUTPUT_H: [BuiltinParamDescriptor; 1] = [OUTPUT_H];
const KSTEST_OUTPUT_H_P: [BuiltinParamDescriptor; 2] = [OUTPUT_H, OUTPUT_P];
const KSTEST_OUTPUT_FULL: [BuiltinParamDescriptor; 4] =
    [OUTPUT_H, OUTPUT_P, OUTPUT_KSSTAT, OUTPUT_CV];

const KSTEST_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = kstest(x)",
        inputs: &KSTEST_INPUTS_X,
        outputs: &KSTEST_OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "h = kstest(x, Name, Value)",
        inputs: &KSTEST_INPUTS_OPTIONS,
        outputs: &KSTEST_OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "[h, p] = kstest(___)",
        inputs: &KSTEST_INPUTS_OPTIONS,
        outputs: &KSTEST_OUTPUT_H_P,
    },
    BuiltinSignatureDescriptor {
        label: "[h, p, ksstat, cv] = kstest(___)",
        inputs: &KSTEST_INPUTS_OPTIONS,
        outputs: &KSTEST_OUTPUT_FULL,
    },
];

const KSTEST_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KSTEST.INVALID_ARGUMENT",
    identifier: Some("RunMat:kstest:InvalidArgument"),
    when: "Inputs, CDF specification, tail, or significance level are malformed.",
    message: "kstest: invalid argument",
};

const KSTEST_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KSTEST.INTERNAL",
    identifier: Some("RunMat:kstest:Internal"),
    when: "RunMat cannot construct kstest outputs.",
    message: "kstest: internal error",
};

const KSTEST_ERRORS: [BuiltinErrorDescriptor; 2] =
    [KSTEST_ERROR_INVALID_ARGUMENT, KSTEST_ERROR_INTERNAL];

pub const KSTEST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &KSTEST_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &KSTEST_ERRORS,
};

fn kstest_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn ttest2_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    ttest2_error(message, &TTEST2_ERROR_INVALID_ARGUMENT)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    ttest2_error(message, &TTEST2_ERROR_INTERNAL)
}

fn kstest_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(KSTEST_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn kstest_invalid_argument(message: impl Into<String>) -> RuntimeError {
    kstest_error(message, &KSTEST_ERROR_INVALID_ARGUMENT)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KstestTail {
    Unequal,
    Larger,
    Smaller,
}

#[derive(Clone, Debug)]
enum CdfSpec {
    StandardNormal,
    Table(Vec<(f64, f64)>),
}

#[derive(Clone, Debug)]
struct KstestOptions {
    alpha: f64,
    tail: KstestTail,
    cdf: CdfSpec,
}

impl Default for KstestOptions {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            tail: KstestTail::Unequal,
            cdf: CdfSpec::StandardNormal,
        }
    }
}

#[derive(Clone, Debug)]
struct KstestEvaluation {
    h: Value,
    p: Value,
    ksstat: Value,
    cv: Value,
}

#[runtime_builtin(
    name = "ttest2",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::hypothesis"
)]
pub(crate) async fn ttest2_builtin(x: Value, y: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_ttest2_sample_extension(&x, &y)?;
    let options = Ttest2OptionPlan::admit(rest)?;
    if crate::output_count::current_output_count().is_some_and(|count| count > 4) {
        return Err(invalid_argument(
            "ttest2: too many output arguments; maximum is 4",
        ));
    }
    let x = gathered_tensor(x).await?;
    let y = gathered_tensor(y).await?;
    ensure_exact_ttest2_integer_tensor(&x, "sample x")?;
    ensure_exact_ttest2_integer_tensor(&y, "sample y")?;
    let options = options.resolve().await?;
    let eval = evaluate(&x, &y, options)?;
    match crate::output_count::current_output_count() {
        Some(out_count) => outputs_for_count(out_count, eval),
        None => Ok(eval.h),
    }
}

#[runtime_builtin(
    name = "kstest",
    category = "stats/summary",
    summary = "Perform a one-sample Kolmogorov-Smirnov test.",
    keywords = "kstest,kolmogorov-smirnov,hypothesis test,cdf,statistics",
    type_resolver(kstest_type),
    descriptor(crate::builtins::stats::summary::hypothesis::KSTEST_DESCRIPTOR),
    extensions(crate::builtins::stats::summary::hypothesis::KSTEST_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::summary::hypothesis::KSTEST_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::hypothesis"
)]
pub(crate) async fn kstest_builtin(x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_kstest_extensions(&x, &rest)?;
    let x = gather_if_needed_async(&x)
        .await
        .map_err(|err| kstest_invalid_argument(format!("kstest: {err}")))?;
    let x = tensor::value_into_tensor_for(KSTEST_NAME, x).map_err(kstest_invalid_argument)?;
    ensure_exact_kstest_integer_tensor(&x, "sample")?;
    let rest = kstest_gather_values(rest).await?;
    let options = parse_kstest_options(rest)?;
    let eval = evaluate_kstest(&x, &options)?;
    match crate::output_count::current_output_count() {
        Some(out_count) => kstest_outputs_for_count(out_count, eval),
        None => Ok(eval.h),
    }
}

fn is_kstest_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn kstest_cdf_contains_typed_integer(value: &Value) -> bool {
    match value {
        Value::Cell(cell) if cell.data.len() == 1 => {
            kstest_cdf_contains_typed_integer(&cell.data[0])
        }
        _ => is_kstest_typed_integer(value),
    }
}

fn kstest_value_contains_explicit_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(handle),
        Value::Cell(cell) => cell.data.iter().any(kstest_value_contains_explicit_gpu),
        Value::Struct(value) => value
            .fields
            .values()
            .any(kstest_value_contains_explicit_gpu),
        Value::Object(value) => value
            .properties
            .values()
            .any(kstest_value_contains_explicit_gpu),
        Value::Closure(value) => value
            .captures
            .iter()
            .any(kstest_value_contains_explicit_gpu),
        Value::OutputList(values) => values.iter().any(kstest_value_contains_explicit_gpu),
        _ => false,
    }
}

fn validate_kstest_cdf_role(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Tensor(_)
        | Value::GpuTensor(_)
        | Value::String(_)
        | Value::CharArray(_)
        | Value::FunctionHandle(_) => Ok(()),
        Value::Cell(cell) if cell.data.len() == 1 => validate_kstest_cdf_role(&cell.data[0]),
        other => Err(kstest_invalid_argument(format!(
            "kstest: CDF must be a two-column numeric matrix or normcdf handle; probability distribution objects are not supported yet, got {other:?}"
        ))),
    }
}

fn validate_kstest_alpha_role(value: &Value) -> BuiltinResult<()> {
    let valid = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
        Value::LogicalArray(array) => array.data.len() == 1,
        Value::GpuTensor(handle) => {
            handle
                .shape()
                .iter()
                .try_fold(1usize, |len, dim| len.checked_mul(*dim))
                == Some(1)
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(kstest_invalid_argument("kstest: Alpha must be numeric"))
    }
}

fn ensure_kstest_extensions(x: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if !rest.len().is_multiple_of(2) {
        return Err(kstest_invalid_argument(
            "kstest: name-value options must be supplied in pairs",
        ));
    }
    for pair in rest.chunks_exact(2) {
        let name = kstest_scalar_text(&pair[0], "option name")?;
        match name.to_ascii_lowercase().as_str() {
            "alpha" => validate_kstest_alpha_role(&pair[1])?,
            "tail" => {
                kstest_scalar_text(&pair[1], "Tail")?;
            }
            "cdf" => validate_kstest_cdf_role(&pair[1])?,
            other => {
                return Err(kstest_invalid_argument(format!(
                    "kstest: unsupported option '{other}'"
                )))
            }
        }
    }
    if is_kstest_typed_integer(x) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &KSTEST_INTEGER_SAMPLE_EXTENSION,
            KSTEST_NAME,
        )?;
    }
    for pair in rest.chunks_exact(2) {
        let name = kstest_scalar_text(&pair[0], "option name")?;
        if name.eq_ignore_ascii_case("cdf") && kstest_cdf_contains_typed_integer(&pair[1]) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &KSTEST_INTEGER_CDF_EXTENSION,
                KSTEST_NAME,
            )?;
        }
    }
    if kstest_value_contains_explicit_gpu(x) || rest.iter().any(kstest_value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &KSTEST_RESIDENT_INPUT_EXTENSION,
            KSTEST_NAME,
        )?;
    }
    Ok(())
}

fn ensure_exact_kstest_integer_tensor(value: &Tensor, role: &str) -> BuiltinResult<()> {
    if let Some(storage) = value.integer_storage() {
        for integer in storage.exact_values() {
            if !crate::builtins::common::validation::integer_is_exact_f64(&integer) {
                return Err(kstest_invalid_argument(format!(
                    "kstest: integer {role} values must be exactly representable as double"
                )));
            }
        }
    }
    Ok(())
}

fn kstest_outputs_for_count(out_count: usize, eval: KstestEvaluation) -> BuiltinResult<Value> {
    match out_count {
        0 => Ok(Value::OutputList(Vec::new())),
        1 => Ok(Value::OutputList(vec![eval.h])),
        2 => Ok(Value::OutputList(vec![eval.h, eval.p])),
        3 => Ok(Value::OutputList(vec![eval.h, eval.p, eval.ksstat])),
        4 => Ok(Value::OutputList(vec![
            eval.h,
            eval.p,
            eval.ksstat,
            eval.cv,
        ])),
        _ => Err(kstest_invalid_argument(
            "kstest: too many output arguments; maximum is 4",
        )),
    }
}

async fn kstest_gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(
            gather_if_needed_async(&value)
                .await
                .map_err(|err| kstest_invalid_argument(format!("kstest: {err}")))?,
        );
    }
    Ok(out)
}

fn parse_kstest_options(rest: Vec<Value>) -> BuiltinResult<KstestOptions> {
    let mut options = KstestOptions::default();
    let mut idx = 0;
    while idx < rest.len() {
        let name = kstest_scalar_text(&rest[idx], "option name")?;
        idx += 1;
        if idx >= rest.len() {
            return Err(kstest_invalid_argument(format!(
                "kstest: option '{name}' requires a value"
            )));
        }
        let value = &rest[idx];
        idx += 1;
        match name.to_ascii_lowercase().as_str() {
            "alpha" => {
                let alpha = kstest_scalar_number(value)
                    .ok_or_else(|| kstest_invalid_argument("kstest: Alpha must be numeric"))?;
                if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
                    return Err(kstest_invalid_argument(
                        "kstest: Alpha must be a scalar in the open interval (0,1)",
                    ));
                }
                options.alpha = alpha;
            }
            "tail" => {
                let tail = kstest_scalar_text(value, "Tail")?;
                options.tail = match tail.to_ascii_lowercase().as_str() {
                    "unequal" | "both" => KstestTail::Unequal,
                    "larger" | "right" => KstestTail::Larger,
                    "smaller" | "left" => KstestTail::Smaller,
                    other => {
                        return Err(kstest_invalid_argument(format!(
                            "kstest: unsupported Tail '{other}'"
                        )))
                    }
                };
            }
            "cdf" => {
                options.cdf = parse_kstest_cdf(value)?;
            }
            other => {
                return Err(kstest_invalid_argument(format!(
                    "kstest: unsupported option '{other}'"
                )))
            }
        }
    }
    Ok(options)
}

fn parse_kstest_cdf(value: &Value) -> BuiltinResult<CdfSpec> {
    match value {
        Value::Tensor(tensor) => parse_kstest_cdf_table(tensor),
        Value::String(text) if is_normal_cdf_name(text) => Ok(CdfSpec::StandardNormal),
        Value::CharArray(chars) if chars.rows == 1 => {
            let text = chars.data.iter().collect::<String>();
            if is_normal_cdf_name(&text) {
                Ok(CdfSpec::StandardNormal)
            } else {
                Err(kstest_invalid_argument(format!(
                    "kstest: unsupported CDF '{text}'"
                )))
            }
        }
        Value::FunctionHandle(name) if is_normal_cdf_name(name) => Ok(CdfSpec::StandardNormal),
        Value::Cell(cell) if cell.data.len() == 1 => parse_kstest_cdf(&cell.data[0]),
        other => Err(kstest_invalid_argument(format!(
            "kstest: CDF must be a two-column numeric matrix or normcdf handle; probability distribution objects are not supported yet, got {other:?}"
        ))),
    }
}

fn is_normal_cdf_name(text: &str) -> bool {
    let normalized = text
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    matches!(
        normalized.as_str(),
        "normcdf" | "normal" | "norm" | "gaussian"
    )
}

fn parse_kstest_cdf_table(tensor: &Tensor) -> BuiltinResult<CdfSpec> {
    ensure_exact_kstest_integer_tensor(tensor, "CDF table")?;
    if tensor.shape.len() != 2 || tensor.shape[1] != 2 || tensor.shape[0] < 2 {
        return Err(kstest_invalid_argument(
            "kstest: CDF table must be an n-by-2 numeric matrix",
        ));
    }
    let rows = tensor.shape[0];
    let mut points = Vec::with_capacity(rows);
    for row in 0..rows {
        let x = tensor::tensor_value_f64(tensor, row);
        let f = tensor::tensor_value_f64(tensor, row + rows);
        if !x.is_finite() || !f.is_finite() || !(0.0..=1.0).contains(&f) {
            return Err(kstest_invalid_argument(
                "kstest: CDF table values must be finite and probabilities must be in [0,1]",
            ));
        }
        points.push((x, f));
    }
    points.sort_by(|(x_a, f_a), (x_b, f_b)| x_a.total_cmp(x_b).then_with(|| f_a.total_cmp(f_b)));

    let mut normalized = Vec::with_capacity(points.len());
    for (x, f) in points {
        if let Some((last_x, last_f)) = normalized.last_mut() {
            if *last_x == x {
                if f > *last_f {
                    *last_f = f;
                }
                continue;
            }
            if f < *last_f {
                return Err(kstest_invalid_argument(
                    "kstest: CDF table probabilities must be nondecreasing after sorting by x",
                ));
            }
        }
        normalized.push((x, f));
    }

    if normalized.len() < 2 {
        return Err(kstest_invalid_argument(
            "kstest: CDF table must contain at least two distinct x values",
        ));
    }
    for window in normalized.windows(2) {
        if window[1].1 < window[0].1 {
            return Err(kstest_invalid_argument(
                "kstest: CDF table probabilities must be nondecreasing",
            ));
        }
    }
    Ok(CdfSpec::Table(normalized))
}

fn kstest_scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::Cell(cell) if cell.data.len() == 1 => kstest_scalar_text(&cell.data[0], context),
        other => Err(kstest_invalid_argument(format!(
            "kstest: {context} must be a string scalar, got {other:?}"
        ))),
    }
}

fn kstest_scalar_number(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_values_f64(tensor).first().copied()
        }
        _ => None,
    }
}

fn evaluate_kstest(x: &Tensor, options: &KstestOptions) -> BuiltinResult<KstestEvaluation> {
    if !is_kstest_vector_shape(&x.shape) {
        return Err(kstest_invalid_argument("kstest: x must be a vector"));
    }
    let mut sample = tensor::tensor_values_f64(x)
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    sample.sort_by(f64::total_cmp);
    if sample.is_empty() {
        return Err(kstest_invalid_argument(
            "kstest: x must contain at least one non-NaN value",
        ));
    }
    validate_kstest_sample_domain(&sample, &options.cdf)?;
    let ksstat = kstest_statistic(&sample, options);
    let p = kstest_p_value(ksstat, sample.len(), options.tail);
    let cv = kstest_critical_value(options.alpha, sample.len(), options.tail);
    Ok(KstestEvaluation {
        h: Value::Bool(p.is_finite() && p <= options.alpha),
        p: Value::Num(p),
        ksstat: Value::Num(ksstat),
        cv: Value::Num(cv),
    })
}

fn is_kstest_vector_shape(shape: &[usize]) -> bool {
    shape.iter().filter(|dim| **dim > 1).count() <= 1
}

fn validate_kstest_sample_domain(sample: &[f64], cdf: &CdfSpec) -> BuiltinResult<()> {
    if let CdfSpec::Table(points) = cdf {
        let min = points[0].0;
        let max = points[points.len() - 1].0;
        if sample[0] < min || sample[sample.len() - 1] > max {
            return Err(kstest_invalid_argument(
                "kstest: sample values must lie within the CDF table range",
            ));
        }
    }
    Ok(())
}

fn kstest_statistic(sample: &[f64], options: &KstestOptions) -> f64 {
    let n = sample.len() as f64;
    let mut d_plus: f64 = 0.0;
    let mut d_minus: f64 = 0.0;
    for (idx, value) in sample.iter().enumerate() {
        let f0 = cdf_value(&options.cdf, *value).clamp(0.0, 1.0);
        let i = idx as f64 + 1.0;
        d_plus = d_plus.max(i / n - f0);
        d_minus = d_minus.max(f0 - (i - 1.0) / n);
    }
    match options.tail {
        KstestTail::Unequal => d_plus.max(d_minus),
        KstestTail::Larger => d_plus,
        KstestTail::Smaller => d_minus,
    }
}

fn cdf_value(cdf: &CdfSpec, x: f64) -> f64 {
    match cdf {
        CdfSpec::StandardNormal => normal_cdf(x),
        CdfSpec::Table(points) => table_cdf_value(points, x),
    }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / SQRT_2)
}

fn table_cdf_value(points: &[(f64, f64)], x: f64) -> f64 {
    if x <= points[0].0 {
        return points[0].1;
    }
    let last = points.len() - 1;
    if x >= points[last].0 {
        return points[last].1;
    }
    let upper = points.partition_point(|(point_x, _)| *point_x < x);
    let (x0, f0) = points[upper - 1];
    let (x1, f1) = points[upper];
    let t = (x - x0) / (x1 - x0);
    f0 + t * (f1 - f0)
}

fn kstest_p_value(ksstat: f64, n: usize, tail: KstestTail) -> f64 {
    if !ksstat.is_finite() || n == 0 {
        return f64::NAN;
    }
    let n = n as f64;
    match tail {
        KstestTail::Unequal => {
            let z = (n.sqrt() + 0.12 + 0.11 / n.sqrt()) * ksstat;
            let mut sum = 0.0;
            for j in 1..=100 {
                let term = (-2.0 * (j as f64).powi(2) * z * z).exp();
                if j % 2 == 1 {
                    sum += term;
                } else {
                    sum -= term;
                }
                if term < 1.0e-14 {
                    break;
                }
            }
            (2.0 * sum).clamp(0.0, 1.0)
        }
        KstestTail::Larger | KstestTail::Smaller => (-2.0 * n * ksstat * ksstat).exp(),
    }
}

fn kstest_critical_value(alpha: f64, n: usize, tail: KstestTail) -> f64 {
    if !(alpha > 0.0 && alpha < 1.0) || n == 0 {
        return f64::NAN;
    }
    let n = n as f64;
    match tail {
        KstestTail::Unequal if (0.01..=0.20).contains(&alpha) => {
            (-0.5 * (alpha / 2.0).ln()).sqrt() / n.sqrt()
        }
        KstestTail::Larger | KstestTail::Smaller if (0.005..=0.10).contains(&alpha) => {
            (-0.5 * alpha.ln()).sqrt() / n.sqrt()
        }
        _ => f64::NAN,
    }
}

#[path = "hypothesis/ttest2_engine.rs"]
mod ttest2_engine;
use ttest2_engine::*;

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    fn output_list(value: Value) -> Vec<Value> {
        match value {
            Value::OutputList(values) => values,
            other => panic!("expected output list, got {other:?}"),
        }
    }

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual}"
        );
    }

    fn poisoned_integer_scalar(storage: IntegerStorage) -> Value {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
        Value::Tensor(tensor)
    }

    fn poisoned_integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).unwrap();
        Value::Tensor(tensor)
    }

    #[test]
    fn ttest2_equal_variance_vector_outputs_match_reference_values() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap());
        let y = Value::Tensor(Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![4, 1]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(4));
        let outputs = output_list(block_on(ttest2_builtin(x, y, Vec::new())).unwrap());
        assert_eq!(outputs[0], Value::Bool(false));
        assert_close(
            match outputs[1] {
                Value::Num(p) => p,
                ref other => panic!("expected p, got {other:?}"),
            },
            0.133_974_596,
            1.0e-8,
        );
        match &outputs[2] {
            Value::Tensor(ci) => {
                assert_eq!(ci.shape, vec![2, 1]);
                assert_close(ci.materialize_f64()[0], -6.031_813, 1.0e-6);
                assert_close(ci.materialize_f64()[1], 1.031_813, 1.0e-6);
            }
            other => panic!("expected ci tensor, got {other:?}"),
        }
        match &outputs[3] {
            Value::Struct(stats) => {
                assert_close(
                    match stats.fields.get("tstat").unwrap() {
                        Value::Num(v) => *v,
                        other => panic!("tstat {other:?}"),
                    },
                    -1.732_051,
                    1.0e-6,
                );
                assert_eq!(stats.fields.get("df").unwrap(), &Value::Num(6.0));
                assert_close(
                    match stats.fields.get("sd").unwrap() {
                        Value::Num(v) => *v,
                        other => panic!("sd {other:?}"),
                    },
                    2.041_241,
                    1.0e-6,
                );
            }
            other => panic!("expected stats struct, got {other:?}"),
        }
    }

    #[test]
    fn ttest2_unequal_variance_tail_and_nan_omission() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, f64::NAN, 4.0], vec![5, 1]).unwrap());
        let y = Value::Tensor(Tensor::new(vec![10.0, 11.0, 12.0, 13.0], vec![4, 1]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(4));
        let outputs = output_list(
            block_on(ttest2_builtin(
                x,
                y,
                vec![
                    Value::from("Vartype"),
                    Value::from("unequal"),
                    Value::from("Tail"),
                    Value::from("left"),
                    Value::from("Alpha"),
                    Value::Num(0.01),
                ],
            ))
            .unwrap(),
        );
        assert_eq!(outputs[0], Value::Bool(true));
        match outputs[1] {
            Value::Num(p) => assert!(p < 0.001),
            ref other => panic!("expected p, got {other:?}"),
        }
        match &outputs[2] {
            Value::Tensor(ci) => {
                assert_eq!(ci.materialize_f64()[0], f64::NEG_INFINITY);
                assert!(ci.materialize_f64()[1] < 0.0);
            }
            other => panic!("expected ci tensor, got {other:?}"),
        }
        match &outputs[3] {
            Value::Struct(stats) => match stats.fields.get("sd").unwrap() {
                Value::Tensor(sd) => {
                    assert_eq!(sd.shape, vec![2, 1]);
                    assert_close(sd.materialize_f64()[0], 1.290_994, 1.0e-6);
                    assert_close(sd.materialize_f64()[1], 1.290_994, 1.0e-6);
                }
                other => panic!("expected unequal sd tensor, got {other:?}"),
            },
            other => panic!("expected stats struct, got {other:?}"),
        }
    }

    #[test]
    fn ttest2_matrix_dim_outputs_shape_and_invalid_samples() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0], vec![3, 2]).unwrap());
        let y = Value::Tensor(Tensor::new(vec![1.0, 1.0, 1.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap());
        {
            let _guard = crate::output_count::push_output_count(Some(2));
            let outputs = output_list(block_on(ttest2_builtin(x, y, Vec::new())).unwrap());
            match &outputs[0] {
                Value::LogicalArray(h) => assert_eq!(h.shape, vec![1, 2]),
                other => panic!("expected h logical array, got {other:?}"),
            }
            match &outputs[1] {
                Value::Tensor(p) => assert_eq!(p.shape, vec![1, 2]),
                other => panic!("expected p tensor, got {other:?}"),
            }
        }

        let row_x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 5.0, 7.0], vec![2, 2]).unwrap());
        let row_y = Value::Tensor(Tensor::new(vec![2.0, 2.0, 4.0, 4.0], vec![2, 2]).unwrap());
        let out = block_on(ttest2_builtin(
            row_x,
            row_y,
            vec![Value::from("Dim"), Value::Num(2.0)],
        ))
        .unwrap();
        match out {
            Value::LogicalArray(h) => assert_eq!(h.shape, vec![2, 1]),
            other => panic!("expected dim-2 h, got {other:?}"),
        }
    }

    #[test]
    fn ttest2_rejects_bad_options_and_shapes() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        let y = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let err = block_on(ttest2_builtin(x, y, Vec::new())).unwrap_err();
        assert!(err.message.contains("same size"));

        let err = block_on(ttest2_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            vec![Value::from("Tail"), Value::from("sideways")],
        ))
        .unwrap_err();
        assert!(err.message.contains("Tail"));
    }

    #[test]
    fn ttest2_host_invalid_controls_precede_sample_provider_access() {
        let resident = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![4, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("genuine unregistered resident fixture")
                .handle,
        );
        let y = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap());
        let error = block_on(ttest2_builtin(
            resident,
            y,
            vec![Value::from("Tail"), Value::from("sideways")],
        ))
        .expect_err("host-invalid Tail must be rejected before sample gather");
        assert!(error.message.contains("unsupported Tail"), "{error}");
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:ttest2:InvalidArgument")
        );
    }

    #[test]
    fn ttest2_sample_provider_failure_preserves_nested_cause() {
        let resident = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![4, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("genuine unregistered resident fixture")
                .handle,
        );
        let y = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap());
        let error = block_on(ttest2_builtin(resident, y, Vec::new()))
            .expect_err("unregistered resident sample must fail at provider access");
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:ttest2:InvalidArgument")
        );
        assert!(
            error.source.is_some(),
            "provider failure must remain a nested cause"
        );
    }

    #[test]
    fn ttest2_rejects_too_many_outputs_and_huge_dim() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let y = Value::Tensor(Tensor::new(vec![1.0, 2.0, 4.0], vec![3, 1]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(5));
        let err = block_on(ttest2_builtin(x, y, Vec::new())).unwrap_err();
        assert!(err.message.contains("too many output arguments"));

        let err = block_on(ttest2_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            vec![Value::from("Dim"), Value::Num(1.0e20)],
        ))
        .unwrap_err();
        assert!(err.message.contains("Dim"));
    }

    #[test]
    fn ttest2_numeric_options_read_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap());
        let y = Value::Tensor(Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap());
        let out = block_on(ttest2_builtin(
            x,
            y,
            vec![
                Value::from("Dim"),
                poisoned_integer_scalar(IntegerStorage::U64(vec![1])),
            ],
        ))
        .unwrap();
        assert_eq!(out, Value::Bool(false));

        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let y = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let err = block_on(ttest2_builtin(
            x,
            y,
            vec![
                Value::from("Alpha"),
                poisoned_integer_scalar(IntegerStorage::I64(vec![1])),
            ],
        ))
        .unwrap_err();
        assert!(err.message.contains("Alpha"));
    }

    #[test]
    fn ttest2_sample_values_read_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let x = poisoned_integer_tensor(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![4, 1]);
        let y = poisoned_integer_tensor(IntegerStorage::U16(vec![1, 2, 4, 8]), vec![4, 1]);
        let out = block_on(ttest2_builtin(x, y, Vec::new())).unwrap();
        assert_eq!(out, Value::Bool(false));
    }

    #[test]
    fn ttest2_gates_integer_roles_and_rejects_lossy_boundaries() {
        let integer = poisoned_integer_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]);
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = ensure_ttest2_extensions(&integer, &Value::Num(1.0), &[])
            .expect_err("integer sample must gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:Ttest2IntegerSampleExtension")
        );
        let error = ensure_ttest2_extensions(
            &Value::Num(1.0),
            &Value::Num(1.0),
            &[Value::from("Dim"), integer],
        )
        .expect_err("integer option must gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:Ttest2IntegerOptionExtension")
        );
        drop(_strict);

        let wide = Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1])
            .expect("wide integer");
        let error = ensure_exact_ttest2_integer_tensor(&wide, "sample x")
            .expect_err("lossy sample must reject");
        assert!(error.message().contains("exactly representable"));
    }

    #[test]
    fn ttest2_empty_and_nd_outputs_preserve_shapes() {
        let empty_x = Value::Tensor(Tensor::new(Vec::new(), vec![3, 0]).unwrap());
        let empty_y = Value::Tensor(Tensor::new(Vec::new(), vec![3, 0]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(4));
        let outputs = output_list(block_on(ttest2_builtin(empty_x, empty_y, Vec::new())).unwrap());
        match &outputs[0] {
            Value::LogicalArray(h) => assert_eq!(h.shape, vec![1, 0]),
            other => panic!("expected empty h logical array, got {other:?}"),
        }
        match &outputs[1] {
            Value::Tensor(p) => assert_eq!(p.shape, vec![1, 0]),
            other => panic!("expected empty p tensor, got {other:?}"),
        }
        match &outputs[2] {
            Value::Tensor(ci) => assert_eq!(ci.shape, vec![2, 0]),
            other => panic!("expected empty ci tensor, got {other:?}"),
        }

        drop(_guard);
        let data = (1..=8).map(|value| value as f64).collect::<Vec<_>>();
        let x = Value::Tensor(Tensor::new(data.clone(), vec![2, 1, 4]).unwrap());
        let y = Value::Tensor(
            Tensor::new(
                data.iter().map(|value| value + 1.0).collect(),
                vec![2, 1, 4],
            )
            .unwrap(),
        );
        let _guard = crate::output_count::push_output_count(Some(4));
        let outputs = output_list(block_on(ttest2_builtin(x, y, Vec::new())).unwrap());
        match &outputs[2] {
            Value::Tensor(ci) => assert_eq!(ci.shape, vec![2, 1, 4]),
            other => panic!("expected nd ci tensor, got {other:?}"),
        }
        match &outputs[3] {
            Value::Struct(stats) => match stats.fields.get("sd").unwrap() {
                Value::Tensor(sd) => assert_eq!(sd.shape, vec![1, 1, 4]),
                other => panic!("expected nd sd tensor, got {other:?}"),
            },
            other => panic!("expected stats struct, got {other:?}"),
        }
    }

    #[test]
    fn kstest_default_normal_accepts_centered_sample() {
        let sample =
            Value::Tensor(Tensor::new(vec![-1.0, -0.25, 0.0, 0.25, 1.0], vec![5, 1]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(4));
        let outputs = output_list(block_on(kstest_builtin(sample, Vec::new())).unwrap());
        assert_eq!(outputs[0], Value::Bool(false));
        match outputs[1] {
            Value::Num(p) => assert_close(p, 0.973_180_532_073_205, 1.0e-12),
            ref other => panic!("expected p value, got {other:?}"),
        }
        match outputs[2] {
            Value::Num(stat) => assert_close(stat, 0.201_293_674_317_076, 1.0e-12),
            ref other => panic!("expected statistic, got {other:?}"),
        }
        match outputs[3] {
            Value::Num(cv) => assert_close(cv, 0.607_361_461_908_305, 1.0e-12),
            ref other => panic!("expected critical value, got {other:?}"),
        }
    }

    #[test]
    fn kstest_rejects_shifted_sample_and_supports_tail() {
        let sample = Value::Tensor(Tensor::new(vec![2.0, 2.5, 3.0, 3.5, 4.0], vec![5, 1]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let outputs = output_list(
            block_on(kstest_builtin(
                sample,
                vec![
                    Value::from("Alpha"),
                    Value::Num(0.05),
                    Value::from("Tail"),
                    Value::from("smaller"),
                ],
            ))
            .unwrap(),
        );
        assert_eq!(outputs[0], Value::Bool(true));
        match outputs[1] {
            Value::Num(p) => assert!(p < 0.05),
            ref other => panic!("expected p value, got {other:?}"),
        }

        let sample = Value::Tensor(Tensor::new(vec![-4.0, -3.5, -3.0, -2.5], vec![4, 1]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let outputs = output_list(
            block_on(kstest_builtin(
                sample,
                vec![Value::from("Tail"), Value::from("larger")],
            ))
            .unwrap(),
        );
        assert_eq!(outputs[0], Value::Bool(true));
        match outputs[1] {
            Value::Num(p) => assert!(p < 0.05),
            ref other => panic!("expected p value, got {other:?}"),
        }
    }

    #[test]
    fn kstest_accepts_two_column_cdf_table() {
        let sample = Value::Tensor(Tensor::new(vec![0.2, 0.4, 0.6, 0.8], vec![4, 1]).unwrap());
        let cdf =
            Value::Tensor(Tensor::new(vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0], vec![3, 2]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(4));
        let outputs =
            output_list(block_on(kstest_builtin(sample, vec![Value::from("CDF"), cdf])).unwrap());
        assert_eq!(outputs[0], Value::Bool(false));
        match outputs[2] {
            Value::Num(stat) => assert_close(stat, 0.2, 1.0e-12),
            ref other => panic!("expected statistic, got {other:?}"),
        }

        let sample = Value::Tensor(Tensor::new(vec![0.25, 0.75], vec![2, 1]).unwrap());
        let unsorted_cdf = Value::Tensor(
            Tensor::new(vec![1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.4, 0.5], vec![4, 2]).unwrap(),
        );
        let _guard = crate::output_count::push_output_count(Some(4));
        let outputs = output_list(
            block_on(kstest_builtin(
                sample,
                vec![Value::from("CDF"), unsorted_cdf],
            ))
            .unwrap(),
        );
        assert_eq!(outputs[0], Value::Bool(false));
        match outputs[2] {
            Value::Num(stat) => assert_close(stat, 0.25, 1.0e-12),
            ref other => panic!("expected statistic, got {other:?}"),
        }

        let sample = Value::Tensor(Tensor::new(vec![-0.1, 0.2, 0.4], vec![3, 1]).unwrap());
        let cdf =
            Value::Tensor(Tensor::new(vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0], vec![3, 2]).unwrap());
        let err = block_on(kstest_builtin(sample, vec![Value::from("CDF"), cdf])).unwrap_err();
        assert!(err.message.contains("CDF table range"));
    }

    #[test]
    fn kstest_reads_typed_integer_sample_and_cdf_table_exactly() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let sample = poisoned_integer_tensor(IntegerStorage::U8(vec![0, 1]), vec![2, 1]);
        let cdf = poisoned_integer_tensor(IntegerStorage::U8(vec![0, 1, 0, 1]), vec![2, 2]);
        let cdf =
            Value::Cell(runmat_value::CellArray::new(vec![cdf], 1, 1).expect("singleton CDF cell"));
        let _guard = crate::output_count::push_output_count(Some(4));
        let outputs =
            output_list(block_on(kstest_builtin(sample, vec![Value::from("CDF"), cdf])).unwrap());
        assert_eq!(outputs[0], Value::Bool(false));
        match outputs[2] {
            Value::Num(stat) => assert!(stat.is_finite()),
            ref other => panic!("expected statistic, got {other:?}"),
        }
    }

    #[test]
    fn kstest_gates_integer_roles_before_gather_and_rejects_lossy_values() {
        let sample = poisoned_integer_tensor(IntegerStorage::U8(vec![0, 1]), vec![2, 1]);
        let cdf = poisoned_integer_tensor(IntegerStorage::U8(vec![0, 1, 0, 1]), vec![2, 2]);
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = ensure_kstest_extensions(&sample, &[])
            .expect_err("strict mode must reject integer samples before gather");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:KstestIntegerSampleExtension")
        );
        let err = ensure_kstest_extensions(&Value::Num(0.0), &[Value::from("CDF"), cdf])
            .expect_err("strict mode must reject integer CDF before gather");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:KstestIntegerCdfExtension")
        );
        drop(_strict);

        let wide =
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap();
        let err = ensure_exact_kstest_integer_tensor(&wide, "sample")
            .expect_err("lossy integer sample must reject");
        assert!(err.message.contains("exactly representable"));
    }

    #[test]
    fn kstest_residency_gate_distinguishes_automatic_and_explicit_handles() {
        let automatic = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![2, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered resident fixture")
            .handle;
        let automatic = automatic
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .expect("validated provenance");
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        ensure_kstest_extensions(&Value::GpuTensor(automatic.clone()), &[])
            .expect("automatic residency must retain transparent gather fallback");
        drop(automatic);

        let explicit = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![2, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered resident fixture")
            .handle;
        let explicit = explicit
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let err = ensure_kstest_extensions(&Value::GpuTensor(explicit.clone()), &[])
            .expect_err("explicit gpuArray input must retain its compatibility gate");
        drop(explicit);
        assert_eq!(
            err.identifier(),
            KSTEST_RESIDENT_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn kstest_recursively_classifies_singleton_cell_cdf_inputs() {
        let integer_cdf = poisoned_integer_tensor(IntegerStorage::U8(vec![0, 1, 0, 1]), vec![2, 2]);
        let integer_cdf = Value::Cell(
            runmat_value::CellArray::new(vec![integer_cdf], 1, 1).expect("singleton cell"),
        );
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = ensure_kstest_extensions(&Value::Num(0.0), &[Value::from("CDF"), integer_cdf])
            .expect_err("nested integer CDF must gate");
        assert_eq!(
            err.identifier(),
            KSTEST_INTEGER_CDF_EXTENSION.error_identifier
        );

        let explicit = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![2, 2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered resident fixture")
            .handle;
        let explicit = explicit
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let resident_cdf = Value::Cell(
            runmat_value::CellArray::new(vec![Value::GpuTensor(explicit.clone())], 1, 1)
                .expect("singleton cell"),
        );
        let err = ensure_kstest_extensions(&Value::Num(0.0), &[Value::from("CDF"), resident_cdf])
            .expect_err("nested explicit CDF must gate");
        drop(explicit);
        assert_eq!(
            err.identifier(),
            KSTEST_RESIDENT_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn kstest_rejects_malformed_or_text_only_roles_before_resident_access() {
        let poison = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered resident fixture")
            .handle;
        let poison = poison
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

        let err = block_on(kstest_builtin(
            Value::GpuTensor(poison.clone()),
            vec![Value::from("CDF")],
        ))
        .expect_err("arity must reject before resident sample access");
        assert_eq!(err.identifier(), Some("RunMat:kstest:InvalidArgument"));
        assert!(err.message.contains("supplied in pairs"));

        let sample = Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
        let err = block_on(kstest_builtin(
            sample,
            vec![Value::from("Tail"), Value::GpuTensor(poison.clone())],
        ))
        .expect_err("text-only Tail must reject before resident access");
        drop(poison);
        assert_eq!(err.identifier(), Some("RunMat:kstest:InvalidArgument"));
        assert!(err.message.contains("Tail must be a string scalar"));
    }

    #[test]
    fn kstest_rejects_bad_options_cdf_and_outputs() {
        let sample = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap());
        let err = block_on(kstest_builtin(
            sample.clone(),
            vec![Value::from("Tail"), Value::from("sideways")],
        ))
        .unwrap_err();
        assert!(err.message.contains("Tail"));

        let cdf = Value::Tensor(Tensor::new(vec![0.0, 1.0, 0.8, 0.2], vec![2, 2]).unwrap());
        let err = block_on(kstest_builtin(
            sample.clone(),
            vec![Value::from("CDF"), cdf],
        ))
        .unwrap_err();
        assert!(err.message.contains("nondecreasing"));

        let matrix = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap());
        let err = block_on(kstest_builtin(matrix, Vec::new())).unwrap_err();
        assert!(err.message.contains("vector"));

        let empty = Value::Tensor(Tensor::new(vec![], vec![0, 1]).unwrap());
        let err = block_on(kstest_builtin(empty, Vec::new())).unwrap_err();
        assert!(err.message.contains("non-NaN"));

        let all_nan = Value::Tensor(Tensor::new(vec![f64::NAN, f64::NAN], vec![2, 1]).unwrap());
        let err = block_on(kstest_builtin(all_nan, Vec::new())).unwrap_err();
        assert!(err.message.contains("non-NaN"));

        let _guard = crate::output_count::push_output_count(Some(4));
        let outputs = output_list(
            block_on(kstest_builtin(
                sample.clone(),
                vec![Value::from("Alpha"), Value::Num(0.5)],
            ))
            .unwrap(),
        );
        match outputs[3] {
            Value::Num(cv) => assert!(cv.is_nan()),
            ref other => panic!("expected cv, got {other:?}"),
        }
        drop(_guard);

        let _guard = crate::output_count::push_output_count(Some(4));
        let outputs = output_list(
            block_on(kstest_builtin(
                sample.clone(),
                vec![
                    Value::from("Alpha"),
                    Value::Num(0.005),
                    Value::from("Tail"),
                    Value::from("smaller"),
                ],
            ))
            .unwrap(),
        );
        match outputs[3] {
            Value::Num(cv) => assert!(cv.is_finite()),
            ref other => panic!("expected cv, got {other:?}"),
        }
        drop(_guard);

        let _guard = crate::output_count::push_output_count(Some(4));
        let outputs = output_list(
            block_on(kstest_builtin(
                sample.clone(),
                vec![
                    Value::from("Alpha"),
                    Value::Num(0.15),
                    Value::from("Tail"),
                    Value::from("smaller"),
                ],
            ))
            .unwrap(),
        );
        match outputs[3] {
            Value::Num(cv) => assert!(cv.is_nan()),
            ref other => panic!("expected cv, got {other:?}"),
        }
        drop(_guard);

        let _guard = crate::output_count::push_output_count(Some(5));
        let err = block_on(kstest_builtin(sample, Vec::new())).unwrap_err();
        assert!(err.message.contains("too many output arguments"));
    }

    #[test]
    fn kstest_integer_alpha_follows_the_ordinary_open_interval_rule() {
        let sample = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap());
        let err = block_on(kstest_builtin(
            sample,
            vec![
                Value::from("Alpha"),
                poisoned_integer_scalar(IntegerStorage::U64(vec![1])),
            ],
        ))
        .unwrap_err();
        assert!(err.message.contains("open interval"));
    }
}
