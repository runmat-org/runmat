//! Hypothesis-test compatibility helpers.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    LogicalArray, ResolveContext, StructValue, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor;
use crate::builtins::stats::summary::distribution_math::{
    student_t_cdf, student_t_cdf_upper, student_t_inv,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "ttest2";
const KSTEST_NAME: &str = "kstest";
const MAX_SUPPORTED_DIM: usize = 1024;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

const TTEST2_INTEGER_SAMPLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ttest2-integer-sample",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ttest2 with native-class integer sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Ttest2IntegerSampleExtension"),
};
const TTEST2_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ttest2-integer-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ttest2 with native-class integer numeric options is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Ttest2IntegerOptionExtension"),
};
pub const TTEST2_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    TTEST2_INTEGER_SAMPLE_EXTENSION,
    TTEST2_INTEGER_OPTION_EXTENSION,
];

const TTEST2_INTEGER_SAMPLE_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Native integer samples are gated before gather and checked for exact binary64 representation.",
    },
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Native integer samples are gated before gather and checked for exact binary64 representation.",
    },
];
const TTEST2_INTEGER_OPTION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Alpha or Dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Native integer numeric options are independently gated and checked before scalar conversion.",
    }];
pub const TTEST2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = ttest2(integer_x, integer_y, ___)",
        inputs: &TTEST2_INTEGER_SAMPLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "h is logical and the remaining outputs are floating point. Lossy integer sample conversion rejects.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = ttest2(x, y, Alpha or Dim, integer_value)",
        inputs: &TTEST2_INTEGER_OPTION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Dim is decoded as a checked structural selector; Alpha crosses one checked binary64 boundary.",
    },
];

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

const OUTPUT_CI: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ci",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Confidence interval for the difference in means.",
};

const OUTPUT_STATS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "stats",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Structure containing tstat, df, and sd fields.",
};

const OUTPUT_H_ONLY: [BuiltinParamDescriptor; 1] = [OUTPUT_H];
const OUTPUT_H_P: [BuiltinParamDescriptor; 2] = [OUTPUT_H, OUTPUT_P];
const OUTPUT_FULL: [BuiltinParamDescriptor; 4] = [OUTPUT_H, OUTPUT_P, OUTPUT_CI, OUTPUT_STATS];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "First sample data.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second sample data.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Alpha, Dim, Tail, and Vartype options.",
};

const INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUTS_X_Y_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_OPTIONS];

const TTEST2_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = ttest2(x, y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_H_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "h = ttest2(x, y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_H_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "[h, p] = ttest2(___)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_H_P,
    },
    BuiltinSignatureDescriptor {
        label: "[h, p, ci, stats] = ttest2(___)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_FULL,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TTEST2.INVALID_ARGUMENT",
    identifier: Some("RunMat:ttest2:InvalidArgument"),
    when: "Inputs, dimensions, tails, variance type, or significance level are malformed.",
    message: "ttest2: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TTEST2.INTERNAL",
    identifier: Some("RunMat:ttest2:Internal"),
    when: "RunMat cannot construct ttest2 outputs.",
    message: "ttest2: internal error",
};

const TTEST2_ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const TTEST2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TTEST2_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TTEST2_ERRORS,
};

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

fn ttest2_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

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
    ttest2_error(message, &ERROR_INVALID_ARGUMENT)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    ttest2_error(message, &ERROR_INTERNAL)
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
enum Tail {
    Both,
    Right,
    Left,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VarType {
    Equal,
    Unequal,
}

#[derive(Clone, Copy, Debug)]
struct Options {
    alpha: f64,
    dim: Option<usize>,
    tail: Tail,
    vartype: VarType,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            dim: None,
            tail: Tail::Both,
            vartype: VarType::Equal,
        }
    }
}

#[derive(Clone, Debug)]
struct Evaluation {
    h: Value,
    p: Value,
    ci: Value,
    stats: Value,
}

#[derive(Clone, Debug)]
struct TestResult {
    h: bool,
    p: f64,
    ci_low: f64,
    ci_high: f64,
    tstat: f64,
    df: f64,
    sd_equal: f64,
    sd_x: f64,
    sd_y: f64,
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
    category = "stats/summary",
    summary = "Perform a two-sample Student's t-test.",
    keywords = "ttest2,t-test,hypothesis test,welch,equal variance,statistics",
    type_resolver(ttest2_type),
    descriptor(crate::builtins::stats::summary::hypothesis::TTEST2_DESCRIPTOR),
    extensions(crate::builtins::stats::summary::hypothesis::TTEST2_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::summary::hypothesis::TTEST2_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::hypothesis"
)]
pub(crate) async fn ttest2_builtin(x: Value, y: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_ttest2_extensions(&x, &y, &rest)?;
    let x = gathered_tensor(x).await?;
    let y = gathered_tensor(y).await?;
    ensure_exact_ttest2_integer_tensor(&x, "sample x")?;
    ensure_exact_ttest2_integer_tensor(&y, "sample y")?;
    let rest = gather_values(rest).await?;
    let options = parse_options(rest)?;
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
                .shape
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
            if !crate::builtins::math::trigonometry::cos::integer_is_exact_f64(&integer) {
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

fn outputs_for_count(out_count: usize, eval: Evaluation) -> BuiltinResult<Value> {
    match out_count {
        0 => Ok(Value::OutputList(Vec::new())),
        1 => Ok(Value::OutputList(vec![eval.h])),
        2 => Ok(Value::OutputList(vec![eval.h, eval.p])),
        3 => Ok(crate::output_count::output_list_with_padding(
            3,
            vec![eval.h, eval.p, eval.ci, eval.stats],
        )),
        4 => Ok(Value::OutputList(vec![eval.h, eval.p, eval.ci, eval.stats])),
        _ => Err(invalid_argument(
            "ttest2: too many output arguments; maximum is 4",
        )),
    }
}

async fn gathered_tensor(value: Value) -> BuiltinResult<Tensor> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_argument(format!("ttest2: {err}")))?;
    tensor::value_into_tensor_for(BUILTIN_NAME, value).map_err(invalid_argument)
}

fn ttest2_is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn ttest2_plain_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

fn ensure_ttest2_extensions(x: &Value, y: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if ttest2_is_typed_integer(x) || ttest2_is_typed_integer(y) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TTEST2_INTEGER_SAMPLE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    for pair in rest.chunks_exact(2) {
        if ttest2_plain_text(&pair[0])
            .is_some_and(|name| matches!(name.to_ascii_lowercase().as_str(), "alpha" | "dim"))
            && ttest2_is_typed_integer(&pair[1])
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &TTEST2_INTEGER_OPTION_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    Ok(())
}

fn ensure_exact_ttest2_integer_tensor(tensor: &Tensor, role: &str) -> BuiltinResult<()> {
    if let Some(storage) = tensor.integer_storage() {
        for integer in storage.exact_values() {
            if !crate::builtins::math::trigonometry::cos::integer_is_exact_f64(&integer) {
                return Err(invalid_argument(format!(
                    "ttest2: integer {role} values must be exactly representable as double"
                )));
            }
        }
    }
    Ok(())
}

fn ensure_exact_ttest2_integer_scalar(value: &Value, role: &str) -> BuiltinResult<()> {
    match value {
        Value::Int(integer) => {
            if crate::builtins::math::trigonometry::cos::integer_is_exact_f64(integer) {
                Ok(())
            } else {
                Err(invalid_argument(format!(
                    "ttest2: integer {role} must be exactly representable as double"
                )))
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            ensure_exact_ttest2_integer_tensor(tensor, role)
        }
        _ => Ok(()),
    }
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(
            gather_if_needed_async(&value)
                .await
                .map_err(|err| invalid_argument(format!("ttest2: {err}")))?,
        );
    }
    Ok(out)
}

fn parse_options(rest: Vec<Value>) -> BuiltinResult<Options> {
    let mut options = Options::default();
    let mut idx = 0;
    while idx < rest.len() {
        let name = scalar_text(&rest[idx], "option name")?;
        idx += 1;
        if idx >= rest.len() {
            return Err(invalid_argument(format!(
                "ttest2: option '{name}' requires a value"
            )));
        }
        let value = &rest[idx];
        idx += 1;
        match name.to_ascii_lowercase().as_str() {
            "alpha" => {
                ensure_exact_ttest2_integer_scalar(value, "Alpha")?;
                let alpha = scalar_number(value)
                    .ok_or_else(|| invalid_argument("ttest2: Alpha must be numeric"))?;
                if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
                    return Err(invalid_argument(
                        "ttest2: Alpha must be a scalar in the open interval (0,1)",
                    ));
                }
                options.alpha = alpha;
            }
            "dim" => {
                let dim = scalar_usize(value)
                    .ok_or_else(|| invalid_argument("ttest2: Dim must be a positive integer"))?;
                if dim == 0 {
                    return Err(invalid_argument("ttest2: Dim must be a positive integer"));
                }
                if dim > MAX_SUPPORTED_DIM {
                    return Err(invalid_argument(format!(
                        "ttest2: Dim exceeds the supported maximum of {MAX_SUPPORTED_DIM}"
                    )));
                }
                options.dim = Some(dim - 1);
            }
            "tail" => {
                let tail = scalar_text(value, "Tail")?;
                options.tail = match tail.to_ascii_lowercase().as_str() {
                    "both" => Tail::Both,
                    "right" | "larger" => Tail::Right,
                    "left" | "smaller" => Tail::Left,
                    other => {
                        return Err(invalid_argument(format!(
                            "ttest2: unsupported Tail '{other}'"
                        )))
                    }
                };
            }
            "vartype" => {
                let vartype = scalar_text(value, "Vartype")?;
                options.vartype = match vartype.to_ascii_lowercase().as_str() {
                    "equal" => VarType::Equal,
                    "unequal" => VarType::Unequal,
                    other => {
                        return Err(invalid_argument(format!(
                            "ttest2: unsupported Vartype '{other}'"
                        )))
                    }
                };
            }
            other => {
                return Err(invalid_argument(format!(
                    "ttest2: unsupported option '{other}'"
                )))
            }
        }
    }
    Ok(options)
}

fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::Cell(cell) if cell.data.len() == 1 => scalar_text(&cell.data[0], context),
        other => Err(invalid_argument(format!(
            "ttest2: {context} must be a string scalar, got {other:?}"
        ))),
    }
}

fn scalar_number(value: &Value) -> Option<f64> {
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

fn scalar_usize(value: &Value) -> Option<usize> {
    match value {
        Value::Int(value) => value.try_to_usize(),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                storage.value_at(0)?.try_to_usize()
            } else {
                let value = tensor::tensor_values_f64(tensor).first().copied()?;
                (value.is_finite()
                    && value >= 0.0
                    && value.fract() == 0.0
                    && value < usize::MAX as f64)
                    .then_some(value as usize)
            }
        }
        Value::Num(value)
            if value.is_finite()
                && *value >= 0.0
                && value.fract() == 0.0
                && *value < usize::MAX as f64 =>
        {
            Some(*value as usize)
        }
        Value::Bool(value) => Some(usize::from(*value)),
        _ => None,
    }
}

fn evaluate(x: &Tensor, y: &Tensor, options: Options) -> BuiltinResult<Evaluation> {
    let vector_collapse = options.dim.is_none()
        && is_nonempty_vector_shape(&x.shape)
        && is_nonempty_vector_shape(&y.shape);
    let dim = if vector_collapse {
        0
    } else {
        options
            .dim
            .unwrap_or_else(|| first_nonsingleton_dim(&x.shape))
    };
    let rank = x.shape.len().max(y.shape.len()).max(dim + 1).max(2);
    let x_shape = normalized_shape(&x.shape, rank);
    let y_shape = normalized_shape(&y.shape, rank);
    let out_shape = if vector_collapse {
        vec![1, 1]
    } else {
        compatible_output_shape(&x_shape, &y_shape, dim)?
    };
    let tests = tensor::element_count(&out_shape);
    let mut h_data = Vec::with_capacity(tests);
    let mut p_data = Vec::with_capacity(tests);
    let mut ci_data = Vec::with_capacity(tests * 2);
    let mut tstat_data = Vec::with_capacity(tests);
    let mut df_data = Vec::with_capacity(tests);
    let mut sd_equal_data = Vec::with_capacity(tests);
    let mut sd_x_data = Vec::with_capacity(tests);
    let mut sd_y_data = Vec::with_capacity(tests);
    let x_values = x.materialize_f64();
    let y_values = y.materialize_f64();

    for linear in 0..tests {
        let result = if vector_collapse {
            evaluate_pair(&x_values, &y_values, options)
        } else {
            let coords = coords_from_linear(linear, &out_shape);
            let xs = sample_along_dim(&x_values, &x_shape, &coords, dim);
            let ys = sample_along_dim(&y_values, &y_shape, &coords, dim);
            evaluate_pair(&xs, &ys, options)
        };
        h_data.push(u8::from(result.h));
        p_data.push(result.p);
        ci_data.push(result.ci_low);
        ci_data.push(result.ci_high);
        tstat_data.push(result.tstat);
        df_data.push(result.df);
        sd_equal_data.push(result.sd_equal);
        sd_x_data.push(result.sd_x);
        sd_y_data.push(result.sd_y);
    }

    let h = logical_output(h_data, out_shape.clone())?;
    let p = numeric_output(p_data, out_shape.clone())?;
    let ci_shape = interval_shape(&out_shape, dim);
    let ci = Value::Tensor(Tensor::new(ci_data, ci_shape).map_err(internal_error)?);
    let mut stats = StructValue::new();
    stats.insert("tstat", numeric_output(tstat_data, out_shape.clone())?);
    stats.insert("df", numeric_output(df_data, out_shape.clone())?);
    if options.vartype == VarType::Equal {
        stats.insert("sd", numeric_output(sd_equal_data, out_shape)?);
    } else {
        let sd_shape = interval_shape(&out_shape, dim);
        let mut sd_data = Vec::with_capacity(tests * 2);
        for idx in 0..tests {
            sd_data.push(sd_x_data[idx]);
            sd_data.push(sd_y_data[idx]);
        }
        stats.insert(
            "sd",
            Value::Tensor(Tensor::new(sd_data, sd_shape).map_err(internal_error)?),
        );
    }
    Ok(Evaluation {
        h,
        p,
        ci,
        stats: Value::Struct(stats),
    })
}

fn compatible_output_shape(
    x_shape: &[usize],
    y_shape: &[usize],
    dim: usize,
) -> BuiltinResult<Vec<usize>> {
    let mut out = Vec::with_capacity(x_shape.len());
    for axis in 0..x_shape.len() {
        if axis == dim {
            out.push(1);
            continue;
        }
        if x_shape[axis] != y_shape[axis] {
            return Err(invalid_argument(
                "ttest2: X and Y must have the same size along all dimensions except Dim",
            ));
        }
        out.push(x_shape[axis]);
    }
    Ok(out)
}

fn evaluate_pair(x: &[f64], y: &[f64], options: Options) -> TestResult {
    let xs = x
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    let ys = y
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    if xs.len() < 2 || ys.len() < 2 {
        return invalid_result();
    }
    let nx = xs.len() as f64;
    let ny = ys.len() as f64;
    let mean_x = mean(&xs);
    let mean_y = mean(&ys);
    let var_x = variance(&xs);
    let var_y = variance(&ys);
    if !var_x.is_finite() || !var_y.is_finite() {
        return invalid_result();
    }
    let diff = mean_x - mean_y;
    let (se, df, sd_equal, sd_x, sd_y) = match options.vartype {
        VarType::Equal => {
            let df = nx + ny - 2.0;
            if df <= 0.0 {
                return invalid_result();
            }
            let pooled_var = ((nx - 1.0) * var_x + (ny - 1.0) * var_y) / df;
            let sd = pooled_var.sqrt();
            (
                sd * (1.0 / nx + 1.0 / ny).sqrt(),
                df,
                sd,
                var_x.sqrt(),
                var_y.sqrt(),
            )
        }
        VarType::Unequal => {
            let ax = var_x / nx;
            let ay = var_y / ny;
            let denom = ax * ax / (nx - 1.0) + ay * ay / (ny - 1.0);
            let df = if denom == 0.0 {
                f64::INFINITY
            } else {
                (ax + ay) * (ax + ay) / denom
            };
            ((ax + ay).sqrt(), df, f64::NAN, var_x.sqrt(), var_y.sqrt())
        }
    };
    let tstat = if se == 0.0 {
        if diff == 0.0 {
            0.0
        } else {
            diff.signum() * f64::INFINITY
        }
    } else {
        diff / se
    };
    let p = p_value(tstat, df, options.tail);
    let (ci_low, ci_high) = confidence_interval(diff, se, df, options);
    TestResult {
        h: p.is_finite() && p <= options.alpha,
        p,
        ci_low,
        ci_high,
        tstat,
        df,
        sd_equal,
        sd_x,
        sd_y,
    }
}

fn invalid_result() -> TestResult {
    TestResult {
        h: false,
        p: f64::NAN,
        ci_low: f64::NAN,
        ci_high: f64::NAN,
        tstat: f64::NAN,
        df: f64::NAN,
        sd_equal: f64::NAN,
        sd_x: f64::NAN,
        sd_y: f64::NAN,
    }
}

fn p_value(tstat: f64, df: f64, tail: Tail) -> f64 {
    if tstat.is_nan() || df.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    match tail {
        Tail::Both => (2.0
            * student_t_cdf(tstat.abs(), df).min(student_t_cdf_upper(tstat.abs(), df)))
        .min(1.0),
        Tail::Right => student_t_cdf_upper(tstat, df),
        Tail::Left => student_t_cdf(tstat, df),
    }
}

fn confidence_interval(diff: f64, se: f64, df: f64, options: Options) -> (f64, f64) {
    if diff.is_nan() || se.is_nan() || df.is_nan() || df <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    match options.tail {
        Tail::Both => {
            let crit = student_t_inv(1.0 - options.alpha / 2.0, df);
            (diff - crit * se, diff + crit * se)
        }
        Tail::Right => {
            let crit = student_t_inv(1.0 - options.alpha, df);
            (diff - crit * se, f64::INFINITY)
        }
        Tail::Left => {
            let crit = student_t_inv(1.0 - options.alpha, df);
            (f64::NEG_INFINITY, diff + crit * se)
        }
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let mean = mean(values);
    values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64
}

fn normalized_shape(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut out = shape.to_vec();
    out.resize(rank, 1);
    out
}

fn first_nonsingleton_dim(shape: &[usize]) -> usize {
    shape.iter().position(|dim| *dim != 1).unwrap_or(0)
}

fn is_nonempty_vector_shape(shape: &[usize]) -> bool {
    shape.iter().all(|dim| *dim > 0) && shape.iter().filter(|dim| **dim > 1).count() <= 1
}

fn sample_along_dim(data: &[f64], shape: &[usize], out_coords: &[usize], dim: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(shape[dim]);
    let mut coords = out_coords.to_vec();
    coords.resize(shape.len(), 0);
    for idx in 0..shape[dim] {
        coords[dim] = idx;
        values.push(data[linear_index(&coords, shape)]);
    }
    values
}

fn coords_from_linear(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(shape.len());
    for dim in shape {
        coords.push(linear % *dim);
        linear /= *dim;
    }
    coords
}

fn linear_index(coords: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1;
    let mut index = 0;
    for (coord, dim) in coords.iter().zip(shape) {
        index += coord * stride;
        stride *= dim;
    }
    index
}

fn logical_output(data: Vec<u8>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if data.len() == 1 {
        Ok(Value::Bool(data[0] != 0))
    } else {
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(internal_error)
    }
}

fn numeric_output(data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(internal_error)
}

fn interval_shape(out_shape: &[usize], dim: usize) -> Vec<usize> {
    let mut shape = vec![2];
    for (axis, extent) in out_shape.iter().copied().enumerate() {
        if axis != dim {
            shape.push(extent);
        }
    }
    if shape.len() == 1 {
        shape.push(1);
    }
    shape
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

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
        let cdf = Value::Cell(
            runmat_builtins::CellArray::new(vec![cdf], 1, 1).expect("singleton CDF cell"),
        );
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
        let automatic = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![2, 1],
            device_id: 0,
            buffer_id: 9_421_001,
        };
        runmat_accelerate_api::mark_handle_automatic(&automatic);
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        ensure_kstest_extensions(&Value::GpuTensor(automatic.clone()), &[])
            .expect("automatic residency must retain transparent gather fallback");
        runmat_accelerate_api::clear_handle_metadata(&automatic);

        let explicit = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![2, 1],
            device_id: 0,
            buffer_id: 9_421_002,
        };
        runmat_accelerate_api::mark_handle_explicit(&explicit);
        let err = ensure_kstest_extensions(&Value::GpuTensor(explicit.clone()), &[])
            .expect_err("explicit gpuArray input must retain its compatibility gate");
        runmat_accelerate_api::clear_handle_metadata(&explicit);
        assert_eq!(
            err.identifier(),
            KSTEST_RESIDENT_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn kstest_recursively_classifies_singleton_cell_cdf_inputs() {
        let integer_cdf = poisoned_integer_tensor(IntegerStorage::U8(vec![0, 1, 0, 1]), vec![2, 2]);
        let integer_cdf = Value::Cell(
            runmat_builtins::CellArray::new(vec![integer_cdf], 1, 1).expect("singleton cell"),
        );
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = ensure_kstest_extensions(&Value::Num(0.0), &[Value::from("CDF"), integer_cdf])
            .expect_err("nested integer CDF must gate");
        assert_eq!(
            err.identifier(),
            KSTEST_INTEGER_CDF_EXTENSION.error_identifier
        );

        let explicit = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![2, 2],
            device_id: 0,
            buffer_id: 9_421_003,
        };
        runmat_accelerate_api::mark_handle_explicit(&explicit);
        let resident_cdf = Value::Cell(
            runmat_builtins::CellArray::new(vec![Value::GpuTensor(explicit.clone())], 1, 1)
                .expect("singleton cell"),
        );
        let err = ensure_kstest_extensions(&Value::Num(0.0), &[Value::from("CDF"), resident_cdf])
            .expect_err("nested explicit CDF must gate");
        runmat_accelerate_api::clear_handle_metadata(&explicit);
        assert_eq!(
            err.identifier(),
            KSTEST_RESIDENT_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn kstest_rejects_malformed_or_text_only_roles_before_resident_access() {
        let poison = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX - 9_421,
        };
        runmat_accelerate_api::mark_handle_explicit(&poison);
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
        runmat_accelerate_api::clear_handle_metadata(&poison);
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
