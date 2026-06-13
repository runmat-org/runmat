//! MATLAB-compatible `tf` transfer-function constructor and SISO operator methods for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{
    control_error, is_discrete_variable, parse_coefficients, scalar_f64, scalar_text,
    two_models_ordered, validate_sample_time, validate_variable, validate_variable_domain, TfModel,
    TfOptions,
};
use crate::builtins::control::type_resolvers::tf_type;
use crate::{build_runtime_error, dispatcher, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "tf";
const DEFAULT_VARIABLE: &str = "s";

const TF_OUTPUT_SYS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO transfer-function object.",
}];
const TF_PARAM_NUMERATOR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "numerator",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numerator coefficient vector.",
};
const TF_PARAM_DENOMINATOR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "denominator",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Denominator coefficient vector.",
};
const TF_PARAM_VARIABLE_SYMBOL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "variable",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Transfer-function indeterminate ('s', 'p', 'z', 'q', 'z^-1', or 'q^-1').",
};
const TF_PARAM_TS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Ts",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("0.0"),
    description: "Sample time (0 for continuous-time model).",
};
const TF_INPUTS_VARIABLE: [BuiltinParamDescriptor; 1] = [TF_PARAM_VARIABLE_SYMBOL];
const TF_INPUTS_VARIABLE_TS: [BuiltinParamDescriptor; 2] = [TF_PARAM_VARIABLE_SYMBOL, TF_PARAM_TS];
const TF_INPUTS_NUM_DEN: [BuiltinParamDescriptor; 2] = [TF_PARAM_NUMERATOR, TF_PARAM_DENOMINATOR];
const TF_INPUTS_NUM_DEN_TS: [BuiltinParamDescriptor; 3] =
    [TF_PARAM_NUMERATOR, TF_PARAM_DENOMINATOR, TF_PARAM_TS];
const TF_INPUTS_NUM_DEN_NAMEVALUE: [BuiltinParamDescriptor; 4] = [
    TF_PARAM_NUMERATOR,
    TF_PARAM_DENOMINATOR,
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option name ('Variable' or 'Ts').",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value.",
    },
];
const TF_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "s = tf('s')",
        inputs: &TF_INPUTS_VARIABLE,
        outputs: &TF_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "z = tf('z', Ts)",
        inputs: &TF_INPUTS_VARIABLE_TS,
        outputs: &TF_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = tf(numerator, denominator)",
        inputs: &TF_INPUTS_NUM_DEN,
        outputs: &TF_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = tf(numerator, denominator, Ts)",
        inputs: &TF_INPUTS_NUM_DEN_TS,
        outputs: &TF_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = tf(numerator, denominator, \"Variable\", variableName)",
        inputs: &TF_INPUTS_NUM_DEN_NAMEVALUE,
        outputs: &TF_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = tf(numerator, denominator, name, value, ...)",
        inputs: &TF_INPUTS_NUM_DEN_NAMEVALUE,
        outputs: &TF_OUTPUT_SYS,
    },
];
const TF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.INVALID_ARGUMENT",
    identifier: Some("RunMat:tf:InvalidArgument"),
    when: "Arguments do not match supported tf invocation forms.",
    message: "tf: invalid argument",
};
const TF_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.INVALID_OPTION",
    identifier: Some("RunMat:tf:InvalidOption"),
    when: "A name/value option token is unsupported or malformed.",
    message: "tf: invalid option",
};
const TF_ERROR_INVALID_SAMPLE_TIME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.INVALID_SAMPLE_TIME",
    identifier: Some("RunMat:tf:InvalidSampleTime"),
    when: "Sample time is not a finite non-negative scalar.",
    message: "tf: sample time must be a finite non-negative scalar",
};
const TF_ERROR_INVALID_VARIABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.INVALID_VARIABLE",
    identifier: Some("RunMat:tf:InvalidVariable"),
    when: "Variable option is not a supported control variable name.",
    message: "tf: invalid Variable option",
};
const TF_ERROR_INVALID_COEFFICIENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.INVALID_COEFFICIENTS",
    identifier: Some("RunMat:tf:InvalidCoefficients"),
    when: "Numerator/denominator coefficients are not valid finite vectors.",
    message: "tf: invalid coefficients",
};
const TF_ERROR_DENOMINATOR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.DENOMINATOR_INVALID",
    identifier: Some("RunMat:tf:DenominatorInvalid"),
    when: "Denominator coefficient vector is empty or all zeros.",
    message: "tf: invalid denominator coefficients",
};
const TF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.INTERNAL",
    identifier: Some("RunMat:tf:Internal"),
    when: "Internal tensor/object construction failed.",
    message: "tf: internal error",
};
const TF_ERRORS: [BuiltinErrorDescriptor; 7] = [
    TF_ERROR_INVALID_ARGUMENT,
    TF_ERROR_INVALID_OPTION,
    TF_ERROR_INVALID_SAMPLE_TIME,
    TF_ERROR_INVALID_VARIABLE,
    TF_ERROR_INVALID_COEFFICIENTS,
    TF_ERROR_DENOMINATOR_INVALID,
    TF_ERROR_INTERNAL,
];
pub const TF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TF_ERRORS,
};

const TF_METHOD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Resulting SISO transfer-function object.",
}];
const TF_METHOD_INPUTS_BINARY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "lhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left operand.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right operand.",
    },
];
const TF_METHOD_INPUTS_UNARY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Operand.",
}];
const TF_METHOD_SIGNATURES_BINARY: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "sys = tf.operator(lhs, rhs)",
    inputs: &TF_METHOD_INPUTS_BINARY,
    outputs: &TF_METHOD_OUTPUT,
}];
const TF_METHOD_SIGNATURES_UNARY: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "sys = tf.operator(sys)",
    inputs: &TF_METHOD_INPUTS_UNARY,
    outputs: &TF_METHOD_OUTPUT,
}];
const TF_METHOD_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.OPERATOR",
    identifier: Some("RunMat:tf:OperatorError"),
    when: "Transfer-function operator arguments are invalid or incompatible.",
    message: "tf operator failed",
};
const TF_METHOD_ERRORS: [BuiltinErrorDescriptor; 1] = [TF_METHOD_ERROR];
pub const TF_METHOD_BINARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TF_METHOD_SIGNATURES_BINARY,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &TF_METHOD_ERRORS,
};
pub const TF_METHOD_UNARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TF_METHOD_SIGNATURES_UNARY,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &TF_METHOD_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::tf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tf",
    op_kind: GpuOpKind::Custom("transfer-function-constructor"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Object construction runs on the host. gpuArray coefficient inputs are gathered before storing the transfer-function metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::tf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tf",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Transfer-function construction is metadata-only and terminates numeric fusion chains.",
};

fn tf_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    tf_error_with_message(error.message, error)
}

fn tf_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    tf_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn tf_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "tf",
    category = "control",
    summary = "Create SISO transfer-function objects from numerator and denominator coefficients.",
    keywords = "tf,transfer function,control system,filter,polynomial",
    type_resolver(tf_type),
    descriptor(crate::builtins::control::tf::TF_DESCRIPTOR),
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    match args.as_slice() {
        [variable] if is_text_value(variable) => variable_model(variable, None)?.to_value("tf"),
        [variable, sample_time] if is_text_value(variable) => {
            variable_model(variable, Some(sample_time))?.to_value("tf")
        }
        [numerator, denominator, rest @ ..] => {
            let options = TfConstructorOptions::parse(rest)?;
            let numerator = parse_coefficients("numerator", numerator.clone(), "tf").await?;
            let denominator = parse_coefficients("denominator", denominator.clone(), "tf").await?;
            TfModel::new(
                numerator,
                denominator,
                TfOptions {
                    variable: options.variable,
                    sample_time: options.sample_time,
                },
            )?
            .to_value("tf")
        }
        [] => Err(tf_error_with_detail(
            &TF_ERROR_INVALID_ARGUMENT,
            "expected tf('s'), tf(num, den), or tf(num, den, ...)",
        )),
        _ => Err(tf_error_with_detail(
            &TF_ERROR_INVALID_ARGUMENT,
            "unsupported tf invocation",
        )),
    }
}

#[derive(Clone)]
struct TfConstructorOptions {
    variable: String,
    sample_time: f64,
    variable_explicit: bool,
    sample_time_explicit: bool,
}

impl TfConstructorOptions {
    fn parse(rest: &[Value]) -> BuiltinResult<Self> {
        let mut options = Self {
            variable: DEFAULT_VARIABLE.to_string(),
            sample_time: 0.0,
            variable_explicit: false,
            sample_time_explicit: false,
        };

        match rest {
            [] => {}
            [sample_time] => {
                options.sample_time = parse_sample_time(sample_time)?;
                options.sample_time_explicit = true;
                if options.sample_time > 0.0 {
                    options.variable = "z".to_string();
                }
            }
            _ => {
                if !rest.len().is_multiple_of(2) {
                    return Err(tf_error_with_detail(
                        &TF_ERROR_INVALID_ARGUMENT,
                        "optional arguments must be name-value pairs or a scalar sample time",
                    ));
                }
                let mut idx = 0;
                while idx < rest.len() {
                    let name = scalar_text(&rest[idx], "option name", "tf")?;
                    let lowered = name.trim().to_ascii_lowercase();
                    let value = &rest[idx + 1];
                    match lowered.as_str() {
                        "variable" => {
                            options.variable = parse_variable(value)?;
                            options.variable_explicit = true;
                        }
                        "ts" | "sampletime" => {
                            options.sample_time = parse_sample_time(value)?;
                            options.sample_time_explicit = true;
                        }
                        _ => {
                            return Err(tf_error_with_detail(
                                &TF_ERROR_INVALID_OPTION,
                                format!("unsupported option '{name}'"),
                            ));
                        }
                    }
                    idx += 2;
                }
                if options.sample_time > 0.0 && !options.variable_explicit {
                    options.variable = "z".to_string();
                }
            }
        }

        if options.variable_explicit
            && is_discrete_variable(&options.variable)
            && !options.sample_time_explicit
        {
            options.sample_time = 1.0;
        }
        validate_variable_domain(&options.variable, options.sample_time, "tf").map_err(|err| {
            let identifier = err.identifier();
            if identifier == TF_ERROR_INVALID_SAMPLE_TIME.identifier {
                tf_error_with_detail(&TF_ERROR_INVALID_SAMPLE_TIME, err.message())
            } else {
                tf_error_with_detail(&TF_ERROR_INVALID_VARIABLE, err.message())
            }
        })?;
        Ok(options)
    }
}

fn parse_sample_time(value: &Value) -> BuiltinResult<f64> {
    let sample_time = scalar_f64(value, "sample time", "tf").map_err(|_| {
        tf_error_with_detail(
            &TF_ERROR_INVALID_SAMPLE_TIME,
            format!("expected non-negative scalar, got {value:?}"),
        )
    })?;
    if let Err(err) = validate_sample_time(sample_time, "tf") {
        let _ = err;
        return Err(tf_error(&TF_ERROR_INVALID_SAMPLE_TIME));
    }
    Ok(sample_time)
}

fn parse_variable(value: &Value) -> BuiltinResult<String> {
    let variable = scalar_text(value, "Variable", "tf")?;
    validate_variable(&variable, "tf").map_err(|_| {
        tf_error_with_detail(
            &TF_ERROR_INVALID_VARIABLE,
            "must be one of 's', 'p', 'z', 'q', 'z^-1', or 'q^-1'",
        )
    })
}

fn is_text_value(value: &Value) -> bool {
    match value {
        Value::String(_) => true,
        Value::StringArray(array) => array.data.len() == 1,
        Value::CharArray(array) => array.rows == 1,
        _ => false,
    }
}

fn variable_model(value: &Value, sample_time: Option<&Value>) -> BuiltinResult<TfModel> {
    let variable = parse_variable(value)?;
    match variable.as_str() {
        "s" | "p" => {
            if let Some(sample_time) = sample_time {
                let parsed = parse_sample_time(sample_time)?;
                if parsed > 0.0 {
                    return Err(tf_error_with_detail(
                        &TF_ERROR_INVALID_SAMPLE_TIME,
                        "continuous transfer-function variables require Ts = 0",
                    ));
                }
            }
            TfModel::continuous_variable(variable)
        }
        "z" | "q" | "z^-1" | "q^-1" => {
            let sample_time = match sample_time {
                Some(value) => parse_sample_time(value)?,
                None => 1.0,
            };
            if sample_time <= 0.0 {
                return Err(tf_error_with_detail(
                    &TF_ERROR_INVALID_SAMPLE_TIME,
                    "discrete transfer-function variables require a positive sample time",
                ));
            }
            TfModel::discrete_variable(variable, sample_time)
        }
        _ => unreachable!("validated variable"),
    }
}

async fn tf_binary(
    lhs: Value,
    rhs: Value,
    op: fn(&TfModel, &TfModel) -> BuiltinResult<TfModel>,
) -> BuiltinResult<Value> {
    let (lhs, rhs) = two_models_ordered(lhs, rhs, "tf").await?;
    op(&lhs, &rhs)?.to_value("tf")
}

fn parse_integer_exponent(value: &Value) -> BuiltinResult<i64> {
    let exponent = scalar_f64(value, "exponent", "tf")?;
    if !exponent.is_finite() || exponent.fract().abs() > 0.0 {
        return Err(control_error(
            "tf",
            "RunMat:tf:InvalidExponent",
            "tf: transfer-function powers require an integer scalar exponent",
        ));
    }
    if exponent < i64::MIN as f64 || exponent > i64::MAX as f64 {
        return Err(control_error(
            "tf",
            "RunMat:tf:InvalidExponent",
            "tf: exponent exceeds integer range",
        ));
    }
    Ok(exponent as i64)
}

#[runtime_builtin(
    name = "tf.plus",
    descriptor(crate::builtins::control::tf::TF_METHOD_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_plus(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    tf_binary(lhs, rhs, TfModel::add).await
}

#[runtime_builtin(
    name = "tf.minus",
    descriptor(crate::builtins::control::tf::TF_METHOD_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_minus(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    tf_binary(lhs, rhs, TfModel::sub).await
}

#[runtime_builtin(
    name = "tf.uplus",
    descriptor(crate::builtins::control::tf::TF_METHOD_UNARY_DESCRIPTOR),
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_uplus(sys: Value) -> BuiltinResult<Value> {
    TfModel::from_value_async(sys, "tf").await?.to_value("tf")
}

#[runtime_builtin(
    name = "tf.uminus",
    descriptor(crate::builtins::control::tf::TF_METHOD_UNARY_DESCRIPTOR),
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_uminus(sys: Value) -> BuiltinResult<Value> {
    TfModel::from_value_async(sys, "tf")
        .await?
        .neg()?
        .to_value("tf")
}

#[runtime_builtin(
    name = "tf.times",
    descriptor(crate::builtins::control::tf::TF_METHOD_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_times(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    tf_binary(lhs, rhs, TfModel::mul).await
}

#[runtime_builtin(
    name = "tf.mtimes",
    descriptor(crate::builtins::control::tf::TF_METHOD_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_mtimes(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    tf_binary(lhs, rhs, TfModel::mul).await
}

#[runtime_builtin(
    name = "tf.rdivide",
    descriptor(crate::builtins::control::tf::TF_METHOD_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_rdivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    tf_binary(lhs, rhs, TfModel::div).await
}

#[runtime_builtin(
    name = "tf.mrdivide",
    descriptor(crate::builtins::control::tf::TF_METHOD_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_mrdivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    tf_binary(lhs, rhs, TfModel::div).await
}

#[runtime_builtin(
    name = "tf.ldivide",
    descriptor(crate::builtins::control::tf::TF_METHOD_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_ldivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    tf_binary(rhs, lhs, TfModel::div).await
}

#[runtime_builtin(
    name = "tf.mldivide",
    descriptor(crate::builtins::control::tf::TF_METHOD_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_mldivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    tf_binary(rhs, lhs, TfModel::div).await
}

#[runtime_builtin(
    name = "tf.power",
    descriptor(crate::builtins::control::tf::TF_METHOD_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_power(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    let sys = TfModel::from_value_async(lhs, "tf").await?;
    let rhs = dispatcher::gather_if_needed_async(&rhs).await?;
    sys.powi(parse_integer_exponent(&rhs)?)?.to_value("tf")
}

#[runtime_builtin(
    name = "tf.mpower",
    descriptor(crate::builtins::control::tf::TF_METHOD_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_mpower(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    tf_power(lhs, rhs).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, IntValue, Tensor};

    fn run_tf(numerator: Value, denominator: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let mut args = vec![numerator, denominator];
        args.extend(rest);
        block_on(tf_builtin(args))
    }

    fn run_tf_args(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(tf_builtin(args))
    }

    fn property<'a>(value: &'a Value, name: &str) -> &'a Value {
        let Value::Object(object) = value else {
            panic!("expected object, got {value:?}");
        };
        object
            .properties
            .get(name)
            .unwrap_or_else(|| panic!("missing property {name}"))
    }

    fn tensor_property(value: &Value, name: &str) -> Vec<f64> {
        match property(value, name) {
            Value::Tensor(tensor) => tensor.data.clone(),
            other => panic!("expected tensor property {name}, got {other:?}"),
        }
    }

    #[test]
    fn tf_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = TF_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"s = tf('s')"));
        assert!(labels.contains(&"z = tf('z', Ts)"));
        assert!(labels.contains(&"sys = tf(numerator, denominator)"));
        assert!(labels.contains(&"sys = tf(numerator, denominator, Ts)"));
        assert!(labels.contains(&"sys = tf(numerator, denominator, \"Variable\", variableName)"));
        assert!(labels.contains(&"sys = tf(numerator, denominator, name, value, ...)"));
    }

    #[test]
    fn tf_constructs_continuous_siso_object() {
        let sys = run_tf(
            Value::Num(20.0),
            Value::Tensor(Tensor::new(vec![1.0, 5.0], vec![1, 2]).unwrap()),
            Vec::new(),
        )
        .expect("tf");

        let Value::Object(object) = &sys else {
            panic!("expected object");
        };
        assert_eq!(object.class_name, "tf");
        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("s"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.0));
        match property(&sys, "Numerator") {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 1]);
                assert_eq!(tensor.data, vec![20.0]);
            }
            other => panic!("expected numerator tensor, got {other:?}"),
        }
        match property(&sys, "Denominator") {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.data, vec![1.0, 5.0]);
            }
            other => panic!("expected denominator tensor, got {other:?}"),
        }
    }

    #[test]
    fn tf_accepts_continuous_variable_constructor() {
        let sys = run_tf_args(vec![Value::from("s")]).expect("tf('s')");
        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("s"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.0));
        assert_eq!(tensor_property(&sys, "Numerator"), vec![1.0, 0.0]);
        assert_eq!(tensor_property(&sys, "Denominator"), vec![1.0]);
    }

    #[test]
    fn tf_accepts_discrete_variable_constructor() {
        let sys = run_tf_args(vec![Value::from("z"), Value::Num(0.2)]).expect("tf('z', Ts)");
        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("z"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.2));
        assert_eq!(tensor_property(&sys, "Numerator"), vec![1.0, 0.0]);
        assert_eq!(tensor_property(&sys, "Denominator"), vec![1.0]);
    }

    #[test]
    fn tf_rejects_continuous_variable_constructor_with_positive_sample_time() {
        let err = run_tf_args(vec![Value::from("s"), Value::Num(0.2)])
            .expect_err("tf('s', positive Ts) should fail");
        assert_eq!(err.identifier(), TF_ERROR_INVALID_SAMPLE_TIME.identifier);
    }

    #[test]
    fn tf_arithmetic_builds_polynomial_transfer_functions() {
        let s = run_tf_args(vec![Value::from("s")]).expect("tf('s')");
        let s_squared = block_on(tf_power(s.clone(), Value::Num(2.0))).expect("s^2");
        let quadratic = block_on(tf_plus(
            block_on(tf_plus(
                block_on(tf_mtimes(Value::Num(0.4), s_squared)).expect("0.4*s^2"),
                block_on(tf_mtimes(Value::Num(1.8), s.clone())).expect("1.8*s"),
            ))
            .expect("sum terms"),
            Value::Num(1.0),
        ))
        .expect("add constant");
        let g = block_on(tf_mrdivide(Value::Num(2.5), quadratic)).expect("2.5/poly");

        assert_eq!(tensor_property(&g, "Numerator"), vec![2.5]);
        assert_eq!(tensor_property(&g, "Denominator"), vec![0.4, 1.8, 1.0]);
    }

    #[test]
    fn tf_normalizes_column_coefficients_to_rows() {
        let sys = run_tf(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0], vec![3, 1]).unwrap()),
            Vec::new(),
        )
        .expect("tf");

        match property(&sys, "Numerator") {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.data, vec![1.0, 2.0]);
            }
            other => panic!("expected numerator tensor, got {other:?}"),
        }
        match property(&sys, "Denominator") {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_eq!(tensor.data, vec![1.0, 3.0, 2.0]);
            }
            other => panic!("expected denominator tensor, got {other:?}"),
        }
    }

    #[test]
    fn tf_accepts_discrete_sample_time() {
        let sys = run_tf(
            Value::Int(IntValue::I32(1)),
            Value::Tensor(Tensor::new(vec![1.0, -0.5], vec![1, 2]).unwrap()),
            vec![Value::Num(0.1)],
        )
        .expect("tf");

        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("z"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.1));
    }

    #[test]
    fn tf_positional_zero_sample_time_remains_continuous() {
        let sys = run_tf(
            Value::Int(IntValue::I32(1)),
            Value::Tensor(Tensor::new(vec![1.0, 5.0], vec![1, 2]).unwrap()),
            vec![Value::Num(0.0)],
        )
        .expect("tf");

        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("s"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.0));
    }

    #[test]
    fn tf_accepts_variable_name_value_option() {
        let sys = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            vec![Value::from("Variable"), Value::from("p")],
        )
        .expect("tf");

        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("p"))
        );
    }

    #[test]
    fn tf_explicit_discrete_variable_defaults_positive_sample_time() {
        let sys = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            vec![Value::from("Variable"), Value::from("z")],
        )
        .expect("tf");

        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("z"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(1.0));
    }

    #[test]
    fn tf_rejects_explicit_continuous_variable_with_positive_sample_time() {
        let err = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            vec![
                Value::from("Variable"),
                Value::from("s"),
                Value::from("Ts"),
                Value::Num(0.5),
            ],
        )
        .expect_err("continuous variable with positive Ts should fail");

        assert_eq!(err.identifier(), TF_ERROR_INVALID_VARIABLE.identifier);
    }

    #[test]
    fn tf_rejects_explicit_discrete_variable_with_zero_sample_time() {
        let err = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            vec![
                Value::from("Variable"),
                Value::from("z"),
                Value::from("Ts"),
                Value::Num(0.0),
            ],
        )
        .expect_err("discrete variable with zero Ts should fail");

        assert_eq!(err.identifier(), TF_ERROR_INVALID_SAMPLE_TIME.identifier);
    }

    #[test]
    fn tf_accepts_explicit_discrete_variable_with_positive_sample_time() {
        let sys = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            vec![
                Value::from("Variable"),
                Value::from("z"),
                Value::from("Ts"),
                Value::Num(0.5),
            ],
        )
        .expect("tf");

        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("z"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.5));
    }

    #[test]
    fn tf_rejects_zero_denominator() {
        let err = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
            Vec::new(),
        )
        .expect_err("zero denominator should fail");
        assert!(err.message().contains("must not all be zero"));
        assert_eq!(err.identifier(), TF_ERROR_DENOMINATOR_INVALID.identifier);
    }

    #[test]
    fn tf_rejects_matrix_coefficients() {
        let err = run_tf(
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 5.0], vec![1, 2]).unwrap()),
            Vec::new(),
        )
        .expect_err("matrix numerator should fail");
        assert!(err
            .message()
            .contains("numerator coefficients must be a vector"));
        assert_eq!(err.identifier(), TF_ERROR_INVALID_COEFFICIENTS.identifier);
    }
}
