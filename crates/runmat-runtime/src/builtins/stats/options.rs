//! Statistics options structure helpers (`statset` / `statget`).

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ResolveContext, StringArray, StructValue, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const STATSET: &str = "statset";
const STATGET: &str = "statget";
const MAX_OPTION_INTEGER: usize = 1_000_000_000;

const OPTION_FIELDS: [&str; 20] = [
    "Display",
    "MaxFunEvals",
    "MaxIter",
    "TolBnd",
    "TolFun",
    "TolTypeFun",
    "TolX",
    "TolTypeX",
    "GradObj",
    "Jacobian",
    "DerivStep",
    "FunValCheck",
    "Robust",
    "RobustWgtFun",
    "WgtFun",
    "Tune",
    "UseParallel",
    "UseSubstreams",
    "Streams",
    "OutputFcn",
];

const COMMON_STATFUNS: [&str; 48] = [
    "bootci",
    "bootstrp",
    "crossval",
    "factoran",
    "fitglm",
    "fitlm",
    "fitlme",
    "fitnlm",
    "fitrgp",
    "gamfit",
    "gevfit",
    "glmfit",
    "gmdistribution",
    "gpfit",
    "kmeans",
    "kmedoids",
    "lasso",
    "lassoglm",
    "lognfit",
    "mlecustom",
    "mlecov",
    "mvncdf",
    "mvtcdf",
    "nbinfit",
    "nlinfit",
    "nnmf",
    "normfit",
    "parallel",
    "pca",
    "plsregress",
    "ppca",
    "rocmetrics",
    "sequentialfs",
    "tsne",
    "wblfit",
    "copulafit",
    "coxphfit",
    "evfit",
    "fitcox",
    "fitglme",
    "fitlmematrix",
    "mdscale",
    "nlmefitsa",
    "treebagger",
    "candexch",
    "cordexch",
    "daugment",
    "dcovary",
];

const OUTPUT_OPTIONS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Statistics options structure.",
}];

const INPUT_STATFUN: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "statfun",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Statistics function name.",
}];

const INPUT_PAIRS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value and additional name-value pairs.",
    },
];

const INPUT_STRUCT_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "oldopts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing statistics options structure.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Option field name or replacement options structure.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value and additional name-value pairs.",
    },
];

const INPUT_OLD_NEW_OPTIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "oldopts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing statistics options structure.",
    },
    BuiltinParamDescriptor {
        name: "newopts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Replacement statistics options structure. Nonempty fields override oldopts.",
    },
];

const STATSET_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "options = statset()",
        inputs: &[],
        outputs: &OUTPUT_OPTIONS,
    },
    BuiltinSignatureDescriptor {
        label: "options = statset(statfun)",
        inputs: &INPUT_STATFUN,
        outputs: &OUTPUT_OPTIONS,
    },
    BuiltinSignatureDescriptor {
        label: "options = statset(name, value, ...)",
        inputs: &INPUT_PAIRS,
        outputs: &OUTPUT_OPTIONS,
    },
    BuiltinSignatureDescriptor {
        label: "options = statset(oldopts, newopts)",
        inputs: &INPUT_OLD_NEW_OPTIONS,
        outputs: &OUTPUT_OPTIONS,
    },
    BuiltinSignatureDescriptor {
        label: "options = statset(oldopts, name, value, ...)",
        inputs: &INPUT_STRUCT_PAIRS,
        outputs: &OUTPUT_OPTIONS,
    },
];

const STATGET_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "val",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Option field value.",
}];

const STATGET_INPUT_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Statistics options structure.",
};
const STATGET_INPUT_FIELD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "field",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Option field name or unique leading prefix.",
};
const STATGET_INPUT_DEFAULT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "defaultData",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Value returned when the selected option is empty.",
};
const STATGET_INPUTS_REQUIRED: [BuiltinParamDescriptor; 2] =
    [STATGET_INPUT_OPTIONS, STATGET_INPUT_FIELD];
const STATGET_INPUTS: [BuiltinParamDescriptor; 3] = [
    STATGET_INPUT_OPTIONS,
    STATGET_INPUT_FIELD,
    STATGET_INPUT_DEFAULT,
];

const STATGET_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "val = statget(options, field)",
        inputs: &STATGET_INPUTS_REQUIRED,
        outputs: &STATGET_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "val = statget(options, field, defaultData)",
        inputs: &STATGET_INPUTS,
        outputs: &STATGET_OUTPUT,
    },
];

const STATSET_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STATSET.INVALID_ARGUMENT",
    identifier: Some("RunMat:statset:InvalidArgument"),
    when: "Argument grammar does not match supported statset forms.",
    message: "statset: invalid argument",
};
const STATSET_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STATSET.INVALID_OPTION",
    identifier: Some("RunMat:statset:InvalidOption"),
    when: "An option name or value is malformed.",
    message: "statset: invalid option",
};
const STATSET_ERROR_INVALID_STATFUN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STATSET.INVALID_STATFUN",
    identifier: Some("RunMat:statset:InvalidStatfun"),
    when: "The statfun argument is not a supported statistics function name.",
    message: "statset: invalid statistics function",
};

const STATSET_ERRORS: [BuiltinErrorDescriptor; 3] = [
    STATSET_ERROR_INVALID_ARGUMENT,
    STATSET_ERROR_INVALID_OPTION,
    STATSET_ERROR_INVALID_STATFUN,
];

const STATGET_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STATGET.INVALID_ARGUMENT",
    identifier: Some("RunMat:statget:InvalidArgument"),
    when: "Argument grammar does not match statget forms.",
    message: "statget: invalid argument",
};
const STATGET_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STATGET.INVALID_OPTION",
    identifier: Some("RunMat:statget:InvalidOption"),
    when: "The options argument is not a struct or the field argument is not text.",
    message: "statget: invalid option",
};

const STATGET_ERRORS: [BuiltinErrorDescriptor; 2] =
    [STATGET_ERROR_INVALID_ARGUMENT, STATGET_ERROR_INVALID_OPTION];

pub const STATSET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STATSET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STATSET_ERRORS,
};

pub const STATGET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STATGET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STATGET_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::stats::options")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "statset/statget",
    op_kind: GpuOpKind::Custom("statistics-options"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host metadata construction and lookup. gpuArray option values are gathered before use.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::stats::options")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "statset/statget",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Option struct construction and lookup are host metadata work and do not fuse.",
};

fn stat_options_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Struct {
        known_fields: Some(
            OPTION_FIELDS
                .iter()
                .map(|field| (*field).to_string())
                .collect(),
        ),
    }
}

fn statget_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

#[runtime_builtin(
    name = "statset",
    category = "stats/options",
    summary = "Create or update statistics options structures.",
    keywords = "statset,statistics options,MaxIter,TolFun,TolX,Display,UseParallel",
    accel = "cpu",
    type_resolver(stat_options_type),
    descriptor(crate::builtins::stats::options::STATSET_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::options"
)]
async fn statset_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_all(rest).await?;
    Ok(Value::Struct(parse_statset(args)?))
}

#[runtime_builtin(
    name = "statget",
    category = "stats/options",
    summary = "Access field values in statistics options structures.",
    keywords = "statget,statset,statistics options",
    accel = "cpu",
    type_resolver(statget_type),
    descriptor(crate::builtins::stats::options::STATGET_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::options"
)]
async fn statget_builtin(options: Value, field: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(statget_error(
            &STATGET_ERROR_INVALID_ARGUMENT,
            "statget: expected at most one default value",
        ));
    }
    let options = gather_if_needed_async(&options)
        .await
        .map_err(|err| statget_error(&STATGET_ERROR_INVALID_ARGUMENT, err.message()))?;
    let field = gather_if_needed_async(&field)
        .await
        .map_err(|err| statget_error(&STATGET_ERROR_INVALID_ARGUMENT, err.message()))?;
    let default_data = rest.into_iter().next();
    let Value::Struct(options) = options else {
        return Err(statget_error(
            &STATGET_ERROR_INVALID_OPTION,
            "statget: options must be a struct",
        ));
    };
    let field = text_scalar(&field).map_err(|err| {
        statget_error(
            &STATGET_ERROR_INVALID_OPTION,
            format!("statget: {}", err.message()),
        )
    })?;
    let Some(canonical) = unique_option_match(&field) else {
        return Ok(empty_numeric());
    };
    let value = lookup_struct_field(&options, canonical)
        .cloned()
        .unwrap_or_else(empty_numeric);
    if is_empty_value(&value) {
        if let Some(default_data) = default_data {
            gather_if_needed_async(&default_data)
                .await
                .map_err(|err| statget_error(&STATGET_ERROR_INVALID_ARGUMENT, err.message()))
        } else {
            Ok(value)
        }
    } else {
        Ok(value)
    }
}

async fn gather_all(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(
            gather_if_needed_async(&value)
                .await
                .map_err(|err| statset_error(&STATSET_ERROR_INVALID_ARGUMENT, err.message()))?,
        );
    }
    Ok(out)
}

fn parse_statset(args: Vec<Value>) -> BuiltinResult<StructValue> {
    if args.is_empty() {
        return Ok(empty_options());
    }
    let mut index = 0usize;
    let mut options;
    match &args[0] {
        Value::Struct(existing) => {
            options = canonicalize_options(existing)?;
            index = 1;
            if index < args.len() {
                if let Value::Struct(newopts) = &args[index] {
                    merge_old_into_new(&mut options, newopts)?;
                    index += 1;
                }
            }
        }
        first if looks_like_option_name(first) => {
            options = empty_options();
        }
        first => {
            let statfun = text_scalar(first).map_err(|err| {
                statset_error(
                    &STATSET_ERROR_INVALID_ARGUMENT,
                    format!("statset: {}", err.message()),
                )
            })?;
            options = defaults_for_statfun(&statfun)?;
            index = 1;
        }
    }
    let remaining = &args[index..];
    if !remaining.is_empty() {
        if !remaining.len().is_multiple_of(2) {
            return Err(statset_error(
                &STATSET_ERROR_INVALID_ARGUMENT,
                "statset: expected option name-value pairs",
            ));
        }
        for pair in remaining.chunks(2) {
            let name = text_scalar(&pair[0]).map_err(|err| {
                statset_error(
                    &STATSET_ERROR_INVALID_OPTION,
                    format!("statset: {}", err.message()),
                )
            })?;
            let canonical = canonical_option_name(&name)?;
            options.insert(canonical, validate_option_value(&name, &pair[1])?);
        }
    }
    Ok(options)
}

fn merge_old_into_new(oldopts: &mut StructValue, newopts: &StructValue) -> BuiltinResult<()> {
    let canonical_new = canonicalize_options(newopts)?;
    for field in OPTION_FIELDS {
        if let Some(new_value) = lookup_struct_field(&canonical_new, field) {
            if !is_empty_value(new_value) {
                oldopts.insert(field, new_value.clone());
            }
        }
    }
    Ok(())
}

fn canonicalize_options(options: &StructValue) -> BuiltinResult<StructValue> {
    let mut out = empty_options();
    for (name, value) in &options.fields {
        let canonical = canonical_option_name(name)?;
        out.insert(canonical, validate_option_value(name, value)?);
    }
    Ok(out)
}

fn defaults_for_statfun(statfun: &str) -> BuiltinResult<StructValue> {
    let key = statfun.to_ascii_lowercase();
    if !COMMON_STATFUNS.contains(&key.as_str()) {
        return Err(statset_error(
            &STATSET_ERROR_INVALID_STATFUN,
            format!("statset: unsupported statistics function '{statfun}'"),
        ));
    }
    let mut options = empty_options();
    match key.as_str() {
        "fitglm" | "fitlm" | "fitnlm" | "glmfit" | "lasso" | "lassoglm" | "nlinfit" | "normfit"
        | "wblfit" => {
            options.insert("Display", Value::from("off"));
            options.insert("MaxIter", Value::Num(100.0));
            options.insert("TolX", Value::Num(1.0e-6));
        }
        "factoran" => {
            options.insert("Display", Value::from("off"));
            options.insert("MaxIter", Value::Num(100.0));
            options.insert("TolX", Value::Num(1.0e-8));
        }
        "nbinfit" => {
            options.insert("Display", Value::from("off"));
            options.insert("MaxFunEvals", Value::Num(400.0));
            options.insert("MaxIter", Value::Num(200.0));
            options.insert("TolBnd", Value::Num(1.0e-6));
            options.insert("TolFun", Value::Num(1.0e-6));
            options.insert("TolX", Value::Num(1.0e-6));
        }
        "kmeans" | "tsne" | "pca" | "ppca" | "gmdistribution" | "kmedoids" => {
            options.insert("Display", Value::from("off"));
            options.insert("MaxIter", Value::Num(100.0));
        }
        "bootci" | "bootstrp" | "crossval" | "parallel" | "sequentialfs" => {
            options.insert("UseParallel", Value::Bool(false));
            options.insert("UseSubstreams", Value::Bool(false));
            options.insert("Streams", empty_cell());
        }
        _ => {}
    }
    Ok(options)
}

fn empty_options() -> StructValue {
    let mut out = StructValue::new();
    for field in OPTION_FIELDS {
        out.insert(
            field,
            if field == "Streams" {
                empty_cell()
            } else {
                empty_numeric()
            },
        );
    }
    out
}

fn canonical_option_name(name: &str) -> BuiltinResult<&'static str> {
    let Some(canonical) = unique_option_match(name) else {
        return Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("statset: unknown option '{name}'"),
        ));
    };
    Ok(canonical)
}

fn unique_option_match(name: &str) -> Option<&'static str> {
    let needle = name.to_ascii_lowercase();
    for field in OPTION_FIELDS {
        if field.eq_ignore_ascii_case(name) {
            return Some(field);
        }
    }
    let mut found = None;
    for field in OPTION_FIELDS {
        if field.to_ascii_lowercase().starts_with(&needle) {
            if found.is_some() {
                return None;
            }
            found = Some(field);
        }
    }
    found
}

fn validate_option_value(name: &str, value: &Value) -> BuiltinResult<Value> {
    let canonical = canonical_option_name(name)?;
    match canonical {
        "Display" => one_of_text(canonical, value, &["off", "final", "iter"]),
        "FunValCheck" | "GradObj" | "Jacobian" => one_of_text(canonical, value, &["off", "on"]),
        "TolTypeFun" | "TolTypeX" => one_of_text(canonical, value, &["abs", "rel"]),
        "RobustWgtFun" => validate_robust_weight(value),
        "MaxFunEvals" | "MaxIter" => positive_integer_value(canonical, value),
        "TolBnd" | "TolFun" | "TolX" | "Tune" => positive_scalar_value(canonical, value),
        "DerivStep" => positive_numeric_value(canonical, value),
        "UseParallel" | "UseSubstreams" => bool_or_on_off_value(canonical, value),
        "Streams" | "OutputFcn" | "Robust" | "WgtFun" => Ok(value.clone()),
        other => Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("statset: unsupported option '{other}'"),
        )),
    }
}

fn validate_robust_weight(value: &Value) -> BuiltinResult<Value> {
    if is_empty_value(value)
        || matches!(
            value,
            Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle { .. }
        )
    {
        return Ok(value.clone());
    }
    one_of_text(
        "RobustWgtFun",
        value,
        &[
            "andrews", "bisquare", "cauchy", "fair", "huber", "logistic", "talwar", "welsch",
        ],
    )
}

fn one_of_text(field: &str, value: &Value, allowed: &[&str]) -> BuiltinResult<Value> {
    if is_empty_value(value) {
        return Ok(value.clone());
    }
    let text = text_scalar(value)?;
    let lower = text.to_ascii_lowercase();
    if allowed.contains(&lower.as_str()) {
        Ok(Value::from(lower))
    } else {
        Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("statset: {field} must be one of {}", allowed.join(", ")),
        ))
    }
}

fn positive_integer_value(field: &str, value: &Value) -> BuiltinResult<Value> {
    if is_empty_value(value) {
        return Ok(value.clone());
    }
    let scalar = numeric_scalar(field, value)?;
    if scalar < 1.0 || scalar.fract() != 0.0 || scalar > MAX_OPTION_INTEGER as f64 {
        return Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("statset: {field} must be a positive integer"),
        ));
    }
    Ok(Value::Num(scalar))
}

fn positive_scalar_value(field: &str, value: &Value) -> BuiltinResult<Value> {
    if is_empty_value(value) {
        return Ok(value.clone());
    }
    let scalar = numeric_scalar(field, value)?;
    if scalar <= 0.0 {
        return Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("statset: {field} must be a positive scalar"),
        ));
    }
    Ok(Value::Num(scalar))
}

fn positive_numeric_value(field: &str, value: &Value) -> BuiltinResult<Value> {
    if is_empty_value(value) {
        return Ok(value.clone());
    }
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => positive_scalar_value(field, value),
        Value::Tensor(tensor) => {
            if tensor
                .data
                .iter()
                .all(|entry| entry.is_finite() && *entry > 0.0)
            {
                Ok(value.clone())
            } else {
                Err(statset_error(
                    &STATSET_ERROR_INVALID_OPTION,
                    format!("statset: {field} must contain positive finite values"),
                ))
            }
        }
        other => Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("statset: {field} must be numeric, got {other:?}"),
        )),
    }
}

fn bool_or_on_off_value(field: &str, value: &Value) -> BuiltinResult<Value> {
    if is_empty_value(value) {
        return Ok(value.clone());
    }
    match value {
        Value::Bool(flag) => Ok(Value::Bool(*flag)),
        Value::Num(n) if *n == 0.0 || *n == 1.0 => Ok(Value::Bool(*n != 0.0)),
        Value::Int(i) if i.to_f64() == 0.0 || i.to_f64() == 1.0 => {
            Ok(Value::Bool(i.to_f64() != 0.0))
        }
        Value::Tensor(tensor)
            if tensor.data.len() == 1 && (tensor.data[0] == 0.0 || tensor.data[0] == 1.0) =>
        {
            Ok(Value::Bool(tensor.data[0] != 0.0))
        }
        _ => {
            let text = text_scalar(value)?;
            match text.to_ascii_lowercase().as_str() {
                "on" | "true" => Ok(Value::Bool(true)),
                "off" | "false" => Ok(Value::Bool(false)),
                _ => Err(statset_error(
                    &STATSET_ERROR_INVALID_OPTION,
                    format!("statset: {field} must be logical or 'on'/'off'"),
                )),
            }
        }
    }
}

fn numeric_scalar(field: &str, value: &Value) -> BuiltinResult<f64> {
    let scalar = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(flag) => {
            if *flag {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        other => {
            return Err(statset_error(
                &STATSET_ERROR_INVALID_OPTION,
                format!("statset: {field} must be a numeric scalar, got {other:?}"),
            ))
        }
    };
    if !scalar.is_finite() {
        return Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("statset: {field} must be finite"),
        ));
    }
    Ok(scalar)
}

fn text_scalar(value: &Value) -> BuiltinResult<String> {
    if let Some(text) = keyword_of(value) {
        return Ok(text);
    }
    match value {
        Value::CharArray(CharArray { data, rows: 1, .. }) => Ok(data.iter().collect()),
        Value::StringArray(StringArray { data, .. }) if data.len() == 1 => Ok(data[0].clone()),
        other => Err(statset_error(
            &STATSET_ERROR_INVALID_OPTION,
            format!("option names must be text scalars, got {other:?}"),
        )),
    }
}

fn looks_like_option_name(value: &Value) -> bool {
    text_scalar(value)
        .ok()
        .and_then(|text| unique_option_match(&text))
        .is_some()
}

fn lookup_struct_field<'a>(options: &'a StructValue, name: &str) -> Option<&'a Value> {
    options
        .fields
        .iter()
        .find(|(field, _)| field.eq_ignore_ascii_case(name))
        .map(|(_, value)| value)
}

fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.data.is_empty(),
        Value::LogicalArray(array) => array.data.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        Value::StringArray(array) => array.data.is_empty(),
        Value::CharArray(array) => array.data.is_empty(),
        _ => false,
    }
}

fn empty_numeric() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor"))
}

fn empty_cell() -> Value {
    Value::Cell(CellArray::new(Vec::new(), 0, 0).expect("empty cell"))
}

fn statset_error(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("statset:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(STATSET);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn statget_error(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("statget:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(STATGET);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn struct_value(value: Value) -> StructValue {
        let Value::Struct(st) = value else {
            panic!("expected struct, got {value:?}");
        };
        st
    }

    fn num_field(options: &StructValue, name: &str) -> f64 {
        match options.fields.get(name).unwrap() {
            Value::Num(value) => *value,
            other => panic!("expected numeric field {name}, got {other:?}"),
        }
    }

    #[test]
    fn statset_builds_custom_options() {
        let options = struct_value(
            block_on(statset_builtin(vec![
                Value::from("FunValCheck"),
                Value::from("on"),
                Value::from("TolX"),
                Value::Num(1.0e-8),
                Value::from("UseParallel"),
                Value::from("off"),
            ]))
            .unwrap(),
        );
        assert!(matches!(options.fields.get("FunValCheck"), Some(Value::String(s)) if s == "on"));
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert!(matches!(
            options.fields.get("UseParallel"),
            Some(Value::Bool(false))
        ));
        assert!(
            matches!(options.fields.get("Streams"), Some(Value::Cell(cell)) if cell.data.is_empty())
        );
    }

    #[test]
    fn statset_applies_function_defaults_and_updates() {
        let base = block_on(statset_builtin(vec![Value::from("nbinfit")])).unwrap();
        let options = struct_value(
            block_on(statset_builtin(vec![
                base,
                Value::from("TolX"),
                Value::Num(1.0e-8),
            ]))
            .unwrap(),
        );
        assert_eq!(num_field(&options, "MaxIter"), 200.0);
        assert_eq!(num_field(&options, "TolX"), 1.0e-8);
        assert_eq!(num_field(&options, "TolFun"), 1.0e-6);
    }

    #[test]
    fn statset_old_new_merge_prefers_nonempty_new_fields() {
        let oldopts = struct_value(
            block_on(statset_builtin(vec![
                Value::from("TolX"),
                Value::Num(1.0e-6),
                Value::from("MaxIter"),
                Value::Num(20.0),
            ]))
            .unwrap(),
        );
        let newopts = struct_value(
            block_on(statset_builtin(vec![
                Value::from("TolX"),
                Value::Num(1.0e-9),
            ]))
            .unwrap(),
        );
        let merged = struct_value(
            block_on(statset_builtin(vec![
                Value::Struct(oldopts),
                Value::Struct(newopts),
            ]))
            .unwrap(),
        );
        assert_eq!(num_field(&merged, "TolX"), 1.0e-9);
        assert_eq!(num_field(&merged, "MaxIter"), 20.0);
    }

    #[test]
    fn statget_supports_unique_prefix_and_default_for_empty() {
        let options = block_on(statset_builtin(vec![
            Value::from("TolX"),
            Value::Num(1.0e-8),
            Value::from("MaxIter"),
            Value::Num(15.0),
        ]))
        .unwrap();
        let value = block_on(statget_builtin(
            options.clone(),
            Value::from("TolX"),
            Vec::new(),
        ))
        .unwrap();
        assert_eq!(value, Value::Num(1.0e-8));

        let value = block_on(statget_builtin(
            options.clone(),
            Value::from("MaxI"),
            Vec::new(),
        ))
        .unwrap();
        assert_eq!(value, Value::Num(15.0));

        let value = block_on(statget_builtin(
            options,
            Value::from("TolFun"),
            vec![Value::Num(3.0)],
        ))
        .unwrap();
        assert_eq!(value, Value::Num(3.0));
    }

    #[test]
    fn statset_rejects_invalid_values() {
        let err = block_on(statset_builtin(vec![
            Value::from("MaxIter"),
            Value::Num(2.5),
        ]))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:statset:InvalidOption"));

        let err = block_on(statset_builtin(vec![
            Value::from("Display"),
            Value::from("verbose"),
        ]))
        .unwrap_err();
        assert!(err.message.contains("Display"));
    }

    #[test]
    fn descriptors_cover_public_forms() {
        let statset_labels = STATSET_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(statset_labels.contains(&"options = statset(statfun)"));
        assert!(statset_labels.contains(&"options = statset(oldopts, newopts)"));

        let statget_labels = STATGET_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert_eq!(
            statget_labels,
            vec![
                "val = statget(options, field)",
                "val = statget(options, field, defaultData)",
            ]
        );
    }
}
