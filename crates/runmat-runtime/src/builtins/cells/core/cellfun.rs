//! MATLAB-compatible `cellfun` builtin with host execution semantics for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, Closure, ComplexTensor, LogicalArray, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::cells::type_resolvers::cellfun_type;
use crate::builtins::common::shape::{dims_to_row_tensor, value_numel};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{
    build_runtime_error, call_builtin_async, gather_if_needed_async, make_cell_with_shape,
    user_functions, BuiltinResult, RuntimeError,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::cells::core::cellfun")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cellfun",
    op_kind: GpuOpKind::Custom("host-cell-map"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Executes on the host and gathers GPU-resident inputs before evaluating callbacks.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::cells::core::cellfun")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cellfun",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Callback execution happens on the host; fusion planners should treat cellfun as a fusion barrier.",
};

const BUILTIN_NAME: &str = "cellfun";

const CELLFUN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mapped callback outputs.",
}];

const CELLFUN_SIG_SINGLE_CELL_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or builtin/function name.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input cell array.",
    },
];

const CELLFUN_SIG_MULTI_CELL_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or builtin/function name.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "One or more input cell arrays followed by optional extra callback args.",
    },
];

const CELLFUN_SIG_UNIFORM_OUTPUT_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or builtin/function name.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Cell input arrays and optional callback args.",
    },
    BuiltinParamDescriptor {
        name: "UniformOutput",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"UniformOutput\""),
        description: "Name-value key for uniform output mode.",
    },
    BuiltinParamDescriptor {
        name: "tf",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Required,
        default: Some("true"),
        description: "Whether callback outputs must be scalar-uniform.",
    },
];

const CELLFUN_SIG_ERROR_HANDLER_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or builtin/function name.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Cell input arrays and optional callback args.",
    },
    BuiltinParamDescriptor {
        name: "ErrorHandler",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"ErrorHandler\""),
        description: "Name-value key for callback error handler.",
    },
    BuiltinParamDescriptor {
        name: "errfunc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Error handler function handle or name.",
    },
];

const CELLFUN_SIG_BOTH_OPTIONS_INPUTS: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or builtin/function name.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Cell input arrays and optional callback args.",
    },
    BuiltinParamDescriptor {
        name: "UniformOutput",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"UniformOutput\""),
        description: "Name-value key for uniform output mode.",
    },
    BuiltinParamDescriptor {
        name: "tf",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Required,
        default: Some("true"),
        description: "Whether callback outputs must be scalar-uniform.",
    },
    BuiltinParamDescriptor {
        name: "ErrorHandler",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"ErrorHandler\""),
        description: "Name-value key for callback error handler.",
    },
    BuiltinParamDescriptor {
        name: "errfunc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Error handler function handle or name.",
    },
];

const CELLFUN_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "Y = cellfun(func, C)",
        inputs: &CELLFUN_SIG_SINGLE_CELL_INPUTS,
        outputs: &CELLFUN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = cellfun(func, C1, C2, ...)",
        inputs: &CELLFUN_SIG_MULTI_CELL_INPUTS,
        outputs: &CELLFUN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = cellfun(func, C..., \"UniformOutput\", tf)",
        inputs: &CELLFUN_SIG_UNIFORM_OUTPUT_INPUTS,
        outputs: &CELLFUN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = cellfun(func, C..., \"ErrorHandler\", errfunc)",
        inputs: &CELLFUN_SIG_ERROR_HANDLER_INPUTS,
        outputs: &CELLFUN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = cellfun(func, C..., \"UniformOutput\", tf, \"ErrorHandler\", errfunc)",
        inputs: &CELLFUN_SIG_BOTH_OPTIONS_INPUTS,
        outputs: &CELLFUN_OUTPUT,
    },
];

const CELLFUN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLFUN.INVALID_INPUT",
    identifier: Some("RunMat:cellfun:InvalidInput"),
    when: "Input callback/cell-array arguments or name-value forms are invalid.",
    message: "cellfun: invalid input arguments",
};

const CELLFUN_ERROR_UNIFORM_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLFUN.UNIFORM_OUTPUT",
    identifier: Some("RunMat:cellfun:UniformOutput"),
    when: "UniformOutput mode requirements are violated.",
    message: "cellfun: uniform output contract violated",
};

const CELLFUN_ERROR_FUNCTION_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLFUN.FUNCTION_ERROR",
    identifier: Some("RunMat:cellfun:FunctionError"),
    when: "Callback-related function execution semantics fail.",
    message: "cellfun: callback execution error",
};

const CELLFUN_ERROR_UNDEFINED_FUNCTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLFUN.UNDEFINED_FUNCTION",
    identifier: Some("RunMat:UndefinedFunction"),
    when: "External callable resolution fails at runtime boundary.",
    message: "cellfun: undefined external function",
};

const CELLFUN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLFUN.INTERNAL",
    identifier: None,
    when: "Internal allocation or struct materialization fails.",
    message: "cellfun: internal error",
};

const CELLFUN_ERRORS: [BuiltinErrorDescriptor; 5] = [
    CELLFUN_ERROR_INVALID_INPUT,
    CELLFUN_ERROR_UNIFORM_OUTPUT,
    CELLFUN_ERROR_FUNCTION_ERROR,
    CELLFUN_ERROR_UNDEFINED_FUNCTION,
    CELLFUN_ERROR_INTERNAL,
];

pub const CELLFUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CELLFUN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CELLFUN_ERRORS,
};

fn cellfun_error_with_message(
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
    name = "cellfun",
    category = "cells/core",
    summary = "Apply a function to each cell element.",
    keywords = "cellfun,cell,array,functional",
    accel = "host",
    type_resolver(cellfun_type),
    descriptor(crate::builtins::cells::core::cellfun::CELLFUN_DESCRIPTOR),
    builtin_path = "crate::builtins::cells::core::cellfun"
)]
async fn cellfun_builtin(func: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let callable = Callable::from_function(func)?;
    let mut args = rest;

    let mut uniform_output = true;
    let mut error_handler: Option<Callable> = None;

    while args.len() >= 2 {
        let name_candidate = args[args.len() - 2].clone();
        let Some(name) = extract_string(&name_candidate) else {
            break;
        };
        let value = args.pop().expect("value present");
        args.pop();
        match name.to_ascii_lowercase().as_str() {
            "uniformoutput" => {
                uniform_output = parse_uniform_output(value)?;
            }
            "errorhandler" => {
                error_handler = Some(Callable::from_function(value)?);
            }
            unknown => {
                return Err(cellfun_error_with_message(
                    format!("cellfun: unknown name-value argument '{unknown}'"),
                    &CELLFUN_ERROR_INVALID_INPUT,
                ));
            }
        }
    }

    if args.is_empty() {
        return Err(cellfun_error_with_message(
            "cellfun: expected at least one cell array input",
            &CELLFUN_ERROR_INVALID_INPUT,
        ));
    }

    let mut cell_inputs: Vec<CellArray> = Vec::new();
    let mut extra_args: Vec<Value> = Vec::new();
    let mut seen_non_cell = false;

    for value in args.into_iter() {
        match value {
            Value::Cell(ca) if !seen_non_cell => cell_inputs.push(ca),
            Value::Cell(_) => {
                return Err(cellfun_error_with_message(
                    "cellfun: cell array inputs must precede extra arguments",
                    &CELLFUN_ERROR_INVALID_INPUT,
                ));
            }
            other => {
                seen_non_cell = true;
                extra_args.push(other);
            }
        }
    }

    if cell_inputs.is_empty() {
        return Err(cellfun_error_with_message(
            "cellfun: expected at least one cell array input",
            &CELLFUN_ERROR_INVALID_INPUT,
        ));
    }

    let reference_shape = cell_inputs[0].shape.clone();
    for (idx, ca) in cell_inputs.iter().enumerate().skip(1) {
        if ca.shape != reference_shape {
            return Err(cellfun_error_with_message(
                format!(
                    "cellfun: cell array input {} does not match the size of the first input",
                    idx + 1
                ),
                &CELLFUN_ERROR_INVALID_INPUT,
            ));
        }
    }

    if uniform_output {
        execute_uniform(
            &callable,
            &cell_inputs,
            &extra_args,
            error_handler,
            &reference_shape,
        )
        .await
    } else {
        execute_cell(
            &callable,
            &cell_inputs,
            &extra_args,
            error_handler,
            &reference_shape,
        )
        .await
    }
}

async fn execute_uniform(
    callable: &Callable,
    cell_inputs: &[CellArray],
    extra_args: &[Value],
    error_handler: Option<Callable>,
    shape: &[usize],
) -> BuiltinResult<Value> {
    let element_count = total_len(shape).ok_or_else(|| {
        cellfun_error_with_message(
            "cellfun: cell array size exceeds platform limits",
            &CELLFUN_ERROR_INVALID_INPUT,
        )
    })?;

    let host_extra_args = prepare_extra_args(extra_args).await?;
    let mut collector = UniformCollector::Pending;
    let mut cell_values: Vec<Value> = Vec::with_capacity(cell_inputs.len());
    let mut call_args: Vec<Value> = Vec::with_capacity(cell_inputs.len() + host_extra_args.len());

    for linear_idx in 0..element_count {
        cell_values.clear();
        for cell in cell_inputs {
            let raw = deref_cell_value(cell, linear_idx);
            let host_value = gather_if_needed_async(&raw).await?;
            cell_values.push(host_value);
        }
        call_args.clear();
        call_args.extend(cell_values.iter().cloned());
        call_args.extend(host_extra_args.iter().cloned());

        let result = match callable.call(&call_args).await {
            Ok(value) => value,
            Err(err) => {
                let Some(handler) = error_handler.as_ref() else {
                    return Err(err);
                };
                let err_value = make_error_struct(&err, linear_idx, shape)?;
                let mut handler_args =
                    Vec::with_capacity(1 + cell_values.len() + host_extra_args.len());
                handler_args.push(err_value);
                handler_args.extend(cell_values.clone());
                handler_args.extend(host_extra_args.iter().cloned());
                handler.call(&handler_args).await?
            }
        };

        let host_value = gather_if_needed_async(&result).await?;
        collector.push(&host_value)?;
    }

    collector.finish(shape)
}

async fn execute_cell(
    callable: &Callable,
    cell_inputs: &[CellArray],
    extra_args: &[Value],
    error_handler: Option<Callable>,
    shape: &[usize],
) -> BuiltinResult<Value> {
    let element_count = total_len(shape).ok_or_else(|| {
        cellfun_error_with_message(
            "cellfun: cell array size exceeds platform limits",
            &CELLFUN_ERROR_INVALID_INPUT,
        )
    })?;
    let host_extra_args = prepare_extra_args(extra_args).await?;
    let mut outputs: Vec<Value> = Vec::with_capacity(element_count);
    let mut cell_values: Vec<Value> = Vec::with_capacity(cell_inputs.len());
    let mut call_args: Vec<Value> = Vec::with_capacity(cell_inputs.len() + host_extra_args.len());

    for linear_idx in 0..element_count {
        cell_values.clear();
        for cell in cell_inputs {
            let raw = deref_cell_value(cell, linear_idx);
            let host_value = gather_if_needed_async(&raw).await?;
            cell_values.push(host_value);
        }
        call_args.clear();
        call_args.extend(cell_values.iter().cloned());
        call_args.extend(host_extra_args.iter().cloned());

        let result = match callable.call(&call_args).await {
            Ok(value) => value,
            Err(err) => {
                let Some(handler) = error_handler.as_ref() else {
                    return Err(err);
                };
                let err_value = make_error_struct(&err, linear_idx, shape)?;
                let mut handler_args =
                    Vec::with_capacity(1 + cell_values.len() + host_extra_args.len());
                handler_args.push(err_value);
                handler_args.extend(cell_values.clone());
                handler_args.extend(host_extra_args.iter().cloned());
                handler.call(&handler_args).await?
            }
        };

        let host_value = gather_if_needed_async(&result).await?;
        outputs.push(host_value);
    }

    make_cell_with_shape(outputs, shape.to_vec())
        .map_err(|e| cellfun_error_with_message(format!("cellfun: {e}"), &CELLFUN_ERROR_INTERNAL))
}

fn deref_cell_value(cell: &CellArray, index: usize) -> Value {
    cell.data
        .get(index)
        .cloned()
        .unwrap_or(Value::Num(f64::NAN))
}

fn total_len(shape: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        Some(0)
    } else {
        shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    }
}

fn extract_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        _ => None,
    }
}

async fn prepare_extra_args(extra_args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut host_args = Vec::with_capacity(extra_args.len());
    for arg in extra_args {
        host_args.push(gather_if_needed_async(arg).await?);
    }
    Ok(host_args)
}

fn parse_uniform_output(value: Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(b),
        Value::Num(n) => Ok(n != 0.0),
        Value::Int(iv) => Ok(iv.to_f64() != 0.0),
        Value::String(s) => parse_bool_string(&s).ok_or_else(|| {
            cellfun_error_with_message(
                "cellfun: UniformOutput must be logical true or false",
                &CELLFUN_ERROR_UNIFORM_OUTPUT,
            )
        }),
        Value::CharArray(ca) if ca.rows == 1 => {
            let s: String = ca.data.iter().collect();
            parse_bool_string(&s).ok_or_else(|| {
                cellfun_error_with_message(
                    "cellfun: UniformOutput must be logical true or false",
                    &CELLFUN_ERROR_UNIFORM_OUTPUT,
                )
            })
        }
        other => Err(cellfun_error_with_message(
            format!("cellfun: UniformOutput must be logical true or false, got {other:?}"),
            &CELLFUN_ERROR_UNIFORM_OUTPUT,
        )),
    }
}

fn parse_bool_string(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "on" => Some(true),
        "false" | "off" => Some(false),
        _ => None,
    }
}

fn make_error_struct(
    raw_error: &RuntimeError,
    linear_index: usize,
    shape: &[usize],
) -> BuiltinResult<Value> {
    let (identifier, message) = error_identifier_and_message(raw_error);
    let mut st = StructValue::new();
    st.fields
        .insert("identifier".to_string(), Value::String(identifier));
    st.fields
        .insert("message".to_string(), Value::String(message));
    st.fields
        .insert("index".to_string(), Value::Num((linear_index + 1) as f64));
    let subs = linear_to_indices(linear_index, shape);
    let subs_tensor = dims_to_row_tensor(&subs).map_err(|e| {
        cellfun_error_with_message(format!("cellfun: {e}"), &CELLFUN_ERROR_INTERNAL)
    })?;
    st.fields
        .insert("indices".to_string(), Value::Tensor(subs_tensor));
    Ok(Value::Struct(st))
}

fn error_identifier_and_message(error: &RuntimeError) -> (String, String) {
    if let Some(identifier) = error.identifier() {
        return (identifier.to_string(), error.message().to_string());
    }
    split_error_message(error.message())
}

fn split_error_message(raw: &str) -> (String, String) {
    let trimmed = raw.trim();
    let mut indices = trimmed.match_indices(':');
    if let Some((_, _)) = indices.next() {
        if let Some((second_idx, _)) = indices.next() {
            let identifier = trimmed[..second_idx].trim().to_string();
            let message = trimmed[second_idx + 1..].trim().to_string();
            if !identifier.is_empty() && identifier.contains(':') {
                return (
                    identifier,
                    if message.is_empty() {
                        trimmed.to_string()
                    } else {
                        message
                    },
                );
            }
        } else if trimmed.len() >= 7
            && (trimmed[..7].eq_ignore_ascii_case("matlab:")
                || trimmed[..7].eq_ignore_ascii_case("runmat:"))
        {
            return (trimmed.to_string(), String::new());
        }
    }
    (
        CELLFUN_ERROR_FUNCTION_ERROR
            .identifier
            .expect("CELLFUN_ERROR_FUNCTION_ERROR must define identifier")
            .to_string(),
        trimmed.to_string(),
    )
}

fn linear_to_indices(mut index: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![1];
    }
    let mut subs = Vec::with_capacity(shape.len());
    for &dim in shape {
        if dim == 0 {
            subs.push(1);
            continue;
        }
        let coord = (index % dim) + 1;
        subs.push(coord);
        index /= dim;
    }
    subs
}

#[derive(Clone)]
enum Callable {
    Builtin { name: String },
    ExternalName { name: String },
    Closure(Closure),
    Special(SpecialCallable),
}

impl Callable {
    fn resolved_semantic_handle(name: &str) -> Option<Self> {
        let function = user_functions::resolve_semantic_function_by_name(name)?;
        Some(Callable::Closure(Closure {
            function_name: name.to_string(),
            bound_function: Some(function),
            captures: Vec::new(),
        }))
    }

    fn from_function(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::String(s) => Self::from_text(&s, true),
            Value::CharArray(ca) => {
                if ca.rows != 1 {
                    Err(cellfun_error_with_message(
                        "cellfun: function name must be a character vector or string scalar",
                        &CELLFUN_ERROR_INVALID_INPUT,
                    ))
                } else {
                    let text: String = ca.data.iter().collect();
                    Self::from_text(&text, true)
                }
            }
            Value::StringArray(sa) => {
                if sa.data.len() == 1 {
                    Self::from_text(&sa.data[0], true)
                } else {
                    Err(cellfun_error_with_message(
                        "cellfun: function name must be a character vector or string scalar",
                        &CELLFUN_ERROR_INVALID_INPUT,
                    ))
                }
            }
            Value::FunctionHandle(name) => Self::from_text(&name, true),
            Value::ExternalFunctionHandle(name) => {
                if let Some(callable) = Self::resolved_semantic_handle(&name) {
                    Ok(callable)
                } else if crate::is_well_formed_qualified_name(&name) {
                    Ok(Callable::ExternalName { name })
                } else {
                    Ok(Callable::Builtin { name })
                }
            }
            Value::BoundFunctionHandle { name, function } => Ok(Callable::Closure(Closure {
                function_name: name,
                bound_function: Some(function),
                captures: Vec::new(),
            })),
            Value::Closure(mut c) => {
                if c.bound_function.is_none() {
                    if let Some(function) =
                        user_functions::resolve_semantic_function_by_name(&c.function_name)
                    {
                        c.bound_function = Some(function);
                    }
                }
                Ok(Callable::Closure(c))
            }
            other => Err(cellfun_error_with_message(
                format!("cellfun: expected function handle or builtin name, got {other:?}"),
                &CELLFUN_ERROR_INVALID_INPUT,
            )),
        }
    }

    fn from_text(text: &str, fold_case: bool) -> BuiltinResult<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(cellfun_error_with_message(
                "cellfun: expected function handle or builtin name, got empty string",
                &CELLFUN_ERROR_INVALID_INPUT,
            ));
        }
        if let Some(rest) = trimmed.strip_prefix('@') {
            let name = rest.trim();
            if name.is_empty() {
                Err(cellfun_error_with_message(
                    "cellfun: empty function handle",
                    &CELLFUN_ERROR_INVALID_INPUT,
                ))
            } else {
                if let Some(callable) = Self::resolved_semantic_handle(name) {
                    return Ok(callable);
                }
                if crate::is_well_formed_qualified_name(name) {
                    return Ok(Callable::ExternalName {
                        name: name.to_string(),
                    });
                }
                Ok(Callable::Builtin {
                    name: name.to_string(),
                })
            }
        } else {
            let lowered = trimmed.to_ascii_lowercase();
            if fold_case && lowered == "isclass" {
                Ok(Callable::Special(SpecialCallable::IsClass))
            } else if fold_case && lowered == "prodofsize" {
                Ok(Callable::Special(SpecialCallable::ProdOfSize))
            } else {
                let name = if fold_case {
                    lowered
                } else {
                    trimmed.to_string()
                };
                if let Some(callable) = Self::resolved_semantic_handle(&name) {
                    return Ok(callable);
                }
                if crate::is_well_formed_qualified_name(&name) {
                    return Ok(Callable::ExternalName { name });
                }
                Ok(Callable::Builtin { name })
            }
        }
    }

    async fn call(&self, args: &[Value]) -> BuiltinResult<Value> {
        match self {
            Callable::Builtin { name } => {
                let request = user_functions::CallableRequest::resolved(
                    runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(name.clone())),
                    runmat_hir::CallableFallbackPolicy::RuntimeNameResolution,
                    args.to_vec(),
                    1,
                );
                if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
                    return result;
                }
                call_builtin_async(name, args).await
            }
            Callable::ExternalName { name } => {
                let identity = crate::external_callable_identity_for_name(name);
                let request = user_functions::CallableRequest::resolved(
                    identity.clone(),
                    runmat_hir::CallableFallbackPolicy::ExternalBoundary,
                    args.to_vec(),
                    1,
                );
                if let Some(result) = user_functions::try_call_semantic_descriptor(request).await {
                    return result;
                }
                Err(cellfun_error_with_message(
                    format!("Undefined function for callable identity {identity:?}"),
                    &CELLFUN_ERROR_UNDEFINED_FUNCTION,
                ))
            }
            Callable::Closure(c) => {
                let mut captures = c.captures.clone();
                captures.extend_from_slice(args);
                if let Some(function) = c.bound_function {
                    let request =
                        user_functions::CallableRequest::semantic(function, captures.clone(), 1);
                    if let Some(result) =
                        user_functions::try_call_semantic_descriptor(request).await
                    {
                        return result;
                    }
                    return Err(cellfun_error_with_message(
                        format!(
                            "cellfun: semantic closure '{}' ({function}) is unavailable",
                            c.function_name
                        ),
                        &CELLFUN_ERROR_INVALID_INPUT,
                    ));
                }
                if let Some(function) =
                    user_functions::resolve_semantic_function_by_name(&c.function_name)
                {
                    let request =
                        user_functions::CallableRequest::semantic(function, captures.clone(), 1);
                    if let Some(result) =
                        user_functions::try_call_semantic_descriptor(request).await
                    {
                        return result;
                    }
                }
                call_builtin_async(&c.function_name, &captures).await
            }
            Callable::Special(special) => special.call(args).await,
        }
    }
}

#[derive(Clone)]
enum SpecialCallable {
    ProdOfSize,
    IsClass,
}

impl SpecialCallable {
    async fn call(&self, args: &[Value]) -> BuiltinResult<Value> {
        match self {
            SpecialCallable::ProdOfSize => {
                let value = args.first().ok_or_else(|| {
                    cellfun_error_with_message(
                        "cellfun: prodofsize requires one input",
                        &CELLFUN_ERROR_INVALID_INPUT,
                    )
                })?;
                Ok(Value::Num(value_numel(value).await? as f64))
            }
            SpecialCallable::IsClass => {
                if args.len() < 2 {
                    return Err(cellfun_error_with_message(
                        "cellfun: 'isclass' requires a class name argument",
                        &CELLFUN_ERROR_INVALID_INPUT,
                    ));
                }
                let left = args[0].clone();
                let class_name = extract_string(&args[1]).ok_or_else(|| {
                    cellfun_error_with_message(
                        "cellfun: class name must be a string scalar",
                        &CELLFUN_ERROR_INVALID_INPUT,
                    )
                })?;
                let class_value = call_builtin_async("class", &[left]).await?;
                let class_str = extract_string(&class_value).ok_or_else(|| {
                    cellfun_error_with_message(
                        "cellfun: failed to evaluate class name",
                        &CELLFUN_ERROR_FUNCTION_ERROR,
                    )
                })?;
                Ok(Value::Bool(
                    class_str.eq_ignore_ascii_case(class_name.trim()),
                ))
            }
        }
    }
}

enum UniformCollector {
    Pending,
    Double(Vec<f64>),
    Logical(Vec<u8>),
    Complex(Vec<(f64, f64)>),
}

impl UniformCollector {
    fn push(&mut self, value: &Value) -> BuiltinResult<()> {
        match self {
            UniformCollector::Pending => match classify_value(value)? {
                ClassifiedValue::Logical(b) => {
                    *self = UniformCollector::Logical(vec![b as u8]);
                    Ok(())
                }
                ClassifiedValue::Double(d) => {
                    *self = UniformCollector::Double(vec![d]);
                    Ok(())
                }
                ClassifiedValue::Complex(c) => {
                    *self = UniformCollector::Complex(vec![c]);
                    Ok(())
                }
            },
            UniformCollector::Logical(bits) => match classify_value(value)? {
                ClassifiedValue::Logical(b) => {
                    bits.push(b as u8);
                    Ok(())
                }
                ClassifiedValue::Double(d) => {
                    let mut data: Vec<f64> = bits
                        .iter()
                        .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                        .collect();
                    data.push(d);
                    *self = UniformCollector::Double(data);
                    Ok(())
                }
                ClassifiedValue::Complex(c) => {
                    let mut data: Vec<(f64, f64)> = bits
                        .iter()
                        .map(|&bit| if bit != 0 { (1.0, 0.0) } else { (0.0, 0.0) })
                        .collect();
                    data.push(c);
                    *self = UniformCollector::Complex(data);
                    Ok(())
                }
            },
            UniformCollector::Double(data) => match classify_value(value)? {
                ClassifiedValue::Logical(b) => {
                    data.push(if b { 1.0 } else { 0.0 });
                    Ok(())
                }
                ClassifiedValue::Double(d) => {
                    data.push(d);
                    Ok(())
                }
                ClassifiedValue::Complex(c) => {
                    let promoted: Vec<(f64, f64)> = data.iter().map(|&v| (v, 0.0)).collect();
                    let mut complex = promoted;
                    complex.push(c);
                    *self = UniformCollector::Complex(complex);
                    Ok(())
                }
            },
            UniformCollector::Complex(data) => match classify_value(value)? {
                ClassifiedValue::Logical(b) => {
                    data.push((if b { 1.0 } else { 0.0 }, 0.0));
                    Ok(())
                }
                ClassifiedValue::Double(d) => {
                    data.push((d, 0.0));
                    Ok(())
                }
                ClassifiedValue::Complex(c) => {
                    data.push(c);
                    Ok(())
                }
            },
        }
    }

    fn finish(self, shape: &[usize]) -> BuiltinResult<Value> {
        match self {
            UniformCollector::Pending => {
                let total = total_len(shape).unwrap_or(0);
                let data = vec![0.0; total];
                let tensor = Tensor::new(data, shape.to_vec()).map_err(|e| {
                    cellfun_error_with_message(format!("cellfun: {e}"), &CELLFUN_ERROR_INTERNAL)
                })?;
                Ok(Value::Tensor(tensor))
            }
            UniformCollector::Double(data) => {
                let tensor = Tensor::new(data, shape.to_vec()).map_err(|e| {
                    cellfun_error_with_message(format!("cellfun: {e}"), &CELLFUN_ERROR_INTERNAL)
                })?;
                Ok(Value::Tensor(tensor))
            }
            UniformCollector::Logical(bits) => {
                let logical = LogicalArray::new(bits, shape.to_vec()).map_err(|e| {
                    cellfun_error_with_message(format!("cellfun: {e}"), &CELLFUN_ERROR_INTERNAL)
                })?;
                Ok(Value::LogicalArray(logical))
            }
            UniformCollector::Complex(data) => {
                let complex = ComplexTensor::new(data, shape.to_vec()).map_err(|e| {
                    cellfun_error_with_message(format!("cellfun: {e}"), &CELLFUN_ERROR_INTERNAL)
                })?;
                Ok(Value::ComplexTensor(complex))
            }
        }
    }
}

enum ClassifiedValue {
    Logical(bool),
    Double(f64),
    Complex((f64, f64)),
}

fn classify_value(value: &Value) -> BuiltinResult<ClassifiedValue> {
    match value {
        Value::Bool(b) => Ok(ClassifiedValue::Logical(*b)),
        Value::Num(n) => Ok(ClassifiedValue::Double(*n)),
        Value::Int(iv) => Ok(ClassifiedValue::Double(iv.to_f64())),
        Value::Complex(re, im) => Ok(ClassifiedValue::Complex((*re, *im))),
        Value::Tensor(t) if t.data.len() == 1 => Ok(ClassifiedValue::Double(t.data[0])),
        Value::LogicalArray(la) if la.data.len() == 1 => {
            Ok(ClassifiedValue::Logical(la.data[0] != 0))
        }
        Value::ComplexTensor(ct) if ct.data.len() == 1 => Ok(ClassifiedValue::Complex(ct.data[0])),
        _ => Err(cellfun_error_with_message(
            "cellfun: callback must return scalar values when 'UniformOutput' is true",
            &CELLFUN_ERROR_UNIFORM_OUTPUT,
        )),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, StringArray};
    use std::convert::TryInto;
    use std::sync::Arc;

    fn cellfun_builtin(func: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::cellfun_builtin(func, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn descriptor_signatures_cover_cellfun_forms() {
        let labels: Vec<&str> = CELLFUN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = cellfun(func, C)"));
        assert!(labels.contains(&"Y = cellfun(func, C..., \"UniformOutput\", tf)"));
        assert!(labels.contains(&"Y = cellfun(func, C..., \"ErrorHandler\", errfunc)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_length_uniform_default() {
        let cell = crate::make_cell(
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
                Value::Tensor(Tensor::new(vec![4.0, 5.0, 6.0, 7.0], vec![1, 4]).unwrap()),
                Value::Tensor(Tensor::new(vec![8.0, 9.0], vec![1, 2]).unwrap()),
            ],
            1,
            3,
        )
        .expect("cell");
        let result =
            cellfun_builtin(Value::String("@length".into()), vec![cell]).expect("cellfun length");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![3.0, 4.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_multiple_cells_plus() {
        let left = crate::make_cell(
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
            1,
            3,
        )
        .expect("cell");
        let right = crate::make_cell(
            vec![Value::Num(4.0), Value::Num(5.0), Value::Num(6.0)],
            1,
            3,
        )
        .expect("cell");
        let result = cellfun_builtin(Value::String("@__cellfun_add".into()), vec![left, right])
            .expect("cellfun add");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![5.0, 7.0, 9.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn cellfun_semantic_function_handle_uses_semantic_invoker() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, args, requested_outputs| {
                assert_eq!(function, 77);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(2.0)]);
                Box::pin(async { Ok(Value::Num(9.0)) })
            },
        )));
        let cell = crate::make_cell(vec![Value::Num(2.0)], 1, 1).expect("cell");
        let handle = Value::BoundFunctionHandle {
            name: "cellfun_target".to_string(),
            function: 77,
        };

        let result = cellfun_builtin(handle, vec![cell]).expect("semantic cellfun");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 1]);
                assert_eq!(tensor.data, vec![9.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn cellfun_name_only_callback_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "resolved_cellfun_target").then_some(79)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 79);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(3.0)]);
                Box::pin(async { Ok(Value::Num(13.0)) })
            }),
        ));
        let cell = crate::make_cell(vec![Value::Num(3.0)], 1, 1).expect("cell");

        let result = cellfun_builtin(
            Value::String("resolved_cellfun_target".to_string()),
            vec![cell],
        )
        .expect("resolved name-only cellfun");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 1]);
                assert_eq!(tensor.data, vec![13.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn cellfun_qualified_text_callback_classifies_as_external_name() {
        let callable =
            Callable::from_text("pkg.callback", true).expect("qualified callback should parse");
        assert!(matches!(
            callable,
            Callable::ExternalName { name } if name == "pkg.callback"
        ));
    }

    #[test]
    fn cellfun_external_handle_uses_semantic_resolver() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.callback").then_some(86)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 86);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(2.0)]);
                Box::pin(async { Ok(Value::Num(23.0)) })
            }),
        ));
        let cell = crate::make_cell(vec![Value::Num(2.0)], 1, 1).expect("cell");

        let result = cellfun_builtin(
            Value::ExternalFunctionHandle("pkg.callback".to_string()),
            vec![cell],
        )
        .expect("resolved external-handle cellfun");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 1]);
                assert_eq!(tensor.data, vec![23.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn cellfun_single_segment_external_handle_uses_runtime_name_resolution() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "callback").then_some(886)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 886);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(2.0)]);
                Box::pin(async { Ok(Value::Num(24.0)) })
            }),
        ));
        let cell = crate::make_cell(vec![Value::Num(2.0)], 1, 1).expect("cell");

        let result = cellfun_builtin(
            Value::ExternalFunctionHandle("callback".to_string()),
            vec![cell],
        )
        .expect("single-segment external-handle cellfun should resolve via runtime-name policy");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 1]);
                assert_eq!(tensor.data, vec![24.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn cellfun_external_handle_prefers_semantic_handle_binding_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.callback").then_some(86)
            })));
        let callable =
            Callable::from_function(Value::ExternalFunctionHandle("pkg.callback".to_string()))
                .expect("external handle should parse");
        assert!(matches!(
            callable,
            Callable::Closure(Closure {
                function_name,
                bound_function: Some(86),
                ..
            }) if function_name == "pkg.callback"
        ));
    }

    #[test]
    fn cellfun_name_only_closure_prefers_semantic_handle_binding_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.callback").then_some(186)
            })));
        let callable = Callable::from_function(Value::Closure(Closure {
            function_name: "pkg.callback".to_string(),
            bound_function: None,
            captures: vec![Value::Num(9.0)],
        }))
        .expect("closure callback should parse");
        assert!(matches!(
            callable,
            Callable::Closure(Closure {
                function_name,
                bound_function: Some(186),
                captures
            }) if function_name == "pkg.callback" && captures == vec![Value::Num(9.0)]
        ));
    }

    #[test]
    fn cellfun_name_only_closure_call_uses_semantic_resolver_when_unbound() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.callback").then_some(286)
            })));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 286);
                assert_eq!(requested_outputs, 1);
                assert_eq!(args, &[Value::Num(9.0), Value::Num(4.0)]);
                Box::pin(async { Ok(Value::Num(13.0)) })
            }),
        ));
        let callable = Callable::Closure(Closure {
            function_name: "pkg.callback".to_string(),
            bound_function: None,
            captures: vec![Value::Num(9.0)],
        });
        let value = block_on(callable.call(&[Value::Num(4.0)])).expect("closure call");
        assert_eq!(value, Value::Num(13.0));
    }

    #[test]
    fn cellfun_external_handle_errors_as_undefined_when_unresolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_| None)));
        let cell = crate::make_cell(vec![Value::Num(2.0)], 1, 1).expect("cell");

        let err = cellfun_builtin(
            Value::ExternalFunctionHandle("pkg.callback".to_string()),
            vec![cell],
        )
        .expect_err("unresolved external callback should error");
        assert_eq!(
            err.identifier(),
            CELLFUN_ERROR_UNDEFINED_FUNCTION.identifier,
            "unexpected error: {}",
            err.message()
        );
        assert!(
            err.message().contains("ExternalName(QualifiedName"),
            "unexpected error: {err:?}"
        );
        assert!(
            !err.message().contains("Undefined function 'pkg.callback'"),
            "well-formed external callback should report typed identity: {err:?}"
        );
    }

    #[test]
    fn cellfun_malformed_external_handle_errors_as_undefined_when_unresolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|_| None)));
        let cell = crate::make_cell(vec![Value::Num(2.0)], 1, 1).expect("cell");

        let err = cellfun_builtin(
            Value::ExternalFunctionHandle("pkg..callback".to_string()),
            vec![cell],
        )
        .expect_err("malformed unresolved external callback should error");
        assert_eq!(
            err.identifier(),
            CELLFUN_ERROR_UNDEFINED_FUNCTION.identifier,
            "unexpected error: {}",
            err.message()
        );
        assert!(
            err.message().contains("pkg..callback"),
            "unexpected error: {err:?}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_uniform_false_returns_cells() {
        let cell = crate::make_cell(
            vec![
                Value::String("Ada".into()),
                Value::String("Linus".into()),
                Value::String("Katherine".into()),
            ],
            1,
            3,
        )
        .expect("cell");
        let result = cellfun_builtin(
            Value::String("@upper".into()),
            vec![
                cell,
                Value::String("UniformOutput".into()),
                Value::Bool(false),
            ],
        )
        .expect("cellfun upper");
        match result {
            Value::Cell(ca) => {
                assert_eq!(ca.shape, vec![1, 3]);
                let upper_a = ca.data[0].clone();
                let upper_b = ca.data[1].clone();
                let upper_c = ca.data[2].clone();
                assert_eq!(extract_string(&upper_a).unwrap(), "ADA");
                assert_eq!(extract_string(&upper_b).unwrap(), "LINUS");
                assert_eq!(extract_string(&upper_c).unwrap(), "KATHERINE");
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_error_handler_recovers() {
        let cells = crate::make_cell(
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
            1,
            3,
        )
        .expect("cell");
        let handler = Value::Closure(Closure {
            function_name: "__cellfun_test_handler".into(),
            bound_function: None,
            captures: vec![Value::Num(0.0)],
        });
        let result = cellfun_builtin(
            Value::String("@nonexistent_builtin".into()),
            vec![cells, Value::String("ErrorHandler".into()), handler],
        )
        .expect("cellfun error handler");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![0.0, 0.0, 0.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_string_identifier() {
        let cells = crate::make_cell(
            vec![
                Value::CharArray(runmat_builtins::CharArray::new_row("")),
                Value::CharArray(runmat_builtins::CharArray::new_row("abc")),
                Value::CharArray(runmat_builtins::CharArray::new_row("")),
            ],
            1,
            3,
        )
        .expect("cell");
        let result = cellfun_builtin(
            Value::CharArray(runmat_builtins::CharArray::new_row("isempty")),
            vec![cells],
        )
        .expect("isempty");
        match result {
            Value::LogicalArray(la) => {
                assert_eq!(la.shape, vec![1, 3]);
                assert_eq!(la.data, vec![1, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_string_array_identifier() {
        let cells = crate::make_cell(
            vec![Value::CharArray(runmat_builtins::CharArray::new_row(""))],
            1,
            1,
        )
        .expect("cell");
        let sa = StringArray::new(vec!["isempty".into()], vec![1, 1]).unwrap();
        let result =
            cellfun_builtin(Value::StringArray(sa), vec![cells]).expect("cellfun string array");
        match result {
            Value::LogicalArray(la) => {
                assert_eq!(la.shape, vec![1, 1]);
                assert_eq!(la.data, vec![1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_uniform_true_non_scalar_errors() {
        let cells = crate::make_cell(
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
            )],
            1,
            1,
        )
        .expect("cell");
        let err = cellfun_builtin(Value::String("@eye".into()), vec![cells])
            .unwrap_err()
            .to_string();
        assert!(
            err.to_ascii_lowercase().contains("uniformoutput"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_uniform_promotes_logical_to_double() {
        let cells = crate::make_cell(vec![Value::Bool(true), Value::Num(2.5)], 1, 2).unwrap();
        let result = cellfun_builtin(Value::String("@__cellfun_identity".into()), vec![cells])
            .expect("cellfun identity");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![1.0, 2.5]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_uniform_promotes_double_to_complex() {
        let cells =
            crate::make_cell(vec![Value::Num(2.0), Value::Complex(0.0, 1.0)], 1, 2).unwrap();
        let result = cellfun_builtin(Value::String("@__cellfun_identity".into()), vec![cells])
            .expect("cellfun identity");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert_eq!(ct.data, vec![(2.0, 0.0), (0.0, 1.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_errors_on_mismatched_cell_sizes() {
        let first = crate::make_cell(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let second = crate::make_cell(vec![Value::Num(3.0)], 1, 1).unwrap();
        let err = cellfun_builtin(
            Value::String("@__cellfun_identity".into()),
            vec![first, second],
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.to_ascii_lowercase().contains("size"),
            "expected size mismatch error, got: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_uniformoutput_accepts_char_flags() {
        let strings =
            crate::make_cell(vec![Value::String("Ada".into())], 1, 1).expect("cell creation");
        let result = cellfun_builtin(
            Value::String("@upper".into()),
            vec![
                strings,
                Value::CharArray(runmat_builtins::CharArray::new_row("UniformOutput")),
                Value::CharArray(runmat_builtins::CharArray::new_row("off")),
            ],
        )
        .expect("cellfun upper char flag");
        assert!(
            matches!(result, Value::Cell(_)),
            "expected cell array result when UniformOutput is 'off'"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_isclass_special_case() {
        let ints = crate::make_cell(
            vec![
                Value::Int(IntValue::I32(5)),
                Value::Num(std::f64::consts::PI),
                Value::Int(IntValue::I16(2)),
            ],
            1,
            3,
        )
        .expect("cell");
        let result = cellfun_builtin(
            Value::String("isclass".into()),
            vec![ints, Value::String("int32".into())],
        )
        .expect("cellfun isclass");
        match result {
            Value::LogicalArray(la) => {
                assert_eq!(la.data, vec![1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_passes_additional_arguments() {
        let matrices = crate::make_cell(
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
                Value::Tensor(Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap()),
            ],
            1,
            2,
        )
        .expect("cell");
        let dimension = Value::Num(2.0);
        let result = cellfun_builtin(Value::String("size".into()), vec![matrices, dimension])
            .expect("cellfun size");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![2.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_handles_string_array_uniform_false() {
        let sa = StringArray::new(vec!["foo".into(), "bar".into()], vec![1, 2]).unwrap();
        let cell = crate::make_cell(vec![Value::StringArray(sa)], 1, 1).unwrap();
        let result = cellfun_builtin(
            Value::String("@strlength".into()),
            vec![
                cell,
                Value::String("UniformOutput".into()),
                Value::Bool(false),
            ],
        )
        .unwrap();
        match result {
            Value::Cell(ca) => {
                assert_eq!(ca.shape, vec![1, 1]);
                let inner = ca.data[0].clone();
                match inner {
                    Value::Tensor(t) => assert_eq!(t.data, vec![3.0, 3.0]),
                    _ => panic!("expected tensor inside cell"),
                }
            }
            other => panic!("expected cell, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_gathers_gpu_inputs() {
        test_support::with_test_provider(|provider| {
            let angle = std::f64::consts::PI / 6.0;
            let tensor = Tensor::new(vec![angle], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let cell = crate::make_cell(vec![Value::GpuTensor(handle)], 1, 1).expect("cell");
            let result =
                cellfun_builtin(Value::String("@sin".into()), vec![cell]).expect("cellfun sin");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            let expected = angle.sin();
            assert!((gathered.data[0] - expected).abs() < 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cellfun_with_wgpu_provider_handles_gpu_cells() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let value = Tensor::new(vec![0.25], vec![1, 1]).unwrap();
        let view = HostTensorView {
            data: &value.data,
            shape: &value.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let cell = crate::make_cell(vec![Value::GpuTensor(handle)], 1, 1).expect("cell");

        let result =
            cellfun_builtin(Value::String("@sin".into()), vec![cell]).expect("cellfun sin");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 1]);
        let expected = value.data[0].sin();
        assert!((gathered.data[0] - expected).abs() < 1e-12);
    }

    const CELLFUN_TEST_HELPER_ERRORS: [BuiltinErrorDescriptor; 0] = [];
    const CELLFUN_TEST_HELPER_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
        name: "out",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Helper output value.",
    }];
    const CELLFUN_TEST_HANDLER_INPUTS: [BuiltinParamDescriptor; 3] = [
        BuiltinParamDescriptor {
            name: "seed",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Seed value.",
        },
        BuiltinParamDescriptor {
            name: "err",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Error context placeholder.",
        },
        BuiltinParamDescriptor {
            name: "rest",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Variadic,
            default: None,
            description: "Additional values.",
        },
    ];
    const CELLFUN_ADD_INPUTS: [BuiltinParamDescriptor; 2] = [
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
    const CELLFUN_IDENTITY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input value.",
    }];
    const CELLFUN_TEST_HANDLER_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
        [BuiltinSignatureDescriptor {
            label: "out = __cellfun_test_handler(seed, err, ...)",
            inputs: &CELLFUN_TEST_HANDLER_INPUTS,
            outputs: &CELLFUN_TEST_HELPER_OUT,
        }];
    const CELLFUN_ADD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
        label: "out = __cellfun_add(lhs, rhs)",
        inputs: &CELLFUN_ADD_INPUTS,
        outputs: &CELLFUN_TEST_HELPER_OUT,
    }];
    const CELLFUN_IDENTITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
        [BuiltinSignatureDescriptor {
            label: "out = __cellfun_identity(value)",
            inputs: &CELLFUN_IDENTITY_INPUTS,
            outputs: &CELLFUN_TEST_HELPER_OUT,
        }];
    const CELLFUN_TEST_HANDLER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
        signatures: &CELLFUN_TEST_HANDLER_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::HiddenInternal,
        errors: &CELLFUN_TEST_HELPER_ERRORS,
    };
    const CELLFUN_ADD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
        signatures: &CELLFUN_ADD_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::HiddenInternal,
        errors: &CELLFUN_TEST_HELPER_ERRORS,
    };
    const CELLFUN_IDENTITY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
        signatures: &CELLFUN_IDENTITY_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::HiddenInternal,
        errors: &CELLFUN_TEST_HELPER_ERRORS,
    };

    #[runmat_macros::runtime_builtin(
        name = "__cellfun_test_handler",
        descriptor(crate::builtins::cells::core::cellfun::tests::CELLFUN_TEST_HANDLER_DESCRIPTOR),
        type_resolver(cellfun_type),
        builtin_path = "crate::builtins::cells::core::cellfun::tests"
    )]
    fn cellfun_test_handler(
        seed: Value,
        _err: Value,
        rest: Vec<Value>,
    ) -> crate::BuiltinResult<Value> {
        // Return the captured seed regardless of the inputs; ensure rest is present for coverage.
        let _ = rest;
        Ok(seed)
    }

    #[runmat_macros::runtime_builtin(
        name = "__cellfun_add",
        descriptor(crate::builtins::cells::core::cellfun::tests::CELLFUN_ADD_DESCRIPTOR),
        type_resolver(cellfun_type),
        builtin_path = "crate::builtins::cells::core::cellfun::tests"
    )]
    fn cellfun_add(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
        let a: f64 = (&lhs).try_into()?;
        let b: f64 = (&rhs).try_into()?;
        Ok(Value::Num(a + b))
    }

    #[runmat_macros::runtime_builtin(
        name = "__cellfun_identity",
        descriptor(crate::builtins::cells::core::cellfun::tests::CELLFUN_IDENTITY_DESCRIPTOR),
        type_resolver(cellfun_type),
        builtin_path = "crate::builtins::cells::core::cellfun::tests"
    )]
    fn cellfun_identity(value: Value) -> crate::BuiltinResult<Value> {
        Ok(value)
    }
}
