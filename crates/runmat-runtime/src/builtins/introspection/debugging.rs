use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_thread_local::runmat_thread_local;
use runmat_value::{CellArray, NumericDType, StructValue, Tensor, Value};
use std::cell::RefCell;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::builtins::common::path_search::{file_candidates, path_is_file};
use crate::builtins::common::tensor;
use crate::console::{record_console_line, ConsoleStream};
use crate::{BuiltinResult, RuntimeError};

const DBSTACK_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ST",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Call-stack entries in RunMat's column-cell representation of a structure array.",
    },
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Current workspace index within the returned stack.",
    },
];

const DBSTACK_STACK_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ST",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Call-stack entries in RunMat's column-cell representation of a structure array.",
}];

const DBSTACK_N_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of leading stack entries to omit.",
}];

const DBSTACK_OPTION_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "`-completenames` requests full source paths.",
}];

const DBSTACK_OPTION_N_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "`-completenames` requests full source paths.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of leading stack entries to omit.",
    },
];

const DBSTACK_N_OPTION_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of leading stack entries to omit.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "`-completenames` requests full source paths.",
    },
];

const DBSTACK_SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
    BuiltinSignatureDescriptor {
        label: "dbstack",
        inputs: &[],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "ST = dbstack",
        inputs: &[],
        outputs: &DBSTACK_STACK_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ST = dbstack(n)",
        inputs: &DBSTACK_N_INPUT,
        outputs: &DBSTACK_STACK_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ST = dbstack('-completenames')",
        inputs: &DBSTACK_OPTION_INPUT,
        outputs: &DBSTACK_STACK_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ST = dbstack('-completenames', n)",
        inputs: &DBSTACK_OPTION_N_INPUTS,
        outputs: &DBSTACK_STACK_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ST = dbstack(n, '-completenames')",
        inputs: &DBSTACK_N_OPTION_INPUTS,
        outputs: &DBSTACK_STACK_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[ST,I] = dbstack",
        inputs: &[],
        outputs: &DBSTACK_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[ST,I] = dbstack(n)",
        inputs: &DBSTACK_N_INPUT,
        outputs: &DBSTACK_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[ST,I] = dbstack('-completenames')",
        inputs: &DBSTACK_OPTION_INPUT,
        outputs: &DBSTACK_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[ST,I] = dbstack('-completenames', n)",
        inputs: &DBSTACK_OPTION_N_INPUTS,
        outputs: &DBSTACK_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[ST,I] = dbstack(n, '-completenames')",
        inputs: &DBSTACK_N_OPTION_INPUTS,
        outputs: &DBSTACK_OUTPUTS,
    },
];

const TEXT_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Function, script, file, command, or option text.",
}];

const DBTYPE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "file",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "MATLAB source file, path, or function name.",
    },
    BuiltinParamDescriptor {
        name: "range",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional line range such as `3:8`.",
    },
];

const DBTYPE_FILE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "file",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "MATLAB source file, path, or function name.",
}];

const DBTYPE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "dbtype file",
        inputs: &DBTYPE_FILE_INPUT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "dbtype file start:end",
        inputs: &DBTYPE_INPUTS,
        outputs: &[],
    },
];

const SIMPLE_NO_OUTPUT_SIGNATURE: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "keyboard",
    inputs: &[],
    outputs: &[],
}];

const MLOCK_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "mlock",
    inputs: &[],
    outputs: &[],
}];

const MUNLOCK_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "munlock",
        inputs: &[],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "munlock(fun)",
        inputs: &TEXT_INPUT,
        outputs: &[],
    },
];

const MISLOCKED_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical scalar indicating whether the function or script is locked.",
}];

const MISLOCKED_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = mislocked",
        inputs: &[],
        outputs: &MISLOCKED_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = mislocked(fun)",
        inputs: &TEXT_INPUT,
        outputs: &MISLOCKED_OUTPUT,
    },
];

const DBSTATUS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "dbstatus",
        inputs: &TEXT_INPUT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "b = dbstatus(file, '-completenames')",
        inputs: &TEXT_INPUT,
        outputs: &DBSTATUS_OUTPUT,
    },
];

const DBSTATUS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Breakpoint entries as a 1-by-N cell row; currently empty until execution breakpoints are implemented.",
}];

const DBCLEAR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "dbclear all | dbclear in file at location | dbclear if condition",
    inputs: &TEXT_INPUT,
    outputs: &[],
}];

const GETCALLINFO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "info = getcallinfo",
    inputs: &[],
    outputs: &GETCALLINFO_OUTPUT,
}];

const GETCALLINFO_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar struct describing the current call context and stack.",
}];

pub const DEBUG_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEBUG.TOO_MANY_INPUTS",
    identifier: Some("RunMat:TooManyInputs"),
    when: "A debugger compatibility helper receives more inputs than it supports.",
    message: "debugger helper: too many input arguments",
};

pub const DEBUG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEBUG.INVALID_INPUT",
    identifier: Some("RunMat:InvalidInput"),
    when: "An input is not a supported text, scalar, or option value.",
    message: "debugger helper: invalid input",
};

pub const DEBUG_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEBUG.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:dbstack:TooManyOutputs"),
    when: "More than two outputs are requested from dbstack.",
    message: "dbstack: too many output arguments",
};

pub const DEBUG_ERROR_NO_CURRENT_FILE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEBUG.NO_CURRENT_FILE",
    identifier: Some("RunMat:NoCurrentFile"),
    when: "`mlock`, `munlock`, or `mislocked` is called without a current function or script.",
    message: "debugger helper: no current function or script is available",
};

pub const DEBUG_ERROR_FILE_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEBUG.FILE_NOT_FOUND",
    identifier: Some("RunMat:FileNotFound"),
    when: "`dbtype` cannot resolve a MATLAB source file.",
    message: "dbtype: file not found",
};

pub const DEBUG_ERROR_FILE_READ: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEBUG.FILE_READ",
    identifier: Some("RunMat:FileReadFailed"),
    when: "`dbtype` cannot read the resolved source file.",
    message: "dbtype: failed to read file",
};

pub const DEBUG_ERRORS: [BuiltinErrorDescriptor; 5] = [
    DEBUG_ERROR_TOO_MANY_INPUTS,
    DEBUG_ERROR_INVALID_INPUT,
    DEBUG_ERROR_NO_CURRENT_FILE,
    DEBUG_ERROR_FILE_NOT_FOUND,
    DEBUG_ERROR_FILE_READ,
];

pub const DBSTACK_ERRORS: [BuiltinErrorDescriptor; 6] = [
    DEBUG_ERROR_TOO_MANY_INPUTS,
    DEBUG_ERROR_TOO_MANY_OUTPUTS,
    DEBUG_ERROR_INVALID_INPUT,
    DEBUG_ERROR_NO_CURRENT_FILE,
    DEBUG_ERROR_FILE_NOT_FOUND,
    DEBUG_ERROR_FILE_READ,
];

pub const DBSTACK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DBSTACK_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DBSTACK_ERRORS,
};

const DBSTACK_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "n is an exact nonnegative scalar count of leading stack entries to omit; values outside the host indexing domain reject.",
    }];

pub const DBSTACK_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "ST or [ST, I] = dbstack(n) or dbstack('-completenames', n)",
        inputs: &DBSTACK_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The skip count is parsed from exact scalar storage. Stack line and workspace-index fields follow the debugger's host metadata representation rather than n's input class.",
    }];

pub const DBTYPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DBTYPE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DEBUG_ERRORS,
};

pub const KEYBOARD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIMPLE_NO_OUTPUT_SIGNATURE,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

pub const MLOCK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MLOCK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DEBUG_ERRORS,
};

pub const MUNLOCK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MUNLOCK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DEBUG_ERRORS,
};

pub const MISLOCKED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MISLOCKED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DEBUG_ERRORS,
};

pub const DBSTATUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DBSTATUS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DEBUG_ERRORS,
};

pub const DBCLEAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DBCLEAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DEBUG_ERRORS,
};

pub const GETCALLINFO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GETCALLINFO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DEBUG_ERRORS,
};

runmat_thread_local! {
    static LOCKED_FUNCTIONS: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
}

pub fn reset_lock_registry_for_tests() {
    LOCKED_FUNCTIONS.with(|locks| locks.borrow_mut().clear());
}

fn debug_error(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        descriptor.message.to_string()
    } else {
        format!("{}: {detail}", descriptor.message)
    };
    let mut builder = crate::build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn empty_value() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor"))
}

fn text_value(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        _ => None,
    }
}

fn integer_value(value: &Value) -> Option<usize> {
    match value {
        Value::Num(n) => nonnegative_platform_usize(*n),
        Value::Tensor(tensor)
            if tensor.numeric_dtype() == NumericDType::F64
                && tensor::tensor_element_len(tensor) == 1 =>
        {
            nonnegative_platform_usize(tensor::tensor_value_f64(tensor, 0))
        }
        _ => tensor::scalar_integer_value(value).and_then(|int| int.try_to_usize()),
    }
}

fn nonnegative_platform_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
}

fn stack_struct(frame: &crate::debug_context::DebugFrameInfo, complete_names: bool) -> Value {
    let mut st = StructValue::new();
    let file = if complete_names {
        frame.file.clone()
    } else {
        Path::new(&frame.file)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&frame.file)
            .to_string()
    };
    st.insert("file", Value::String(file));
    st.insert("name", Value::String(frame.function.clone()));
    st.insert("line", Value::Num(frame.line as f64));
    Value::Struct(st)
}

fn cell_column(values: Vec<Value>) -> BuiltinResult<Value> {
    if values.is_empty() {
        return Ok(Value::Cell(CellArray::new(Vec::new(), 0, 0).map_err(
            |err| debug_error("debugger", &DEBUG_ERROR_INVALID_INPUT, &err),
        )?));
    }
    let rows = values.len();
    Ok(Value::Cell(CellArray::new(values, rows, 1).map_err(
        |err| debug_error("debugger", &DEBUG_ERROR_INVALID_INPUT, &err),
    )?))
}

fn stack_value(skip: usize, complete_names: bool) -> BuiltinResult<Value> {
    let mut frames = crate::debug_context::current_frames();
    if frames.is_empty() {
        if let Some(source) = crate::source_context::current_source_info() {
            frames.push(crate::debug_context::DebugFrameInfo {
                function: source.name.to_string(),
                file: source
                    .fullpath_name
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| source.name.to_string()),
                line: 0,
            });
        }
    }
    let entries = frames
        .iter()
        .skip(skip)
        .map(|frame| stack_struct(frame, complete_names))
        .collect::<Vec<_>>();
    cell_column(entries)
}

fn parse_dbstack_args(args: &[Value]) -> BuiltinResult<(usize, bool)> {
    if args.len() > 2 {
        return Err(debug_error("dbstack", &DEBUG_ERROR_TOO_MANY_INPUTS, ""));
    }
    match args {
        [] => Ok((0, false)),
        [arg] if is_complete_names(arg) => Ok((0, true)),
        [arg] => integer_value(arg)
            .map(|skip| (skip, false))
            .ok_or_else(|| debug_error("dbstack", &DEBUG_ERROR_INVALID_INPUT, "")),
        [option, n] if is_complete_names(option) => integer_value(n)
            .map(|skip| (skip, true))
            .ok_or_else(|| debug_error("dbstack", &DEBUG_ERROR_INVALID_INPUT, "")),
        [n, option] if is_complete_names(option) => integer_value(n)
            .map(|skip| (skip, true))
            .ok_or_else(|| debug_error("dbstack", &DEBUG_ERROR_INVALID_INPUT, "")),
        _ => Err(debug_error("dbstack", &DEBUG_ERROR_INVALID_INPUT, "")),
    }
}

fn is_complete_names(value: &Value) -> bool {
    text_value(value).is_some_and(|text| text.trim().eq_ignore_ascii_case("-completenames"))
}

pub(crate) fn dispatch_dbstack(args: Vec<Value>) -> BuiltinResult<Value> {
    let (skip, complete_names) = parse_dbstack_args(&args)?;
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 2) {
        return Err(debug_error("dbstack", &DEBUG_ERROR_TOO_MANY_OUTPUTS, ""));
    }
    let stack = stack_value(skip, complete_names)?;
    match crate::output_count::current_output_count() {
        Some(0) => {
            record_console_line(ConsoleStream::Stdout, format_stack_for_display(&stack));
            Ok(empty_value())
        }
        Some(n) if n >= 2 => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![stack, Value::Num(1.0)],
        )),
        _ => Ok(stack),
    }
}

fn format_stack_for_display(stack: &Value) -> String {
    let Value::Cell(cell) = stack else {
        return String::new();
    };
    let lines = cell
        .data
        .iter()
        .filter_map(|value| {
            let Value::Struct(st) = value else {
                return None;
            };
            let name = st
                .fields
                .get("name")
                .and_then(text_value)
                .unwrap_or_default();
            let file = st
                .fields
                .get("file")
                .and_then(text_value)
                .unwrap_or_default();
            let line = st
                .fields
                .get("line")
                .and_then(|value| match value {
                    Value::Num(n) => nonnegative_platform_usize(*n),
                    Value::Int(int) => int.try_to_usize(),
                    _ => None,
                })
                .unwrap_or(0);
            Some(if file.is_empty() {
                format!("In {name} at line {line}")
            } else {
                format!("In {name} ({file}) at line {line}")
            })
        })
        .collect::<Vec<_>>();
    lines.join("\n")
}

fn normalize_lock_key(raw: &str) -> String {
    let path = Path::new(raw.trim());
    let name = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| raw.trim());
    name.to_ascii_lowercase()
}

fn current_lock_key() -> BuiltinResult<String> {
    if let Some(name) = crate::debug_context::current_function_name() {
        if !name.is_empty() && name != "<main>" && name != "<anonymous>" {
            return Ok(normalize_lock_key(&name));
        }
    }
    crate::source_context::current_source_info()
        .map(|source| normalize_lock_key(&source.name))
        .filter(|key| !key.is_empty())
        .ok_or_else(|| debug_error("mlock", &DEBUG_ERROR_NO_CURRENT_FILE, ""))
}

fn optional_lock_key(args: &[Value], builtin: &'static str) -> BuiltinResult<String> {
    match args {
        [] => current_lock_key(),
        [value] => text_value(value)
            .map(|text| normalize_lock_key(&text))
            .filter(|key| !key.is_empty())
            .ok_or_else(|| debug_error(builtin, &DEBUG_ERROR_INVALID_INPUT, "")),
        _ => Err(debug_error(builtin, &DEBUG_ERROR_TOO_MANY_INPUTS, "")),
    }
}

pub(crate) fn dispatch_mlock(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(debug_error("mlock", &DEBUG_ERROR_TOO_MANY_INPUTS, ""));
    }
    let key = current_lock_key()?;
    LOCKED_FUNCTIONS.with(|locks| {
        locks.borrow_mut().insert(key);
    });
    Ok(empty_value())
}

pub(crate) fn dispatch_munlock(args: Vec<Value>) -> BuiltinResult<Value> {
    let key = optional_lock_key(&args, "munlock")?;
    LOCKED_FUNCTIONS.with(|locks| {
        locks.borrow_mut().remove(&key);
    });
    Ok(empty_value())
}

pub(crate) fn dispatch_mislocked(args: Vec<Value>) -> BuiltinResult<Value> {
    let key = optional_lock_key(&args, "mislocked")?;
    let locked = LOCKED_FUNCTIONS.with(|locks| locks.borrow().contains(&key));
    Ok(Value::Bool(locked))
}

fn empty_breakpoint_list() -> BuiltinResult<Value> {
    cell_column(Vec::new())
}

pub(crate) fn dispatch_dbstatus(args: Vec<Value>) -> BuiltinResult<Value> {
    for arg in &args {
        if text_value(arg).is_none() {
            return Err(debug_error("dbstatus", &DEBUG_ERROR_INVALID_INPUT, ""));
        }
    }
    let status = empty_breakpoint_list()?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(empty_value()),
        _ => Ok(status),
    }
}

pub(crate) fn dispatch_dbclear(args: Vec<Value>) -> BuiltinResult<Value> {
    for arg in &args {
        if text_value(arg).is_none() {
            return Err(debug_error("dbclear", &DEBUG_ERROR_INVALID_INPUT, ""));
        }
    }
    Ok(empty_value())
}

fn parse_line_range(value: Option<&Value>) -> BuiltinResult<Option<(usize, usize)>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let text =
        text_value(value).ok_or_else(|| debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, ""))?;
    let Some((start, end)) = text.split_once(':') else {
        let line = text
            .trim()
            .parse::<usize>()
            .map_err(|_| debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, text.clone()))?;
        return Ok(Some((line, line)));
    };
    let start = start
        .trim()
        .parse::<usize>()
        .map_err(|_| debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, text.clone()))?;
    let end = end
        .trim()
        .parse::<usize>()
        .map_err(|_| debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, text.clone()))?;
    if start == 0 || end == 0 || start > end {
        return Err(debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, text));
    }
    Ok(Some((start, end)))
}

async fn resolve_dbtype_path(name: &str) -> Result<Option<PathBuf>, String> {
    for candidate in file_candidates(name, &[".m", ""], "dbtype")? {
        if path_is_file(&candidate).await {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

pub(crate) async fn dispatch_dbtype(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, ""));
    }
    if args.len() > 2 {
        return Err(debug_error("dbtype", &DEBUG_ERROR_TOO_MANY_INPUTS, ""));
    }
    let file = text_value(&args[0])
        .ok_or_else(|| debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, "file must be text"))?;
    let range = parse_line_range(args.get(1))?;
    let path = resolve_dbtype_path(&file)
        .await
        .map_err(|err| debug_error("dbtype", &DEBUG_ERROR_FILE_NOT_FOUND, err))?
        .ok_or_else(|| debug_error("dbtype", &DEBUG_ERROR_FILE_NOT_FOUND, file.clone()))?;
    let text = runmat_filesystem::read_to_string_async(&path)
        .await
        .map_err(|err| debug_error("dbtype", &DEBUG_ERROR_FILE_READ, err.to_string()))?;
    let lines = text.lines().collect::<Vec<_>>();
    let (start, end) = range.unwrap_or((1, lines.len()));
    let rendered = lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| {
            let line_number = idx + 1;
            if line_number < start || line_number > end {
                None
            } else {
                Some(format!("{line_number:>5} {line}"))
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    record_console_line(ConsoleStream::Stdout, rendered);
    Ok(empty_value())
}

pub(crate) fn dispatch_keyboard(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(debug_error("keyboard", &DEBUG_ERROR_TOO_MANY_INPUTS, ""));
    }
    Ok(empty_value())
}

pub(crate) fn dispatch_getcallinfo(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(debug_error("getcallinfo", &DEBUG_ERROR_TOO_MANY_INPUTS, ""));
    }
    let mut info = StructValue::new();
    let current = crate::debug_context::current_frames()
        .into_iter()
        .next()
        .unwrap_or_else(|| crate::debug_context::DebugFrameInfo {
            function: String::new(),
            file: String::new(),
            line: 0,
        });
    info.insert("name", Value::String(current.function));
    info.insert("file", Value::String(current.file));
    info.insert("line", Value::Num(current.line as f64));
    info.insert("stack", stack_value(0, false)?);
    Ok(Value::Struct(info))
}

#[runtime_builtin(
    name = "dbstack",
    category = "introspection",
    summary = "Return or display the active RunMat call stack.",
    descriptor(self::DBSTACK_DESCRIPTOR),
    integer_capabilities(self::DBSTACK_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn dbstack_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    dispatch_dbstack(args)
}

#[runtime_builtin(
    name = "dbtype",
    category = "introspection",
    summary = "Display MATLAB source text with line numbers.",
    sink = true,
    suppress_auto_output = true,
    descriptor(self::DBTYPE_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::debugging"
)]
async fn dbtype_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    dispatch_dbtype(args).await
}

#[runtime_builtin(
    name = "keyboard",
    category = "introspection",
    summary = "Compatibility no-op for MATLAB keyboard debugging stops.",
    sink = true,
    suppress_auto_output = true,
    descriptor(self::KEYBOARD_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn keyboard_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    dispatch_keyboard(args)
}

#[runtime_builtin(
    name = "mlock",
    category = "introspection",
    summary = "Mark the current function or script as locked in RunMat's compatibility registry.",
    sink = true,
    suppress_auto_output = true,
    descriptor(self::MLOCK_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn mlock_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    dispatch_mlock(args)
}

#[runtime_builtin(
    name = "munlock",
    category = "introspection",
    summary = "Remove a function or script from RunMat's lock compatibility registry.",
    sink = true,
    suppress_auto_output = true,
    descriptor(self::MUNLOCK_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn munlock_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    dispatch_munlock(args)
}

#[runtime_builtin(
    name = "mislocked",
    category = "introspection",
    summary = "Return whether a function or script is in RunMat's lock compatibility registry.",
    descriptor(self::MISLOCKED_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn mislocked_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    dispatch_mislocked(args)
}

#[runtime_builtin(
    name = "dbstatus",
    category = "introspection",
    summary = "Return RunMat debugger breakpoint status.",
    descriptor(self::DBSTATUS_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn dbstatus_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    dispatch_dbstatus(args)
}

#[runtime_builtin(
    name = "dbclear",
    category = "introspection",
    summary = "Clear RunMat debugger breakpoint status entries.",
    sink = true,
    suppress_auto_output = true,
    descriptor(self::DBCLEAR_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn dbclear_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    dispatch_dbclear(args)
}

#[runtime_builtin(
    name = "getcallinfo",
    category = "introspection",
    summary = "Return deterministic information about the current RunMat call context.",
    descriptor(self::GETCALLINFO_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn getcallinfo_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    dispatch_getcallinfo(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_hir::SourceId;
    use runmat_value::{IntValue, IntegerStorage};

    fn cell_len(value: &Value) -> usize {
        let Value::Cell(cell) = value else {
            panic!("expected cell row, got {value:?}");
        };
        cell.data.len()
    }

    #[test]
    fn typed_debug_offsets_preserve_platform_representable_uint64() {
        assert_eq!(integer_value(&Value::Int(IntValue::U64(3))), Some(3));
        assert_eq!(
            integer_value(&Value::Int(IntValue::U64(u64::MAX))),
            usize::try_from(u64::MAX).ok()
        );
        assert_eq!(integer_value(&Value::Int(IntValue::I64(-1))), None);
        assert_eq!(integer_value(&Value::Num(1.5)), None);
        assert_eq!(integer_value(&Value::Num(usize::MAX as f64 + 1.0)), None);
        if usize::BITS == 64 {
            assert_eq!(integer_value(&Value::Num(usize::MAX as f64)), None);
        }
    }

    #[test]
    fn dbstack_accepts_every_exact_integer_scalar_class() {
        for storage in [
            IntegerStorage::I8(vec![0]),
            IntegerStorage::I16(vec![0]),
            IntegerStorage::I32(vec![0]),
            IntegerStorage::I64(vec![0]),
            IntegerStorage::U8(vec![0]),
            IntegerStorage::U16(vec![0]),
            IntegerStorage::U32(vec![0]),
            IntegerStorage::U64(vec![0]),
        ] {
            let scalar = Tensor::new_integer(storage, vec![1, 1]).expect("integer scalar");
            assert_eq!(
                parse_dbstack_args(&[Value::Tensor(scalar)]).unwrap(),
                (0, false)
            );
        }
        let scalar_double = Tensor::new(vec![0.0], vec![1, 1]).expect("double scalar");
        assert_eq!(
            parse_dbstack_args(&[Value::Tensor(scalar_double)]).unwrap(),
            (0, false)
        );
    }

    #[test]
    fn dbstack_accepts_documented_option_orders_and_rejects_duplicates() {
        assert!(parse_dbstack_args(&[Value::Num(1.0), Value::Num(2.0)]).is_err());
        assert_eq!(
            parse_dbstack_args(&[Value::Num(1.0), Value::String("-completenames".into())]).unwrap(),
            (1, true)
        );
        assert_eq!(
            parse_dbstack_args(&[Value::String("-completenames".into()), Value::Num(1.0)]).unwrap(),
            (1, true)
        );
        assert!(parse_dbstack_args(&[
            Value::String("-completenames".into()),
            Value::String("-completenames".into())
        ])
        .is_err());
        assert!(parse_dbstack_args(&[
            Value::Num(1.0),
            Value::String("-completenames".into()),
            Value::String("-completenames".into())
        ])
        .is_err());
    }

    #[test]
    fn dbstack_rejects_more_than_two_outputs() {
        let _outputs = crate::output_count::push_output_count(Some(3));
        let err = dispatch_dbstack(Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:dbstack:TooManyOutputs"));
    }

    #[test]
    fn dbstack_returns_source_backed_stack_cell() {
        let source_id = SourceId(31);
        let _catalog = crate::source_context::replace_source_catalog_with_fullpaths(vec![(
            source_id,
            "worker.m".to_string(),
            Some("/tmp/worker.m".to_string()),
            "function worker\nx = dbstack;\nend\n".to_string(),
        )]);
        let _guard = crate::debug_context::push_frame("worker", Some(source_id), Some((16, 25)));
        let value = dispatch_dbstack(Vec::new()).expect("dbstack");
        let Value::Cell(cell) = value else {
            panic!("expected cell row");
        };
        assert_eq!(cell.data.len(), 1);
        let Value::Struct(st) = &cell.data[0] else {
            panic!("expected stack struct");
        };
        assert_eq!(st.fields.get("name"), Some(&Value::String("worker".into())));
        assert_eq!(
            st.fields.get("file"),
            Some(&Value::String("worker.m".into()))
        );
        assert_eq!(cell.shape, vec![1, 1]);
        assert_eq!(st.fields.len(), 3);
        let complete =
            dispatch_dbstack(vec![Value::String("-completenames".into())]).expect("complete names");
        let Value::Cell(complete) = complete else {
            panic!("expected cell column");
        };
        let Value::Struct(complete_frame) = &complete.data[0] else {
            panic!("expected stack struct");
        };
        assert_eq!(
            complete_frame.fields.get("file"),
            Some(&Value::String("/tmp/worker.m".into()))
        );
    }

    #[test]
    fn dbstack_skips_leading_entries() {
        let _outer = crate::debug_context::push_frame("outer", None, None);
        let _inner = crate::debug_context::push_frame("inner", None, None);
        let value = dispatch_dbstack(vec![Value::Num(1.0)]).expect("dbstack");
        assert_eq!(cell_len(&value), 1);
        let Value::Cell(cell) = value else {
            unreachable!();
        };
        let Value::Struct(st) = &cell.data[0] else {
            panic!("expected stack struct");
        };
        assert_eq!(st.fields.get("name"), Some(&Value::String("outer".into())));
    }

    #[test]
    fn mlock_mislocked_munlock_current_function() {
        let _guard = crate::debug_context::push_frame("locked_fn", None, None);
        dispatch_mlock(Vec::new()).expect("mlock");
        assert_eq!(
            dispatch_mislocked(Vec::new()).expect("mislocked"),
            Value::Bool(true)
        );
        dispatch_munlock(vec![Value::String("locked_fn".into())]).expect("munlock");
        assert_eq!(
            dispatch_mislocked(vec![Value::String("locked_fn".into())]).expect("mislocked"),
            Value::Bool(false)
        );
    }

    #[test]
    fn dbstatus_returns_empty_breakpoint_cell() {
        let value = dispatch_dbstatus(Vec::new()).expect("dbstatus");
        assert_eq!(cell_len(&value), 0);
    }

    #[test]
    fn dbtype_reads_source_file_with_line_range() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("demo_dbtype.m");
        std::fs::write(&path, "a = 1;\nb = 2;\nc = 3;\n").expect("write source");
        let result = futures::executor::block_on(dispatch_dbtype(vec![
            Value::String(path.to_string_lossy().to_string()),
            Value::String("2:3".to_string()),
        ]))
        .expect("dbtype");
        assert!(matches!(result, Value::Tensor(t) if t.materialize_f64().is_empty()));
    }

    #[test]
    fn getcallinfo_reports_current_frame() {
        let _guard = crate::debug_context::push_frame("callsite", None, None);
        let value = dispatch_getcallinfo(Vec::new()).expect("getcallinfo");
        let Value::Struct(st) = value else {
            panic!("expected struct");
        };
        assert_eq!(
            st.fields.get("name"),
            Some(&Value::String("callsite".into()))
        );
        assert!(matches!(st.fields.get("stack"), Some(Value::Cell(_))));
    }
}
