//! MATLAB-compatible `pad` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, StringArray, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::builtins::strings::type_resolvers::text_preserve_type;
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::pad")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pad",
    op_kind: GpuOpKind::Custom("string-transform"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Executes on the CPU; GPU-resident inputs are gathered before padding to preserve MATLAB semantics.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::pad")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pad",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String transformation builtin; always gathers inputs and is not eligible for fusion.",
};

const BUILTIN_NAME: &str = "pad";
const PAD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Padded text preserving input container kind and shape.",
}];

const PAD_INPUTS_BASE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text (string/char/cell).",
}];

const PAD_INPUTS_LENGTH: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "len",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target length (non-negative integer).",
    },
];

const PAD_INPUTS_DIRECTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"right\""),
        description: "Padding direction (`\"left\"|\"right\"|\"both\"`).",
    },
];

const PAD_INPUTS_PADCHAR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "padCharacter",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\" \""),
        description: "Single-character padding value.",
    },
];

const PAD_INPUTS_LENGTH_DIRECTION: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "len",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target length (non-negative integer).",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"right\""),
        description: "Padding direction (`\"left\"|\"right\"|\"both\"`).",
    },
];

const PAD_INPUTS_LENGTH_PADCHAR: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "len",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target length (non-negative integer).",
    },
    BuiltinParamDescriptor {
        name: "padCharacter",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\" \""),
        description: "Single-character padding value.",
    },
];

const PAD_INPUTS_DIRECTION_PADCHAR: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"right\""),
        description: "Padding direction (`\"left\"|\"right\"|\"both\"`).",
    },
    BuiltinParamDescriptor {
        name: "padCharacter",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\" \""),
        description: "Single-character padding value.",
    },
];

const PAD_INPUTS_LENGTH_DIRECTION_PADCHAR: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "len",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target length (non-negative integer).",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"right\""),
        description: "Padding direction (`\"left\"|\"right\"|\"both\"`).",
    },
    BuiltinParamDescriptor {
        name: "padCharacter",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\" \""),
        description: "Single-character padding value.",
    },
];

const PAD_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "out = pad(str)",
        inputs: &PAD_INPUTS_BASE,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = pad(str, len)",
        inputs: &PAD_INPUTS_LENGTH,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = pad(str, direction)",
        inputs: &PAD_INPUTS_DIRECTION,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = pad(str, padCharacter)",
        inputs: &PAD_INPUTS_PADCHAR,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = pad(str, len, direction)",
        inputs: &PAD_INPUTS_LENGTH_DIRECTION,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = pad(str, len, padCharacter)",
        inputs: &PAD_INPUTS_LENGTH_PADCHAR,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = pad(str, direction, padCharacter)",
        inputs: &PAD_INPUTS_DIRECTION_PADCHAR,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = pad(str, len, direction, padCharacter)",
        inputs: &PAD_INPUTS_LENGTH_DIRECTION_PADCHAR,
        outputs: &PAD_OUTPUT,
    },
];

const PAD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.INVALID_INPUT",
    identifier: Some("RunMat:pad:InvalidInput"),
    when: "First argument is not a string array, char array, or cell array of text scalars.",
    message:
        "pad: first argument must be a string array, character array, or cell array of character vectors",
};

const PAD_ERROR_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.LENGTH",
    identifier: Some("RunMat:pad:Length"),
    when: "Length argument is not a non-negative integer scalar.",
    message: "pad: target length must be a non-negative integer scalar",
};

const PAD_ERROR_DIRECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.DIRECTION",
    identifier: Some("RunMat:pad:Direction"),
    when: "Direction argument is not one of left/right/both.",
    message: "pad: direction must be 'left', 'right', or 'both'",
};

const PAD_ERROR_PAD_CHAR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.PAD_CHAR",
    identifier: Some("RunMat:pad:PadChar"),
    when: "Padding character is not a single-character string/char scalar.",
    message:
        "pad: padding character must be a string scalar or character vector containing one character",
};

const PAD_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.CELL_ELEMENT",
    identifier: Some("RunMat:pad:CellElement"),
    when: "Cell arrays contain non-text elements or non-row char arrays.",
    message: "pad: cell array elements must be string scalars or character vectors",
};

const PAD_ERROR_ARGUMENT_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.ARGUMENT_CONFIG",
    identifier: Some("RunMat:pad:ArgumentConfig"),
    when: "Second/third arguments cannot be interpreted as valid pad argument combinations.",
    message: "pad: unable to interpret input arguments",
};

const PAD_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.ARG_COUNT",
    identifier: Some("RunMat:pad:ArgCount"),
    when: "More than four total arguments are supplied.",
    message: "pad: too many input arguments",
};

const PAD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.INTERNAL",
    identifier: Some("RunMat:pad:InternalError"),
    when: "Internal output container construction failed.",
    message: "pad: internal error",
};

const PAD_ERRORS: [BuiltinErrorDescriptor; 8] = [
    PAD_ERROR_INVALID_INPUT,
    PAD_ERROR_LENGTH,
    PAD_ERROR_DIRECTION,
    PAD_ERROR_PAD_CHAR,
    PAD_ERROR_CELL_ELEMENT,
    PAD_ERROR_ARGUMENT_CONFIG,
    PAD_ERROR_ARG_COUNT,
    PAD_ERROR_INTERNAL,
];

pub const PAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PAD_ERRORS,
};

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

fn pad_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn pad_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    pad_error_with_message(error.message, error)
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum PadDirection {
    Left,
    Right,
    Both,
}

#[derive(Clone, Copy)]
enum PadTarget {
    Auto,
    Length(usize),
}

#[derive(Clone, Copy)]
struct PadOptions {
    target: PadTarget,
    direction: PadDirection,
    pad_char: char,
}

impl Default for PadOptions {
    fn default() -> Self {
        Self {
            target: PadTarget::Auto,
            direction: PadDirection::Right,
            pad_char: ' ',
        }
    }
}

impl PadOptions {
    fn base_target(&self, auto_target: usize) -> usize {
        match self.target {
            PadTarget::Auto => auto_target,
            PadTarget::Length(len) => len,
        }
    }
}

#[runtime_builtin(
    name = "pad",
    category = "strings/transform",
    summary = "Pad strings, character arrays, and cell arrays to a target length.",
    keywords = "pad,align,strings,character array",
    accel = "sink",
    type_resolver(text_preserve_type),
    descriptor(crate::builtins::strings::transform::pad::PAD_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::pad"
)]
async fn pad_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let options = parse_arguments(&rest)?;
    let gathered = gather_if_needed_async(&value).await.map_err(map_flow)?;
    match gathered {
        Value::String(text) => pad_string(text, options),
        Value::StringArray(array) => pad_string_array(array, options),
        Value::CharArray(array) => pad_char_array(array, options),
        Value::Cell(cell) => pad_cell_array(cell, options).await,
        _ => Err(pad_error(&PAD_ERROR_INVALID_INPUT)),
    }
}

fn pad_string(text: String, options: PadOptions) -> BuiltinResult<Value> {
    if is_missing_string(&text) {
        return Ok(Value::String(text));
    }
    let char_count = string_length(&text);
    let base_target = options.base_target(char_count);
    let target_len = element_target_length(&options, base_target, char_count);
    let padded = apply_padding_owned(text, char_count, target_len, &options);
    Ok(Value::String(padded))
}

fn pad_string_array(array: StringArray, options: PadOptions) -> BuiltinResult<Value> {
    let StringArray { data, shape, .. } = array;
    let mut auto_len: usize = 0;
    if matches!(options.target, PadTarget::Auto) {
        for text in &data {
            if !is_missing_string(text) {
                auto_len = auto_len.max(string_length(text));
            }
        }
    }
    let base_target = options.base_target(auto_len);
    let mut padded: Vec<String> = Vec::with_capacity(data.len());
    for text in data.into_iter() {
        if is_missing_string(&text) {
            padded.push(text);
            continue;
        }
        let char_count = string_length(&text);
        let target_len = element_target_length(&options, base_target, char_count);
        let new_text = apply_padding_owned(text, char_count, target_len, &options);
        padded.push(new_text);
    }
    let result = StringArray::new(padded, shape)
        .map_err(|e| pad_error_with_message(format!("{BUILTIN_NAME}: {e}"), &PAD_ERROR_INTERNAL))?;
    Ok(Value::StringArray(result))
}

fn pad_char_array(array: CharArray, options: PadOptions) -> BuiltinResult<Value> {
    let CharArray { data, rows, cols } = array;
    if rows == 0 {
        return Ok(Value::CharArray(CharArray { data, rows, cols }));
    }

    let mut rows_text: Vec<String> = Vec::with_capacity(rows);
    let mut auto_len = 0usize;
    for row in 0..rows {
        let text = char_row_to_string_slice(&data, cols, row);
        auto_len = auto_len.max(string_length(&text));
        rows_text.push(text);
    }

    let base_target = options.base_target(auto_len);
    let mut padded_rows: Vec<String> = Vec::with_capacity(rows);
    let mut final_cols: usize = 0;
    for row_text in rows_text.into_iter() {
        let char_count = string_length(&row_text);
        let target_len = element_target_length(&options, base_target, char_count);
        let padded = apply_padding_owned(row_text, char_count, target_len, &options);
        final_cols = final_cols.max(string_length(&padded));
        padded_rows.push(padded);
    }

    let mut new_data: Vec<char> = Vec::with_capacity(rows * final_cols);
    for row_text in padded_rows.into_iter() {
        let mut chars: Vec<char> = row_text.chars().collect();
        if chars.len() < final_cols {
            chars.resize(final_cols, ' ');
        }
        new_data.extend(chars.into_iter());
    }

    CharArray::new(new_data, rows, final_cols)
        .map(Value::CharArray)
        .map_err(|e| pad_error_with_message(format!("{BUILTIN_NAME}: {e}"), &PAD_ERROR_INTERNAL))
}

async fn pad_cell_array(cell: CellArray, options: PadOptions) -> BuiltinResult<Value> {
    let rows = cell.rows;
    let cols = cell.cols;
    let total = rows * cols;
    let mut items: Vec<CellItem> = Vec::with_capacity(total);
    let mut auto_len = 0usize;

    for idx in 0..total {
        let value = &cell.data[idx];
        let gathered = gather_if_needed_async(value).await.map_err(map_flow)?;
        let item = match gathered {
            Value::String(text) => {
                let is_missing = is_missing_string(&text);
                let len = if is_missing { 0 } else { string_length(&text) };
                if !is_missing {
                    auto_len = auto_len.max(len);
                }
                CellItem {
                    kind: CellKind::String,
                    text,
                    char_count: len,
                    is_missing,
                }
            }
            Value::StringArray(sa) if sa.data.len() == 1 => {
                let text = sa.data.into_iter().next().unwrap_or_default();
                let is_missing = is_missing_string(&text);
                let len = if is_missing { 0 } else { string_length(&text) };
                if !is_missing {
                    auto_len = auto_len.max(len);
                }
                CellItem {
                    kind: CellKind::String,
                    text,
                    char_count: len,
                    is_missing,
                }
            }
            Value::CharArray(ca) if ca.rows <= 1 => {
                let text = if ca.rows == 0 {
                    String::new()
                } else {
                    char_row_to_string_slice(&ca.data, ca.cols, 0)
                };
                let len = string_length(&text);
                auto_len = auto_len.max(len);
                CellItem {
                    kind: CellKind::Char { rows: ca.rows },
                    text,
                    char_count: len,
                    is_missing: false,
                }
            }
            Value::CharArray(_) => return Err(pad_error(&PAD_ERROR_CELL_ELEMENT)),
            _ => return Err(pad_error(&PAD_ERROR_CELL_ELEMENT)),
        };
        items.push(item);
    }

    let base_target = options.base_target(auto_len);
    let mut results: Vec<Value> = Vec::with_capacity(total);
    for item in items.into_iter() {
        if item.is_missing {
            results.push(Value::String(item.text));
            continue;
        }
        let target_len = element_target_length(&options, base_target, item.char_count);
        let padded = apply_padding_owned(item.text, item.char_count, target_len, &options);
        match item.kind {
            CellKind::String => results.push(Value::String(padded)),
            CellKind::Char { rows } => {
                let chars: Vec<char> = padded.chars().collect();
                let cols = chars.len();
                let array = CharArray::new(chars, rows, cols).map_err(|e| {
                    pad_error_with_message(format!("{BUILTIN_NAME}: {e}"), &PAD_ERROR_INTERNAL)
                })?;
                results.push(Value::CharArray(array));
            }
        }
    }

    make_cell(results, rows, cols)
        .map_err(|e| pad_error_with_message(format!("{BUILTIN_NAME}: {e}"), &PAD_ERROR_INTERNAL))
}

#[derive(Clone)]
struct CellItem {
    kind: CellKind,
    text: String,
    char_count: usize,
    is_missing: bool,
}

#[derive(Clone)]
enum CellKind {
    String,
    Char { rows: usize },
}

fn parse_arguments(args: &[Value]) -> BuiltinResult<PadOptions> {
    let mut options = PadOptions::default();
    match args.len() {
        0 => Ok(options),
        1 => {
            if let Some(length) = parse_length(&args[0])? {
                options.target = PadTarget::Length(length);
                return Ok(options);
            }
            if let Some(direction) = try_parse_direction(&args[0], false)? {
                options.direction = direction;
                return Ok(options);
            }
            let pad_char = parse_pad_char(&args[0])?;
            options.pad_char = pad_char;
            Ok(options)
        }
        2 => {
            if let Some(length) = parse_length(&args[0])? {
                options.target = PadTarget::Length(length);
                if let Some(direction) = try_parse_direction(&args[1], false)? {
                    options.direction = direction;
                } else {
                    match parse_pad_char(&args[1]) {
                        Ok(pad_char) => options.pad_char = pad_char,
                        Err(_) => return Err(pad_error(&PAD_ERROR_DIRECTION)),
                    }
                }
                Ok(options)
            } else if let Some(direction) = try_parse_direction(&args[0], false)? {
                options.direction = direction;
                let pad_char = parse_pad_char(&args[1])?;
                options.pad_char = pad_char;
                Ok(options)
            } else {
                Err(pad_error(&PAD_ERROR_ARGUMENT_CONFIG))
            }
        }
        3 => {
            let length = parse_length(&args[0])?.ok_or_else(|| pad_error(&PAD_ERROR_LENGTH))?;
            let direction = try_parse_direction(&args[1], true)?
                .ok_or_else(|| pad_error(&PAD_ERROR_DIRECTION))?;
            let pad_char = parse_pad_char(&args[2])?;
            options.target = PadTarget::Length(length);
            options.direction = direction;
            options.pad_char = pad_char;
            Ok(options)
        }
        _ => Err(pad_error(&PAD_ERROR_ARG_COUNT)),
    }
}

fn parse_length(value: &Value) -> BuiltinResult<Option<usize>> {
    match value {
        Value::Num(n) => {
            if !n.is_finite() || *n < 0.0 {
                return Err(pad_error(&PAD_ERROR_LENGTH));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(pad_error(&PAD_ERROR_LENGTH));
            }
            Ok(Some(*n as usize))
        }
        Value::Int(i) => {
            let val = i.to_i64();
            if val < 0 {
                return Err(pad_error(&PAD_ERROR_LENGTH));
            }
            Ok(Some(val as usize))
        }
        _ => Ok(None),
    }
}

fn try_parse_direction(value: &Value, strict: bool) -> BuiltinResult<Option<PadDirection>> {
    let Some(text) = value_to_single_string(value) else {
        return if strict {
            Err(pad_error(&PAD_ERROR_DIRECTION))
        } else {
            Ok(None)
        };
    };
    let lowered = text.trim().to_ascii_lowercase();
    if lowered.is_empty() {
        return if strict {
            Err(pad_error(&PAD_ERROR_DIRECTION))
        } else {
            Ok(None)
        };
    }
    let direction = match lowered.as_str() {
        "left" => PadDirection::Left,
        "right" => PadDirection::Right,
        "both" => PadDirection::Both,
        _ => {
            return if strict {
                Err(pad_error(&PAD_ERROR_DIRECTION))
            } else {
                Ok(None)
            };
        }
    };
    Ok(Some(direction))
}

fn parse_pad_char(value: &Value) -> BuiltinResult<char> {
    let text = value_to_single_string(value).ok_or_else(|| pad_error(&PAD_ERROR_PAD_CHAR))?;
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return Err(pad_error(&PAD_ERROR_PAD_CHAR));
    };
    if chars.next().is_some() {
        return Err(pad_error(&PAD_ERROR_PAD_CHAR));
    }
    Ok(first)
}

fn value_to_single_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Some(sa.data[0].clone())
            } else {
                None
            }
        }
        Value::CharArray(ca) if ca.rows <= 1 => {
            if ca.rows == 0 {
                Some(String::new())
            } else {
                Some(char_row_to_string_slice(&ca.data, ca.cols, 0))
            }
        }
        _ => None,
    }
}

fn string_length(text: &str) -> usize {
    text.chars().count()
}

fn element_target_length(options: &PadOptions, base_target: usize, current_len: usize) -> usize {
    match options.target {
        PadTarget::Auto => base_target.max(current_len),
        PadTarget::Length(_) => base_target.max(current_len),
    }
}

fn apply_padding_owned(
    text: String,
    current_len: usize,
    target_len: usize,
    options: &PadOptions,
) -> String {
    if current_len >= target_len {
        return text;
    }
    let delta = target_len - current_len;
    let (left_pad, right_pad) = match options.direction {
        PadDirection::Left => (delta, 0),
        PadDirection::Right => (0, delta),
        PadDirection::Both => {
            let left = delta / 2;
            (left, delta - left)
        }
    };
    let mut result = String::with_capacity(text.len() + delta * options.pad_char.len_utf8());
    for _ in 0..left_pad {
        result.push(options.pad_char);
    }
    result.push_str(&text);
    for _ in 0..right_pad {
        result.push(options.pad_char);
    }
    result
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::builtins::common::test_support;
    use runmat_builtins::{ResolveContext, Type};

    fn pad_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::pad_builtin(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_length_right() {
        let result = pad_builtin(Value::String("GPU".into()), vec![Value::Num(5.0)]).expect("pad");
        assert_eq!(result, Value::String("GPU  ".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_left_with_custom_char() {
        let result = pad_builtin(
            Value::String("42".into()),
            vec![
                Value::Num(4.0),
                Value::String("left".into()),
                Value::String("0".into()),
            ],
        )
        .expect("pad");
        assert_eq!(result, Value::String("0042".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_both_with_odd_count() {
        let result = pad_builtin(
            Value::String("core".into()),
            vec![
                Value::Num(9.0),
                Value::String("both".into()),
                Value::String("*".into()),
            ],
        )
        .expect("pad");
        assert_eq!(result, Value::String("**core***".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_array_auto_uses_longest_element() {
        let strings =
            StringArray::new(vec!["GPU".into(), "Accelerate".into()], vec![2, 1]).unwrap();
        let result = pad_builtin(Value::StringArray(strings), Vec::new()).expect("pad");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data[0], "GPU       ");
                assert_eq!(sa.data[1], "Accelerate");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_array_pad_character_only() {
        let strings = StringArray::new(vec!["A".into(), "Run".into()], vec![2, 1]).unwrap();
        let result =
            pad_builtin(Value::StringArray(strings), vec![Value::String("*".into())]).expect("pad");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data[0], "A**");
                assert_eq!(sa.data[1], "Run");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_array_length_with_pad_character() {
        let strings = StringArray::new(vec!["7".into(), "512".into()], vec![2, 1]).unwrap();
        let result = pad_builtin(
            Value::StringArray(strings),
            vec![Value::Num(4.0), Value::String("0".into())],
        )
        .expect("pad");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data[0], "7000");
                assert_eq!(sa.data[1], "5120");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_string_array_direction_only() {
        let strings =
            StringArray::new(vec!["Mary".into(), "Elizabeth".into()], vec![2, 1]).unwrap();
        let result = pad_builtin(
            Value::StringArray(strings),
            vec![Value::String("left".into())],
        )
        .expect("pad");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data[0], "     Mary");
                assert_eq!(sa.data[1], "Elizabeth");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_single_string_pad_character_only_leaves_length() {
        let result =
            pad_builtin(Value::String("GPU".into()), vec![Value::String("-".into())]).expect("pad");
        assert_eq!(result, Value::String("GPU".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_char_array_resizes_columns() {
        let chars: Vec<char> = "GPUrun".chars().collect();
        let array = CharArray::new(chars, 2, 3).unwrap();
        let result = pad_builtin(Value::CharArray(array), vec![Value::Num(5.0)]).expect("pad");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 5);
                let expected: Vec<char> = "GPU  run  ".chars().collect();
                assert_eq!(ca.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_cell_array_mixed_content() {
        let cell = CellArray::new(
            vec![
                Value::String("solver".into()),
                Value::CharArray(CharArray::new_row("jit")),
                Value::String("planner".into()),
            ],
            1,
            3,
        )
        .unwrap();
        let result = pad_builtin(
            Value::Cell(cell),
            vec![Value::String("right".into()), Value::String(".".into())],
        )
        .expect("pad");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 3);
                assert_eq!(out.get(0, 0).unwrap(), Value::String("solver.".into()));
                assert_eq!(
                    out.get(0, 1).unwrap(),
                    Value::CharArray(CharArray::new_row("jit...."))
                );
                assert_eq!(out.get(0, 2).unwrap(), Value::String("planner".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_preserves_missing_string() {
        let result =
            pad_builtin(Value::String("<missing>".into()), vec![Value::Num(8.0)]).expect("pad");
        assert_eq!(result, Value::String("<missing>".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_errors_on_invalid_input_type() {
        let err = pad_builtin(Value::Num(1.0), Vec::new()).unwrap_err();
        assert_eq!(err.to_string(), PAD_ERROR_INVALID_INPUT.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_errors_on_negative_length() {
        let err = pad_builtin(Value::String("data".into()), vec![Value::Num(-1.0)]).unwrap_err();
        assert_eq!(err.to_string(), PAD_ERROR_LENGTH.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_errors_on_invalid_direction() {
        let err = pad_builtin(
            Value::String("data".into()),
            vec![Value::Num(6.0), Value::String("around".into())],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), PAD_ERROR_DIRECTION.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pad_errors_on_invalid_pad_character() {
        let err = pad_builtin(
            Value::String("data".into()),
            vec![Value::String("left".into()), Value::String("##".into())],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), PAD_ERROR_PAD_CHAR.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn pad_works_with_wgpu_provider_active() {
        test_support::with_test_provider(|_| {
            let result =
                pad_builtin(Value::String("GPU".into()), vec![Value::Num(6.0)]).expect("pad");
            assert_eq!(result, Value::String("GPU   ".into()));
        });
    }

    #[test]
    fn pad_type_preserves_text() {
        assert_eq!(
            text_preserve_type(&[Type::String], &ResolveContext::new(Vec::new())),
            Type::String
        );
    }
}
