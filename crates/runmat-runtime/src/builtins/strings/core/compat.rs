//! MATLAB text compatibility helpers that do not warrant larger domain modules yet.

use encoding_rs::{Encoding, UTF_8};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, IntValue, LogicalArray, NumericScalar, ObjectInstance, ResolveContext, StringArray,
    Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast as matlab_broadcast;
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::{build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult};

const PATTERN_CLASS: &str = "pattern";

const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output value.",
}];

const OUT_BOOL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical result.",
}];

const IN_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value.",
}];

const IN_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text.",
}];

const IN_BOUNDARY_TYPE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "type",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"either\""),
    description: "Boundary type: \"either\", \"start\", or \"end\".",
}];

const IN_TEXT_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "text",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional arguments.",
    },
];

const IN_A_B_N: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First text input.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second text input.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of leading characters to compare.",
    },
];

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const NO_ERRORS: [BuiltinErrorDescriptor; 0] = [];

macro_rules! descriptor {
    ($name:ident, $label:expr, $inputs:expr, $outputs:expr) => {
        const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[BuiltinSignatureDescriptor {
                label: $label,
                inputs: $inputs,
                outputs: $outputs,
            }],
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &NO_ERRORS,
        };
    };
}

macro_rules! descriptor_by_outputs {
    ($name:ident, $label:expr, $inputs:expr, $outputs:expr) => {
        const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[BuiltinSignatureDescriptor {
                label: $label,
                inputs: $inputs,
                outputs: $outputs,
            }],
            output_mode: BuiltinOutputMode::ByRequestedOutputCount,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &NO_ERRORS,
        };
    };
}

descriptor!(NEWLINE_DESCRIPTOR, "s = newline", &NO_INPUTS, &OUT_ANY);
descriptor!(BLANKS_DESCRIPTOR, "s = blanks(n)", &IN_VALUE, &OUT_ANY);
descriptor!(
    IS_STRING_SCALAR_DESCRIPTOR,
    "tf = isStringScalar(value)",
    &IN_VALUE,
    &OUT_BOOL
);
descriptor_by_outputs!(
    CONVERT_STRINGS_TO_CHARS_DESCRIPTOR,
    "[out1, ...] = convertStringsToChars(value1, ...)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    CONVERT_CHARS_TO_STRINGS_DESCRIPTOR,
    "out = convertCharsToStrings(value)",
    &IN_VALUE,
    &OUT_ANY
);
descriptor!(
    CONVERT_CONTAINED_STRINGS_TO_CHARS_DESCRIPTOR,
    "out = convertContainedStringsToChars(value)",
    &IN_VALUE,
    &OUT_ANY
);
descriptor!(
    STRNCMPI_DESCRIPTOR,
    "tf = strncmpi(A, B, N)",
    &IN_A_B_N,
    &OUT_BOOL
);
descriptor!(
    ISSTRPROP_DESCRIPTOR,
    "tf = isstrprop(text, category)",
    &IN_TEXT_REST,
    &OUT_BOOL
);
descriptor!(
    ISLETTER_DESCRIPTOR,
    "tf = isletter(text)",
    &IN_TEXT,
    &OUT_BOOL
);
descriptor!(
    ISSPACE_DESCRIPTOR,
    "tf = isspace(text)",
    &IN_TEXT,
    &OUT_BOOL
);
descriptor_by_outputs!(
    STRTOK_DESCRIPTOR,
    "[tok, rem] = strtok(text, delimiters)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor_by_outputs!(
    STR2NUM_DESCRIPTOR,
    "[x, tf] = str2num(text)",
    &IN_TEXT,
    &OUT_ANY
);
descriptor!(
    MAT2STR_DESCRIPTOR,
    "s = mat2str(A)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    NATIVE2UNICODE_DESCRIPTOR,
    "s = native2unicode(bytes, encoding)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor_by_outputs!(
    SSCANF_DESCRIPTOR,
    "[A, count, errmsg, nextindex] = sscanf(text, format, size)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    PATTERN_DESCRIPTOR,
    "pat = pattern(text)",
    &IN_TEXT,
    &OUT_ANY
);
descriptor!(
    REGEXP_PATTERN_DESCRIPTOR,
    "pat = regexpPattern(expr)",
    &IN_TEXT,
    &OUT_ANY
);
descriptor!(
    DIGITS_PATTERN_DESCRIPTOR,
    "pat = digitsPattern(N)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    LETTERS_PATTERN_DESCRIPTOR,
    "pat = lettersPattern(N)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    WILDCARD_PATTERN_DESCRIPTOR,
    "pat = wildcardPattern",
    &IN_TEXT_REST,
    &OUT_ANY
);
const TEXT_BOUNDARY_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "pat = textBoundary",
        inputs: &[],
        outputs: &OUT_ANY,
    },
    BuiltinSignatureDescriptor {
        label: "pat = textBoundary(type)",
        inputs: &IN_BOUNDARY_TYPE,
        outputs: &OUT_ANY,
    },
];
pub const TEXT_BOUNDARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TEXT_BOUNDARY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NO_ERRORS,
};

fn any_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

fn string_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::String
}

fn bool_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

fn tensor_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

fn compat_error(name: &str, message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

fn map_flow(name: &'static str) -> impl Fn(crate::RuntimeError) -> crate::RuntimeError {
    move |err| map_control_flow_with_builtin(err, name)
}

#[runtime_builtin(
    name = "newline",
    category = "strings/core",
    summary = "Return a newline string scalar.",
    keywords = "newline,string,text,line break",
    accel = "metadata",
    type_resolver(string_type),
    descriptor(crate::builtins::strings::core::compat::NEWLINE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
fn newline_builtin() -> BuiltinResult<Value> {
    Ok(Value::String("\n".to_string()))
}

#[runtime_builtin(
    name = "blanks",
    category = "strings/core",
    summary = "Return a character row vector of spaces.",
    keywords = "blanks,char,space,text",
    accel = "metadata",
    type_resolver(string_type),
    descriptor(crate::builtins::strings::core::compat::BLANKS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn blanks_builtin(n: Value) -> BuiltinResult<Value> {
    let n = gather_if_needed_async(&n)
        .await
        .map_err(map_flow("blanks"))?;
    let n = parse_nonnegative_usize(&n, "blanks")?;
    Ok(Value::CharArray(CharArray::new_row(&" ".repeat(n))))
}

#[runtime_builtin(
    name = "isStringScalar",
    category = "strings/core",
    summary = "Return true for a scalar MATLAB string.",
    keywords = "isStringScalar,string scalar,type predicate",
    accel = "metadata",
    type_resolver(bool_type),
    descriptor(crate::builtins::strings::core::compat::IS_STRING_SCALAR_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
fn is_string_scalar_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(match value {
        Value::String(_) => true,
        Value::StringArray(array) => array.data.len() == 1,
        _ => false,
    }))
}

#[runtime_builtin(
    name = "convertStringsToChars",
    category = "strings/core",
    summary = "Convert string scalars and arrays to character vectors.",
    keywords = "convertStringsToChars,string,char,compatibility",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::core::compat::CONVERT_STRINGS_TO_CHARS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn convert_strings_to_chars_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let mut inputs = Vec::with_capacity(rest.len() + 1);
    inputs.push(value);
    inputs.extend(rest);

    let mut outputs = Vec::with_capacity(inputs.len());
    for value in inputs {
        let value = gather_if_needed_async(&value)
            .await
            .map_err(map_flow("convertStringsToChars"))?;
        outputs.push(convert_strings_to_chars(value, false)?);
    }

    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(n) => Ok(crate::output_count::output_list_with_padding(n, outputs)),
        None => Ok(outputs
            .into_iter()
            .next()
            .unwrap_or(Value::String(String::new()))),
    }
}

#[runtime_builtin(
    name = "convertCharsToStrings",
    category = "strings/core",
    summary = "Convert character arrays and cellstr values to string arrays.",
    keywords = "convertCharsToStrings,char,string,compatibility",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::core::compat::CONVERT_CHARS_TO_STRINGS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn convert_chars_to_strings_builtin(value: Value) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_flow("convertCharsToStrings"))?;
    convert_chars_to_strings(value)
}

#[runtime_builtin(
    name = "convertContainedStringsToChars",
    category = "strings/core",
    summary = "Convert string values contained in cells and structs to character vectors.",
    keywords = "convertContainedStringsToChars,string,char,cell,struct",
    accel = "sink",
    type_resolver(any_type),
    descriptor(
        crate::builtins::strings::core::compat::CONVERT_CONTAINED_STRINGS_TO_CHARS_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn convert_contained_strings_to_chars_builtin(value: Value) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_flow("convertContainedStringsToChars"))?;
    convert_strings_to_chars(value, true)
}

#[runtime_builtin(
    name = "strncmpi",
    category = "strings/core",
    summary = "Compare text inputs case-insensitively up to N leading characters.",
    keywords = "strncmpi,string compare,prefix,text equality",
    accel = "sink",
    type_resolver(bool_type),
    descriptor(crate::builtins::strings::core::compat::STRNCMPI_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn strncmpi_builtin(a: Value, b: Value, n: Value) -> BuiltinResult<Value> {
    let a = gather_if_needed_async(&a)
        .await
        .map_err(map_flow("strncmpi"))?;
    let b = gather_if_needed_async(&b)
        .await
        .map_err(map_flow("strncmpi"))?;
    let n = gather_if_needed_async(&n)
        .await
        .map_err(map_flow("strncmpi"))?;
    let n = parse_nonnegative_usize(&n, "strncmpi")?;
    let left = TextList::from_value(a, "strncmpi")?;
    let right = TextList::from_value(b, "strncmpi")?;
    let shape = broadcast_shape(&left.shape, &right.shape, "strncmpi")?;
    let total: usize = shape.iter().product();
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let li = broadcast_flat_index(idx, &shape, &left.shape);
        let ri = broadcast_flat_index(idx, &shape, &right.shape);
        let matched = match (&left.items[li], &right.items[ri]) {
            (Some(a), Some(b)) => prefix_eq_ignore_case(a, b, n),
            _ => false,
        };
        out.push(u8::from(matched));
    }
    logical_value(out, shape, "strncmpi")
}

#[runtime_builtin(
    name = "isstrprop",
    category = "strings/core",
    summary = "Classify characters in text by character property.",
    keywords = "isstrprop,isletter,isspace,char classification,text",
    accel = "sink",
    type_resolver(tensor_type),
    descriptor(crate::builtins::strings::core::compat::ISSTRPROP_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn isstrprop_builtin(text: Value, prop: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("isstrprop"))?;
    let prop = gather_if_needed_async(&prop)
        .await
        .map_err(map_flow("isstrprop"))?;
    let prop = scalar_text(&prop, "isstrprop")?.to_ascii_lowercase();
    classify_text_value(text, "isstrprop", |ch| char_matches_prop(ch, &prop))
}

#[runtime_builtin(
    name = "isletter",
    category = "strings/core",
    summary = "Return true for letters in text.",
    keywords = "isletter,letter,char classification,text",
    accel = "sink",
    type_resolver(tensor_type),
    descriptor(crate::builtins::strings::core::compat::ISLETTER_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn isletter_builtin(text: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("isletter"))?;
    classify_text_value(text, "isletter", |ch| ch.is_alphabetic())
}

#[runtime_builtin(
    name = "isspace",
    category = "strings/core",
    summary = "Return true for whitespace characters in text.",
    keywords = "isspace,whitespace,char classification,text",
    accel = "sink",
    type_resolver(tensor_type),
    descriptor(crate::builtins::strings::core::compat::ISSPACE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn isspace_builtin(text: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("isspace"))?;
    classify_text_value(text, "isspace", |ch| ch.is_whitespace())
}

#[runtime_builtin(
    name = "strtok",
    category = "strings/core",
    summary = "Return the first token from text using delimiter characters.",
    keywords = "strtok,tokenize,delimiter,text",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::core::compat::STRTOK_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn strtok_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("strtok"))?;
    let delimiters = if let Some(value) = rest.first() {
        let value = gather_if_needed_async(value)
            .await
            .map_err(map_flow("strtok"))?;
        scalar_text(&value, "strtok")?
    } else {
        " \t\n\r".to_string()
    };
    let (tokens, remainders) =
        map_text_pair_preserve(text, "strtok", |s| strtok_pair(s, &delimiters))?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![tokens])),
        Some(n) => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![tokens, remainders],
        )),
        None => Ok(tokens),
    }
}

#[runtime_builtin(
    name = "str2num",
    category = "strings/core",
    summary = "Convert text containing numeric literals to a numeric array.",
    keywords = "str2num,string numeric conversion,text",
    accel = "sink",
    type_resolver(tensor_type),
    descriptor(crate::builtins::strings::core::compat::STR2NUM_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn str2num_builtin(text: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("str2num"))?;
    let text = scalar_text(&text, "str2num")?;
    let (value, ok) = parse_str2num_matrix(&text);
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![value])),
        Some(n) => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![value, Value::Bool(ok)],
        )),
        None => Ok(value),
    }
}

#[runtime_builtin(
    name = "mat2str",
    category = "strings/core",
    summary = "Convert numeric, logical, character, and string values to MATLAB expression text.",
    keywords = "mat2str,array string conversion,text",
    accel = "sink",
    type_resolver(string_type),
    descriptor(crate::builtins::strings::core::compat::MAT2STR_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn mat2str_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_flow("mat2str"))?;
    let precision = if let Some(arg) = rest.first() {
        let arg = gather_if_needed_async(arg)
            .await
            .map_err(map_flow("mat2str"))?;
        Some(parse_nonnegative_usize(&arg, "mat2str")?)
    } else {
        None
    };
    Ok(Value::String(mat2str_value(&value, precision)))
}

#[runtime_builtin(
    name = "native2unicode",
    category = "strings/core",
    summary = "Decode native byte values into Unicode text.",
    keywords = "native2unicode,unicode,encoding,text,uint8",
    accel = "sink",
    type_resolver(string_type),
    descriptor(crate::builtins::strings::core::compat::NATIVE2UNICODE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn native2unicode_builtin(bytes: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let bytes = gather_if_needed_async(&bytes)
        .await
        .map_err(map_flow("native2unicode"))?;
    let encoding = if let Some(value) = rest.first() {
        let value = gather_if_needed_async(value)
            .await
            .map_err(map_flow("native2unicode"))?;
        scalar_text(&value, "native2unicode")?
    } else {
        "UTF-8".to_string()
    };
    let bytes = bytes_from_value(&bytes, "native2unicode")?;
    decode_bytes(&bytes, &encoding)
}

#[runtime_builtin(
    name = "sscanf",
    category = "strings/core",
    summary = "Parse formatted numeric values from text.",
    keywords = "sscanf,scan,format,text,numeric",
    accel = "sink",
    type_resolver(tensor_type),
    descriptor(crate::builtins::strings::core::compat::SSCANF_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn sscanf_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("sscanf"))?;
    let text = scalar_text(&text, "sscanf")?;
    let format = if let Some(fmt) = rest.first() {
        let fmt = gather_if_needed_async(fmt)
            .await
            .map_err(map_flow("sscanf"))?;
        scalar_text(&fmt, "sscanf")?
    } else {
        "%f".to_string()
    };
    let size = if let Some(size) = rest.get(1) {
        let size = gather_if_needed_async(size)
            .await
            .map_err(map_flow("sscanf"))?;
        Some(scan_size_from_value(&size)?)
    } else {
        None
    };
    let scan = sscanf_scan(&text, &format, size)?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![scan.value])),
        Some(n) => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![
                scan.value,
                Value::Num(scan.count as f64),
                Value::String(scan.errmsg),
                Value::Num(scan.next_index as f64),
            ],
        )),
        None => Ok(scan.value),
    }
}

#[runtime_builtin(
    name = "pattern",
    category = "strings/pattern",
    summary = "Create a literal string pattern.",
    keywords = "pattern,string pattern,text",
    accel = "metadata",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::core::compat::PATTERN_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn pattern_builtin(text: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("pattern"))?;
    Ok(pattern_object(&regex::escape(&scalar_text(
        &text, "pattern",
    )?)))
}

#[runtime_builtin(
    name = "regexpPattern",
    category = "strings/pattern",
    summary = "Create a regular expression string pattern.",
    keywords = "regexpPattern,pattern,regular expression,text",
    accel = "metadata",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::core::compat::REGEXP_PATTERN_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn regexp_pattern_builtin(text: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("regexpPattern"))?;
    Ok(pattern_object(&scalar_text(&text, "regexpPattern")?))
}

#[runtime_builtin(
    name = "digitsPattern",
    category = "strings/pattern",
    summary = "Create a pattern matching digit characters.",
    keywords = "digitsPattern,pattern,digits,text",
    accel = "metadata",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::core::compat::DIGITS_PATTERN_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn digits_pattern_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    bounded_pattern(rest, "\\d", "digitsPattern").await
}

#[runtime_builtin(
    name = "lettersPattern",
    category = "strings/pattern",
    summary = "Create a pattern matching letter characters.",
    keywords = "lettersPattern,pattern,letters,text",
    accel = "metadata",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::core::compat::LETTERS_PATTERN_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn letters_pattern_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    bounded_pattern(rest, r"\p{Alphabetic}", "lettersPattern").await
}

#[runtime_builtin(
    name = "wildcardPattern",
    category = "strings/pattern",
    summary = "Create a pattern matching arbitrary text.",
    keywords = "wildcardPattern,pattern,wildcard,text",
    accel = "metadata",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::core::compat::WILDCARD_PATTERN_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn wildcard_pattern_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.is_empty() {
        return Ok(pattern_object(".*"));
    }
    bounded_pattern(rest, ".", "wildcardPattern").await
}

#[runtime_builtin(
    name = "textBoundary",
    category = "strings/pattern",
    summary = "Create a pattern matching the start or end of text.",
    keywords = "textBoundary,pattern,boundary,start,end,text",
    accel = "metadata",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::core::compat::TEXT_BOUNDARY_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn text_boundary_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    let boundary_type = match rest.as_slice() {
        [] => "either".to_string(),
        [value] => {
            let value = gather_if_needed_async(value)
                .await
                .map_err(map_flow("textBoundary"))?;
            scalar_text(&value, "textBoundary")?
        }
        _ => {
            return Err(compat_error(
                "textBoundary",
                "textBoundary: expected zero inputs or one boundary type",
            ))
        }
    };
    let regex = match boundary_type.to_ascii_lowercase().as_str() {
        "either" => r"(?:^|$)",
        "start" => r"^",
        "end" => r"$",
        other => {
            return Err(compat_error(
                "textBoundary",
                format!("textBoundary: unsupported boundary type '{other}'"),
            ))
        }
    };
    Ok(pattern_object(regex))
}

pub(crate) fn scalar_text(value: &Value, fn_name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 0 => Ok(String::new()),
        Value::CharArray(array) if array.rows == 1 => {
            Ok(char_row_to_string_slice(&array.data, array.cols, 0))
        }
        other => Err(compat_error(
            fn_name,
            format!("{fn_name}: expected a text scalar, got {other:?}"),
        )),
    }
}

pub(crate) fn pattern_regex(value: &Value, fn_name: &str) -> BuiltinResult<String> {
    match value {
        Value::Object(object) if object.is_class(PATTERN_CLASS) => {
            match object.properties.get("Regex") {
                Some(Value::String(regex)) => Ok(regex.clone()),
                _ => Err(compat_error(
                    fn_name,
                    format!("{fn_name}: invalid pattern object"),
                )),
            }
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Ok(regex::escape(&scalar_text(value, fn_name)?))
        }
        other => Err(compat_error(
            fn_name,
            format!("{fn_name}: expected text or pattern, got {other:?}"),
        )),
    }
}

pub(crate) fn pattern_object(regex: &str) -> Value {
    let mut object = ObjectInstance::new(PATTERN_CLASS.to_string());
    object
        .properties
        .insert("Regex".to_string(), Value::String(regex.to_string()));
    Value::Object(object)
}

pub(crate) fn text_items(value: Value, fn_name: &str) -> BuiltinResult<TextList> {
    TextList::from_value(value, fn_name)
}

pub(crate) fn logical_value(
    data: Vec<u8>,
    shape: Vec<usize>,
    fn_name: &str,
) -> BuiltinResult<Value> {
    if data.len() == 1 {
        Ok(Value::Bool(data[0] != 0))
    } else {
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(|e| compat_error(fn_name, format!("{fn_name}: {e}")))
    }
}

pub(crate) struct TextList {
    pub(crate) items: Vec<Option<String>>,
    pub(crate) shape: Vec<usize>,
}

impl TextList {
    fn from_value(value: Value, fn_name: &str) -> BuiltinResult<Self> {
        match value {
            Value::String(text) => Ok(Self {
                items: vec![missing_to_none(text)],
                shape: vec![1, 1],
            }),
            Value::StringArray(array) => Ok(Self {
                items: array.data.into_iter().map(missing_to_none).collect(),
                shape: array.shape,
            }),
            Value::CharArray(array) => {
                let mut items = Vec::with_capacity(array.rows.max(1));
                if array.rows == 0 {
                    return Ok(Self {
                        items,
                        shape: vec![0, 1],
                    });
                }
                for row in 0..array.rows {
                    items.push(Some(char_row_to_string_slice(&array.data, array.cols, row)));
                }
                Ok(Self {
                    items,
                    shape: vec![array.rows, 1],
                })
            }
            Value::Cell(cell) => {
                let mut items = Vec::with_capacity(cell.data.len());
                for value in cell.data {
                    items.push(Some(scalar_text(&value, fn_name)?));
                }
                Ok(Self {
                    items,
                    shape: cell.shape,
                })
            }
            other => Err(compat_error(
                fn_name,
                format!("{fn_name}: expected text input, got {other:?}"),
            )),
        }
    }
}

pub(crate) fn broadcast_shape(
    a: &[usize],
    b: &[usize],
    fn_name: &str,
) -> BuiltinResult<Vec<usize>> {
    matlab_broadcast::broadcast_shapes(fn_name, a, b).map_err(|err| compat_error(fn_name, err))
}

pub(crate) fn broadcast_flat_index(
    linear: usize,
    shape: &[usize],
    source_shape: &[usize],
) -> usize {
    if source_shape.iter().product::<usize>() <= 1 {
        return 0;
    }
    let mut extended = Vec::with_capacity(shape.len());
    extended.extend(std::iter::repeat_n(
        1,
        shape.len().saturating_sub(source_shape.len()),
    ));
    extended.extend_from_slice(source_shape);
    let strides = matlab_broadcast::compute_strides(&extended);
    matlab_broadcast::broadcast_index(linear, shape, &extended, &strides)
}

fn missing_to_none(text: String) -> Option<String> {
    if is_missing_string(&text) {
        None
    } else {
        Some(text)
    }
}

fn parse_nonnegative_usize(value: &Value, fn_name: &str) -> BuiltinResult<usize> {
    let index = match value {
        Value::Num(n) => nonnegative_platform_usize(*n),
        Value::Int(i) => i.try_to_usize(),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                storage.value_at(0).and_then(|value| value.try_to_usize())
            } else {
                nonnegative_platform_usize(tensor_utils::tensor_value_f64(tensor, 0))
            }
        }
        _ => {
            return Err(compat_error(
                fn_name,
                format!("{fn_name}: expected a nonnegative integer scalar"),
            ))
        }
    };
    index.ok_or_else(|| {
        compat_error(
            fn_name,
            format!("{fn_name}: expected a nonnegative integer scalar"),
        )
    })
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

fn convert_strings_to_chars(value: Value, contained_only: bool) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => Ok(Value::CharArray(CharArray::new_row(&text))),
        Value::StringArray(array) if array.data.len() == 1 && !contained_only => {
            Ok(Value::CharArray(CharArray::new_row(&array.data[0])))
        }
        Value::StringArray(array) if !contained_only => {
            let values = array
                .data
                .into_iter()
                .map(|text| Value::CharArray(CharArray::new_row(&text)))
                .collect();
            make_cell_with_shape(values, array.shape)
                .map_err(|e| compat_error("convertStringsToChars", e))
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .map(|value| convert_strings_to_chars(value, true))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape)
                .map_err(|e| compat_error("convertContainedStringsToChars", e))
        }
        Value::Struct(mut st) => {
            for value in st.fields.values_mut() {
                *value = convert_strings_to_chars(value.clone(), true)?;
            }
            Ok(Value::Struct(st))
        }
        other => Ok(other),
    }
}

fn convert_chars_to_strings(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::CharArray(array) => {
            let data = (0..array.rows)
                .map(|row| {
                    char_row_to_string_slice(&array.data, array.cols, row)
                        .trim_end()
                        .to_string()
                })
                .collect::<Vec<_>>();
            StringArray::new(data, vec![array.rows, 1])
                .map(Value::StringArray)
                .map_err(|e| compat_error("convertCharsToStrings", e))
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .map(convert_chars_to_strings)
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape)
                .map_err(|e| compat_error("convertCharsToStrings", e))
        }
        other => Ok(other),
    }
}

fn prefix_eq_ignore_case(a: &str, b: &str, n: usize) -> bool {
    a.chars()
        .take(n)
        .map(|ch| ch.to_lowercase().collect::<String>())
        .eq(b
            .chars()
            .take(n)
            .map(|ch| ch.to_lowercase().collect::<String>()))
        && a.chars().count() >= n
        && b.chars().count() >= n
}

fn classify_text_value(
    value: Value,
    fn_name: &str,
    pred: impl Fn(char) -> bool + Copy,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => {
            let data = text
                .chars()
                .map(|ch| u8::from(pred(ch)))
                .collect::<Vec<_>>();
            logical_value(data, vec![1, text.chars().count()], fn_name)
        }
        Value::StringArray(array) => {
            let values = array
                .data
                .into_iter()
                .map(|text| classify_text_value(Value::String(text), fn_name, pred))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, array.shape).map_err(|e| compat_error(fn_name, e))
        }
        Value::CharArray(array) => {
            let data = array
                .data
                .iter()
                .map(|ch| u8::from(pred(*ch)))
                .collect::<Vec<_>>();
            logical_value(data, vec![array.rows, array.cols], fn_name)
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .map(|value| classify_text_value(value, fn_name, pred))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape).map_err(|e| compat_error(fn_name, e))
        }
        other => Err(compat_error(
            fn_name,
            format!("{fn_name}: expected text input, got {other:?}"),
        )),
    }
}

fn char_matches_prop(ch: char, prop: &str) -> bool {
    match prop {
        "alpha" | "letter" | "walpha" => ch.is_alphabetic(),
        "alphanum" | "alphanumeric" | "walphanum" => ch.is_alphanumeric(),
        "digit" | "wdigit" => ch.is_ascii_digit(),
        "xdigit" => ch.is_ascii_hexdigit(),
        "space" | "wspace" => ch.is_whitespace(),
        "upper" | "wupper" => ch.is_uppercase(),
        "lower" | "wlower" => ch.is_lowercase(),
        "punct" | "wpunct" => ch.is_ascii_punctuation(),
        "cntrl" | "control" => ch.is_control(),
        "graphic" | "wgraphic" | "print" | "wprint" => !ch.is_control(),
        _ => false,
    }
}

fn map_text_pair_preserve(
    value: Value,
    fn_name: &str,
    map: impl Fn(&str) -> (String, String) + Copy,
) -> BuiltinResult<(Value, Value)> {
    match value {
        Value::String(text) => {
            let (first, second) = map(&text);
            Ok((Value::String(first), Value::String(second)))
        }
        Value::StringArray(array) => {
            let mut first = Vec::with_capacity(array.data.len());
            let mut second = Vec::with_capacity(array.data.len());
            for text in array.data {
                let (a, b) = map(&text);
                first.push(a);
                second.push(b);
            }
            let first = StringArray::new(first, array.shape.clone())
                .map(Value::StringArray)
                .map_err(|e| compat_error(fn_name, e))?;
            let second = StringArray::new(second, array.shape)
                .map(Value::StringArray)
                .map_err(|e| compat_error(fn_name, e))?;
            Ok((first, second))
        }
        Value::CharArray(array) => {
            let mut first = Vec::with_capacity(array.rows);
            let mut second = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                let (a, b) = map(&char_row_to_string_slice(&array.data, array.cols, row));
                first.push(a);
                second.push(b);
            }
            Ok((
                char_rows_from_strings(first, fn_name)?,
                char_rows_from_strings(second, fn_name)?,
            ))
        }
        Value::Cell(cell) => {
            let mut first = Vec::with_capacity(cell.data.len());
            let mut second = Vec::with_capacity(cell.data.len());
            for value in cell.data {
                let (a, b) = map_text_pair_preserve(value, fn_name, map)?;
                first.push(a);
                second.push(b);
            }
            Ok((
                make_cell_with_shape(first, cell.shape.clone())
                    .map_err(|e| compat_error(fn_name, e))?,
                make_cell_with_shape(second, cell.shape).map_err(|e| compat_error(fn_name, e))?,
            ))
        }
        other => Err(compat_error(
            fn_name,
            format!("{fn_name}: expected text input, got {other:?}"),
        )),
    }
}

fn strtok_pair(text: &str, delimiters: &str) -> (String, String) {
    let start = text
        .char_indices()
        .find_map(|(idx, ch)| (!delimiters.contains(ch)).then_some(idx))
        .unwrap_or(text.len());
    if start == text.len() {
        return (String::new(), String::new());
    }
    let token_end = text[start..]
        .char_indices()
        .find_map(|(idx, ch)| delimiters.contains(ch).then_some(start + idx))
        .unwrap_or(text.len());
    (
        text[start..token_end].to_string(),
        text[token_end..].to_string(),
    )
}

fn char_rows_from_strings(rows: Vec<String>, fn_name: &str) -> BuiltinResult<Value> {
    let row_count = rows.len();
    let cols = rows.iter().map(|s| s.chars().count()).max().unwrap_or(0);
    let mut data = Vec::with_capacity(row_count * cols);
    for row in rows {
        let mut chars = row.chars().collect::<Vec<_>>();
        chars.resize(cols, ' ');
        data.extend(chars);
    }
    CharArray::new(data, row_count, cols)
        .map(Value::CharArray)
        .map_err(|e| compat_error(fn_name, e))
}

fn parse_str2num_matrix(text: &str) -> (Value, bool) {
    match parse_numeric_matrix(text, "str2num") {
        Ok(value) => (value, true),
        Err(_) => (Value::Tensor(Tensor::zeros(vec![0, 0])), false),
    }
}

fn parse_numeric_matrix(text: &str, fn_name: &str) -> BuiltinResult<Value> {
    let text = text.trim();
    let text = text.strip_prefix('[').unwrap_or(text);
    let text = text.strip_suffix(']').unwrap_or(text).trim();
    let rows = text
        .split(';')
        .map(|row| {
            row.split(|ch: char| ch.is_whitespace() || ch == ',')
                .filter(|part| !part.is_empty())
                .map(|part| {
                    part.parse::<f64>().map_err(|_| {
                        compat_error(
                            fn_name,
                            format!("{fn_name}: invalid numeric literal '{part}'"),
                        )
                    })
                })
                .collect::<BuiltinResult<Vec<_>>>()
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    if rows.is_empty() || rows.iter().all(Vec::is_empty) {
        return Ok(Value::Tensor(Tensor::zeros(vec![0, 0])));
    }
    let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    if rows.iter().any(|row| row.len() != cols) {
        return Err(compat_error(
            fn_name,
            format!("{fn_name}: rows must have the same number of columns"),
        ));
    }
    let mut data = Vec::with_capacity(rows.len() * cols);
    for col in 0..cols {
        for row in &rows {
            data.push(row[col]);
        }
    }
    Tensor::new(data, vec![rows.len(), cols])
        .map(Value::Tensor)
        .map_err(|e| compat_error(fn_name, e))
}

fn mat2str_value(value: &Value, precision: Option<usize>) -> String {
    match value {
        Value::Num(n) => format_number(*n, precision),
        Value::Int(i) => i.decimal_string(),
        Value::Bool(b) => {
            if *b {
                "true".into()
            } else {
                "false".into()
            }
        }
        Value::String(text) => format!("\"{}\"", text.replace('"', "\"\"")),
        Value::CharArray(array) if array.rows <= 1 => {
            format!(
                "'{}'",
                char_row_to_string_slice(&array.data, array.cols, 0).replace('\'', "''")
            )
        }
        Value::Tensor(tensor) => tensor_to_matrix_string(tensor, precision),
        Value::LogicalArray(array) => {
            let rows = array.shape.first().copied().unwrap_or(array.data.len());
            let cols = array.shape.get(1).copied().unwrap_or(1);
            let data = array
                .data
                .iter()
                .map(|v| f64::from(*v != 0))
                .collect::<Vec<_>>();
            matrix_to_string(&data, rows, cols, precision)
        }
        _ => value.to_string(),
    }
}

fn matrix_to_string(data: &[f64], rows: usize, cols: usize, precision: Option<usize>) -> String {
    matrix_to_string_with(rows, cols, |index| format_number(data[index], precision))
}

fn tensor_to_matrix_string(tensor: &Tensor, precision: Option<usize>) -> String {
    matrix_to_string_with(tensor.rows(), tensor.cols(), |index| {
        let value = tensor
            .numeric_value_at(index)
            .expect("validated tensor storage index");
        match value {
            NumericScalar::F64(value) => format_number(value, precision),
            NumericScalar::F32(value) => format_number(f64::from(value), precision),
            integer => integer
                .into_int_value()
                .expect("non-floating numeric scalar is integer")
                .decimal_string(),
        }
    })
}

fn matrix_to_string_with(
    rows: usize,
    cols: usize,
    mut format_value: impl FnMut(usize) -> String,
) -> String {
    let mut out = String::from("[");
    for row in 0..rows {
        if row > 0 {
            out.push(';');
        }
        for col in 0..cols {
            if col > 0 {
                out.push(' ');
            }
            out.push_str(&format_value(row + col * rows));
        }
    }
    out.push(']');
    out
}

fn format_number(value: f64, precision: Option<usize>) -> String {
    if let Some(precision) = precision {
        format!("{value:.precision$}")
    } else if value.fract() == 0.0 && value.is_finite() {
        format!("{value:.0}")
    } else {
        format!("{value:.15}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

fn bytes_from_value(value: &Value, fn_name: &str) -> BuiltinResult<Vec<u8>> {
    match value {
        Value::Tensor(tensor) => (0..tensor.len())
            .map(|index| {
                let value = tensor.numeric_value_at(index).ok_or_else(|| {
                    compat_error(
                        fn_name,
                        format!("{fn_name}: numeric storage is inconsistent"),
                    )
                })?;
                match value {
                    NumericScalar::F64(value) => byte_from_f64(value, fn_name),
                    NumericScalar::F32(value) => byte_from_f64(f64::from(value), fn_name),
                    integer => Ok(byte_from_intvalue(
                        &integer
                            .into_int_value()
                            .expect("non-floating numeric scalar is integer"),
                    )),
                }
            })
            .collect(),
        Value::Int(i) => Ok(vec![byte_from_intvalue(i)]),
        Value::Num(n) => Ok(vec![byte_from_f64(*n, fn_name)?]),
        Value::CharArray(array) => {
            Ok(char_row_to_string_slice(&array.data, array.cols, 0).into_bytes())
        }
        Value::String(text) => Ok(text.as_bytes().to_vec()),
        other => Err(compat_error(
            fn_name,
            format!("{fn_name}: expected bytes or text, got {other:?}"),
        )),
    }
}

fn byte_from_intvalue(value: &IntValue) -> u8 {
    value
        .try_to_u64()
        .map(|value| value.min(255) as u8)
        .unwrap_or(0)
}

fn byte_from_f64(value: f64, fn_name: &str) -> BuiltinResult<u8> {
    if !value.is_finite() {
        return Err(compat_error(
            fn_name,
            format!("{fn_name}: byte values must be finite"),
        ));
    }
    Ok(value.round().clamp(0.0, 255.0) as u8)
}

fn decode_bytes(bytes: &[u8], encoding: &str) -> BuiltinResult<Value> {
    let encoding = Encoding::for_label(encoding.as_bytes()).unwrap_or(UTF_8);
    let (text, _, _) = encoding.decode(bytes);
    Ok(Value::String(text.into_owned()))
}

struct SscanfResult {
    value: Value,
    count: usize,
    errmsg: String,
    next_index: usize,
}

#[derive(Clone, Copy)]
enum ScanKind {
    Float,
    Integer,
    String,
    Char,
}

#[derive(Clone)]
enum ScanToken {
    Whitespace,
    Literal(char),
    Spec {
        kind: ScanKind,
        width: Option<usize>,
        suppress: bool,
    },
}

fn sscanf_scan(text: &str, format: &str, size: Option<Vec<usize>>) -> BuiltinResult<SscanfResult> {
    let tokens = parse_scan_format(format)?;
    if tokens.is_empty() {
        return Err(compat_error("sscanf", "sscanf: format must not be empty"));
    }

    let mut values = Vec::new();
    let mut pos = 0usize;
    let mut last_success = 0usize;
    loop {
        let start_pos = pos;
        let start_len = values.len();
        let mut matched_all = true;
        for token in &tokens {
            match token {
                ScanToken::Whitespace => {
                    pos = skip_whitespace(text, pos);
                }
                ScanToken::Literal(ch) => {
                    let Some(next) = text[pos..].chars().next() else {
                        matched_all = false;
                        break;
                    };
                    if next != *ch {
                        matched_all = false;
                        break;
                    }
                    pos += next.len_utf8();
                }
                ScanToken::Spec {
                    kind,
                    width,
                    suppress,
                } => {
                    if !matches!(kind, ScanKind::Char) {
                        pos = skip_whitespace(text, pos);
                    }
                    let Some((parsed, next_pos)) = scan_one(text, pos, *kind, *width) else {
                        matched_all = false;
                        break;
                    };
                    pos = next_pos;
                    if !*suppress {
                        values.extend(parsed);
                    }
                }
            }
        }
        if !matched_all {
            break;
        }
        if pos == start_pos || values.len() == start_len && pos >= text.len() {
            break;
        }
        last_success = pos;
        if pos >= text.len() {
            break;
        }
    }

    let count = values.len();
    let mut shape = size.unwrap_or_else(|| vec![count, 1]);
    let limit = shape.iter().product::<usize>();
    if limit > 0 && values.len() > limit {
        values.truncate(limit);
    }
    if shape.iter().product::<usize>() != values.len() {
        shape = vec![values.len(), 1];
    }
    let value = Tensor::new(values, shape)
        .map(Value::Tensor)
        .map_err(|e| compat_error("sscanf", e))?;
    Ok(SscanfResult {
        value,
        count,
        errmsg: String::new(),
        next_index: last_success.saturating_add(1),
    })
}

fn parse_scan_format(format: &str) -> BuiltinResult<Vec<ScanToken>> {
    let mut chars = format.chars().peekable();
    let mut tokens = Vec::new();
    while let Some(ch) = chars.next() {
        if ch.is_whitespace() {
            while chars.peek().is_some_and(|next| next.is_whitespace()) {
                chars.next();
            }
            tokens.push(ScanToken::Whitespace);
            continue;
        }
        if ch != '%' {
            tokens.push(ScanToken::Literal(ch));
            continue;
        }
        if chars.peek() == Some(&'%') {
            chars.next();
            tokens.push(ScanToken::Literal('%'));
            continue;
        }
        let suppress = if chars.peek() == Some(&'*') {
            chars.next();
            true
        } else {
            false
        };
        let mut width = String::new();
        while chars.peek().is_some_and(|next| next.is_ascii_digit()) {
            width.push(chars.next().unwrap());
        }
        let width = if width.is_empty() {
            None
        } else {
            Some(
                width
                    .parse::<usize>()
                    .map_err(|_| compat_error("sscanf", "sscanf: invalid field width"))?,
            )
        };
        let Some(specifier) = chars.next() else {
            return Err(compat_error(
                "sscanf",
                "sscanf: incomplete format specifier",
            ));
        };
        let kind = match specifier {
            'f' | 'e' | 'E' | 'g' | 'G' => ScanKind::Float,
            'd' | 'i' | 'u' => ScanKind::Integer,
            's' => ScanKind::String,
            'c' => ScanKind::Char,
            other => {
                return Err(compat_error(
                    "sscanf",
                    format!("sscanf: unsupported format specifier %{other}"),
                ))
            }
        };
        tokens.push(ScanToken::Spec {
            kind,
            width,
            suppress,
        });
    }
    Ok(tokens)
}

fn scan_one(
    text: &str,
    pos: usize,
    kind: ScanKind,
    width: Option<usize>,
) -> Option<(Vec<f64>, usize)> {
    if pos > text.len() {
        return None;
    }
    let end_limit = width
        .and_then(|w| byte_index_after_n_chars(&text[pos..], w).map(|idx| pos + idx))
        .unwrap_or(text.len());
    match kind {
        ScanKind::Float | ScanKind::Integer => {
            let fragment = &text[pos..end_limit];
            let len = numeric_prefix_len(fragment, matches!(kind, ScanKind::Integer))?;
            let token = &fragment[..len];
            let value = if matches!(kind, ScanKind::Integer) {
                token
                    .parse::<i64>()
                    .map(|value| value as f64)
                    .or_else(|_| token.parse::<f64>())
                    .ok()?
            } else {
                token.parse::<f64>().ok()?
            };
            Some((vec![value], pos + len))
        }
        ScanKind::String => {
            let fragment = &text[pos..end_limit];
            let len = fragment
                .char_indices()
                .find_map(|(idx, ch)| ch.is_whitespace().then_some(idx))
                .unwrap_or(fragment.len());
            if len == 0 {
                None
            } else {
                Some((
                    fragment[..len].chars().map(|ch| ch as u32 as f64).collect(),
                    pos + len,
                ))
            }
        }
        ScanKind::Char => {
            let count = width.unwrap_or(1);
            let len = byte_index_after_n_chars(&text[pos..], count)?;
            Some((
                text[pos..pos + len]
                    .chars()
                    .map(|ch| ch as u32 as f64)
                    .collect(),
                pos + len,
            ))
        }
    }
}

fn numeric_prefix_len(text: &str, integer: bool) -> Option<usize> {
    let mut end = 0usize;
    for (idx, ch) in text.char_indices() {
        let allowed = if integer {
            ch.is_ascii_digit() || ((ch == '+' || ch == '-') && idx == 0)
        } else {
            ch.is_ascii_digit() || matches!(ch, '+' | '-' | '.' | 'e' | 'E')
        };
        if !allowed {
            break;
        }
        end = idx + ch.len_utf8();
    }
    (end > 0).then_some(end)
}

fn skip_whitespace(text: &str, mut pos: usize) -> usize {
    while pos < text.len() {
        let Some(ch) = text[pos..].chars().next() else {
            break;
        };
        if !ch.is_whitespace() {
            break;
        }
        pos += ch.len_utf8();
    }
    pos
}

fn byte_index_after_n_chars(text: &str, count: usize) -> Option<usize> {
    if count == 0 {
        return Some(0);
    }
    text.char_indices()
        .nth(count)
        .map(|(idx, _)| idx)
        .or_else(|| (text.chars().count() == count).then_some(text.len()))
}

fn scan_size_from_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Num(n) => Ok(vec![scan_size_dim(*n)?, 1]),
        Value::Int(value) => Ok(vec![value.try_to_usize().ok_or_else(scan_size_error)?, 1]),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Ok(vec![scan_tensor_size_dim(tensor, 0)?, 1])
        }
        Value::Tensor(tensor) if tensor_element_len(tensor) == 2 => Ok(vec![
            scan_tensor_size_dim(tensor, 0)?,
            scan_tensor_size_dim(tensor, 1)?,
        ]),
        other => Err(compat_error(
            "sscanf",
            format!("sscanf: invalid size argument {other:?}"),
        )),
    }
}

fn tensor_element_len(tensor: &Tensor) -> usize {
    tensor.len()
}

fn scan_tensor_size_dim(tensor: &Tensor, index: usize) -> BuiltinResult<usize> {
    match tensor.numeric_value_at(index).ok_or_else(scan_size_error)? {
        NumericScalar::F64(value) => scan_size_dim(value),
        NumericScalar::F32(value) => scan_size_dim(f64::from(value)),
        integer => integer
            .into_int_value()
            .expect("non-floating numeric scalar is integer")
            .try_to_usize()
            .ok_or_else(scan_size_error),
    }
}

fn scan_size_error() -> crate::RuntimeError {
    compat_error(
        "sscanf",
        "sscanf: size dimensions must be nonnegative integers",
    )
}

fn scan_size_dim(value: f64) -> BuiltinResult<usize> {
    if value.is_infinite() && value.is_sign_positive() {
        return Ok(usize::MAX / 2);
    }
    nonnegative_platform_usize(value).ok_or_else(scan_size_error)
}

async fn bounded_pattern(
    rest: Vec<Value>,
    atom: &str,
    fn_name: &'static str,
) -> BuiltinResult<Value> {
    let regex = if let Some(value) = rest.first() {
        let value = gather_if_needed_async(value)
            .await
            .map_err(map_flow(fn_name))?;
        let n = parse_nonnegative_usize(&value, fn_name)?;
        format!("{atom}{{{n}}}")
    } else {
        format!("{atom}+")
    };
    Ok(pattern_object(&regex))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntValue, IntegerStorage, NumericDType};

    #[test]
    fn mat2str_preserves_exact_uint64_scalar_text() {
        assert_eq!(
            mat2str_value(&Value::Int(runmat_builtins::IntValue::U64(u64::MAX)), None),
            "18446744073709551615"
        );
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            vec![1, 2],
        )
        .expect("typed integer matrix");
        assert_eq!(
            mat2str_value(&Value::Tensor(tensor), None),
            "[18446744073709551615 9007199254740993]"
        );
    }

    #[test]
    fn bounded_count_parsers_preserve_typed_integer_values() {
        assert_eq!(
            parse_nonnegative_usize(&Value::Int(IntValue::U64(7)), "digitsPattern").unwrap(),
            7
        );
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![9]), vec![1, 1])
            .expect("typed scalar tensor");
        assert_eq!(
            parse_nonnegative_usize(&Value::Tensor(tensor), "digitsPattern").unwrap(),
            9
        );
        let pattern =
            block(digits_pattern_builtin(vec![Value::Int(IntValue::U8(3))])).expect("typed count");
        assert_eq!(pattern_regex(&pattern, "test").unwrap(), r"\d{3}");

        for value in [
            Value::Num(1.5),
            Value::Num(usize::MAX as f64 + 1.0),
            Value::Int(IntValue::I8(-1)),
        ] {
            assert!(parse_nonnegative_usize(&value, "digitsPattern").is_err());
        }
    }

    #[test]
    fn sscanf_size_parsing_preserves_typed_integer_dimensions() {
        assert_eq!(
            scan_size_from_value(&Value::Int(IntValue::U16(4))).unwrap(),
            vec![4, 1]
        );
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![1, 2])
            .expect("typed size vector");
        assert_eq!(
            scan_size_from_value(&Value::Tensor(tensor)).unwrap(),
            vec![2, 3]
        );
        let single = Tensor::from_f32(vec![5.0, 6.0], vec![1, 2]).expect("single size vector");
        assert_eq!(
            scan_size_from_value(&Value::Tensor(single)).unwrap(),
            vec![5, 6]
        );
        assert_eq!(
            scan_size_from_value(&Value::Num(f64::INFINITY)).unwrap(),
            vec![usize::MAX / 2, 1]
        );

        for value in [
            Value::Num(1.5),
            Value::Num(usize::MAX as f64 + 1.0),
            Value::Int(IntValue::I16(-1)),
        ] {
            assert!(scan_size_from_value(&value).is_err());
        }
    }

    #[test]
    fn native2unicode_reads_typed_integer_byte_storage_exactly() {
        let bytes = Tensor::new_integer(IntegerStorage::U8(vec![104, 105]), vec![1, 2]).unwrap();
        assert_eq!(
            block(native2unicode_builtin(Value::Tensor(bytes), Vec::new())).unwrap(),
            Value::String("hi".into())
        );
        let single = Tensor::from_f32(vec![111.0, 107.0], vec![1, 2]).expect("single bytes");
        assert_eq!(
            block(native2unicode_builtin(Value::Tensor(single), Vec::new())).unwrap(),
            Value::String("ok".into())
        );
    }

    fn block(
        value: impl std::future::Future<Output = BuiltinResult<Value>>,
    ) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
    }

    #[test]
    fn basic_text_core_helpers_work() {
        assert_eq!(newline_builtin().unwrap(), Value::String("\n".into()));
        assert_eq!(
            block(blanks_builtin(Value::Num(3.0))).unwrap(),
            Value::CharArray(CharArray::new_row("   "))
        );
        assert_eq!(
            is_string_scalar_builtin(Value::String("x".into())).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn strncmpi_and_classifiers_work() {
        assert_eq!(
            block(strncmpi_builtin(
                Value::String("RunMat".into()),
                Value::String("runway".into()),
                Value::Num(3.0),
            ))
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            block(isletter_builtin(Value::CharArray(CharArray::new_row("a1")))).unwrap(),
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap())
        );
    }

    #[test]
    fn conversions_and_numeric_parsing_work() {
        assert_eq!(
            block(convert_strings_to_chars_builtin(
                Value::String("abc".into()),
                Vec::new(),
            ))
            .unwrap(),
            Value::CharArray(CharArray::new_row("abc"))
        );
        let _guard = crate::output_count::push_output_count(Some(2));
        assert!(matches!(
            block(convert_strings_to_chars_builtin(
                Value::String("a".into()),
                vec![Value::String("b".into())],
            ))
            .unwrap(),
            Value::OutputList(outputs) if outputs.len() == 2
        ));
        drop(_guard);
        assert_eq!(
            block(convert_chars_to_strings_builtin(Value::CharArray(
                CharArray::new_row("abc")
            )))
            .unwrap(),
            Value::StringArray(StringArray::new(vec!["abc".into()], vec![1, 1]).unwrap())
        );
        assert_eq!(
            block(str2num_builtin(Value::String("1 2; 3 4".into()))).unwrap(),
            Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap())
        );
        assert_eq!(
            block(mat2str_builtin(
                Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap()),
                Vec::new(),
            ))
            .unwrap(),
            Value::String("[1 2;3 4]".into())
        );
    }

    #[test]
    fn tokenizing_encoding_and_scanning_work() {
        assert_eq!(
            block(strtok_builtin(
                Value::String("  alpha,beta".into()),
                vec![Value::String(" ,".into())],
            ))
            .unwrap(),
            Value::String("alpha".into())
        );
        assert_eq!(
            block(native2unicode_builtin(
                Value::Tensor(
                    Tensor::new_with_dtype(vec![104.0, 105.0], vec![1, 2], NumericDType::U8)
                        .unwrap()
                ),
                Vec::new(),
            ))
            .unwrap(),
            Value::String("hi".into())
        );
        assert_eq!(
            block(sscanf_builtin(
                Value::String("1 2 x".into()),
                vec![Value::String("%f".into())],
            ))
            .unwrap(),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap())
        );
    }

    #[test]
    fn pattern_constructors_store_regex() {
        let value = block(digits_pattern_builtin(vec![Value::Num(2.0)])).unwrap();
        assert_eq!(pattern_regex(&value, "test").unwrap(), "\\d{2}");
        assert_eq!(
            pattern_regex(&block(letters_pattern_builtin(Vec::new())).unwrap(), "test").unwrap(),
            r"\p{Alphabetic}+"
        );
        assert_eq!(
            pattern_regex(
                &block(wildcard_pattern_builtin(Vec::new())).unwrap(),
                "test"
            )
            .unwrap(),
            ".*"
        );
        assert_eq!(
            pattern_regex(&block(text_boundary_builtin(Vec::new())).unwrap(), "test").unwrap(),
            r"(?:^|$)"
        );
        assert_eq!(
            pattern_regex(
                &block(text_boundary_builtin(vec![Value::String("start".into())])).unwrap(),
                "test"
            )
            .unwrap(),
            r"^"
        );
        assert_eq!(
            pattern_regex(
                &block(text_boundary_builtin(vec![Value::String("end".into())])).unwrap(),
                "test"
            )
            .unwrap(),
            r"$"
        );
    }

    #[test]
    fn text_boundary_rejects_invalid_type() {
        let err = block(text_boundary_builtin(vec![Value::String("middle".into())]))
            .expect_err("expected invalid boundary type");
        assert!(err.to_string().contains("unsupported boundary type"));
    }

    #[test]
    fn text_boundary_pattern_works_with_replace() {
        let pattern = block(text_boundary_builtin(vec![Value::String("start".into())])).unwrap();
        let result = block(crate::call_builtin_async(
            "replace",
            &[
                Value::String("abc".into()),
                pattern,
                Value::String(">".into()),
            ],
        ))
        .expect("replace");
        assert_eq!(result, Value::String(">abc".into()));
    }
}
