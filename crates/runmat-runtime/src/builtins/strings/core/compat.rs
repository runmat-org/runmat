//! MATLAB text compatibility helpers that do not warrant larger domain modules yet.

use encoding_rs::Encoding;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_builtins::{BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind};
use runmat_macros::runtime_builtin;
use runmat_value::NumericStorage;
use runmat_value::{
    CharArray, IntValue, LogicalArray, NumericScalar, ObjectInstance, StringArray, Tensor, Value,
};

use crate::builtins::common::broadcast as matlab_broadcast;
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::common::{
    char_row_to_string_slice, contains_numeric_or_resident_text_input, is_missing_string,
};
use crate::builtins::strings::core::numeric_length::{
    nonnegative_platform_usize, parse_nonnegative_usize,
};
use crate::{build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult};

const PATTERN_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("pattern");

const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output value.",
}];

const IN_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text.",
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

descriptor_by_outputs!(
    STRTOK_DESCRIPTOR,
    "[tok, rem] = strtok(text, delimiters)",
    &IN_TEXT_REST,
    &OUT_ANY
);
pub const STRTOK_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "strtok accepts host text and an optional text delimiter. Integer, numeric, and provider-resident values in either role reject before provider access and are not interpreted as character codes.",
};
descriptor_by_outputs!(
    STR2NUM_DESCRIPTOR,
    "[x, tf] = str2num(text)",
    &IN_TEXT,
    &OUT_ANY
);
pub const STR2NUM_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "str2num evaluates scalar string or character text and therefore has no integer input role. Integer, numeric, and provider-resident input rejects before provider access; parsed numeric output is a separate result-class concern.",
};
const MAT2STR_INPUT_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric or logical matrix to serialize.",
}];
const MAT2STR_INPUT_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric or logical matrix to serialize.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive number of significant digits.",
    },
];
const MAT2STR_INPUT_CLASS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric or logical matrix to serialize.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional positive number of significant digits.",
    },
    BuiltinParamDescriptor {
        name: "class",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"class\""),
        description: "Include the input class constructor in the expression.",
    },
];
const MAT2STR_INPUT_CLASS_ONLY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric or logical matrix to serialize.",
    },
    BuiltinParamDescriptor {
        name: "class",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"class\""),
        description: "Include the input class constructor in the expression.",
    },
];
const MAT2STR_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "s = mat2str(A)",
        inputs: &MAT2STR_INPUT_A,
        outputs: &OUT_ANY,
    },
    BuiltinSignatureDescriptor {
        label: "s = mat2str(A, n)",
        inputs: &MAT2STR_INPUT_N,
        outputs: &OUT_ANY,
    },
    BuiltinSignatureDescriptor {
        label: "s = mat2str(A, 'class')",
        inputs: &MAT2STR_INPUT_CLASS_ONLY,
        outputs: &OUT_ANY,
    },
    BuiltinSignatureDescriptor {
        label: "s = mat2str(A, n, 'class')",
        inputs: &MAT2STR_INPUT_CLASS,
        outputs: &OUT_ANY,
    },
];
const MAT2STR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MAT2STR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NO_ERRORS,
};

const MAT2STR_INTEGER_PRECISION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "mat2str-integer-precision",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "typed integer mat2str precision is a RunMat extension because the public MATLAB page requires a positive integer without enumerating native integer storage classes",
        error_identifier: Some("RunMat:compatibility:Mat2strIntegerPrecisionExtension"),
    };
const MAT2STR_TEXT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mat2str-text-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mat2str input outside the documented numeric and logical matrix domain is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Mat2strTextInputExtension"),
};
pub const MAT2STR_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    MAT2STR_INTEGER_PRECISION_EXTENSION,
    MAT2STR_TEXT_INPUT_EXTENSION,
];

const MAT2STR_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer matrix classes serialize from exact native values; the 'class' option emits the matching constructor.",
    }];
const MAT2STR_INTEGER_PRECISION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat accepts an exact positive typed integer precision under its extension policy; strict compatibility retains the publicly evidenced floating scalar form.",
    }];
pub const MAT2STR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "s = mat2str(integer_A[, n][, 'class'])",
        inputs: &MAT2STR_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Decimal serialization never crosses binary64; automatic residency gathers authoritatively, while explicit gpuArray input is unsupported.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "s = mat2str(A, integer_n[, 'class'])",
        inputs: &MAT2STR_INTEGER_PRECISION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The RunMat-only typed precision is read exactly and must be positive; explicit resident precision rejects before transfer.",
    },
];

const NATIVE2UNICODE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bytes",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric byte vector with values in the range 0 through 255.",
    },
    BuiltinParamDescriptor {
        name: "encoding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"UTF-8\""),
        description: "Source character encoding.",
    },
];
descriptor!(
    NATIVE2UNICODE_DESCRIPTOR,
    "text = native2unicode(bytes, encoding)",
    &NATIVE2UNICODE_INPUTS,
    &OUT_ANY
);

const NATIVE2UNICODE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "bytes",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented for byte vectors; every element must be exactly within 0 through 255.",
    }];
pub const NATIVE2UNICODE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "text = native2unicode(integer_bytes[, encoding])",
        inputs: &NATIVE2UNICODE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Native byte extraction is exact and range checked before decoding to a host character vector; explicit gpuArray input is unsupported and automatic residency gathers transparently.",
    }];
descriptor_by_outputs!(
    SSCANF_DESCRIPTOR,
    "[A, count, errmsg, nextindex] = sscanf(text, format, size)",
    &IN_TEXT_REST,
    &OUT_ANY
);
const SSCANF_INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sizeA",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The optional output-size scalar or two-element vector accepts every built-in integer class and is decoded directly from native storage.",
    }];
pub const SSCANF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "A = sscanf(text, format, integer_sizeA)",
        inputs: &SSCANF_INTEGER_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "sizeA controls output shape only. Pure %ld/%li scans return int64, pure %lu/%lo/%lx scans return uint64, and other numeric scan formats return double.",
    }];

fn any_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

fn string_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::String
}

fn tensor_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

pub(crate) fn compat_error(name: &str, message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

fn map_flow(name: &'static str) -> impl Fn(crate::RuntimeError) -> crate::RuntimeError {
    move |err| map_control_flow_with_builtin(err, name)
}

#[runtime_builtin(
    name = "strtok",
    category = "strings/core",
    summary = "Return the first token from text using delimiter characters.",
    keywords = "strtok,tokenize,delimiter,text",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::core::compat::STRTOK_DESCRIPTOR),
    integer_audit(crate::builtins::strings::core::compat::STRTOK_INTEGER_AUDIT),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn strtok_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if contains_numeric_or_resident_text_input(&text)
        || rest
            .first()
            .is_some_and(contains_numeric_or_resident_text_input)
    {
        return Err(compat_error(
            "strtok",
            "strtok: expected host text and a text delimiter",
        ));
    }
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
    integer_audit(crate::builtins::strings::core::compat::STR2NUM_INTEGER_AUDIT),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn str2num_builtin(text: Value) -> BuiltinResult<Value> {
    if contains_numeric_or_resident_text_input(&text) {
        return Err(compat_error(
            "str2num",
            "str2num: expected a string scalar or character array",
        ));
    }
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
    integer_capabilities(crate::builtins::strings::core::compat::MAT2STR_INTEGER_CAPABILITIES),
    extensions(crate::builtins::strings::core::compat::MAT2STR_EXTENSIONS),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn mat2str_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 2 {
        return Err(compat_error(
            "mat2str",
            "mat2str: expected A, A,n, A,'class', or A,n,'class'",
        ));
    }
    reject_explicit_gpu_text_sink(&value, "mat2str", "input array")?;
    if matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MAT2STR_TEXT_INPUT_EXTENSION,
            "mat2str",
        )?;
    }

    let mut precision = None;
    let mut include_class = false;
    for (index, arg) in rest.iter().enumerate() {
        reject_explicit_gpu_text_sink(arg, "mat2str", "precision")?;
        if let Ok(keyword) = scalar_text(arg, "mat2str") {
            if keyword.eq_ignore_ascii_case("class") && index + 1 == rest.len() {
                include_class = true;
                continue;
            }
            return Err(compat_error(
                "mat2str",
                "mat2str: the only supported text option is final 'class'",
            ));
        }
        if precision.is_some() || include_class {
            return Err(compat_error(
                "mat2str",
                "mat2str: precision must precede the optional 'class' flag",
            ));
        }
        if is_typed_integer_value(arg) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &MAT2STR_INTEGER_PRECISION_EXTENSION,
                "mat2str",
            )?;
        }
        let arg = gather_if_needed_async(arg)
            .await
            .map_err(map_flow("mat2str"))?;
        let parsed = parse_nonnegative_usize(&arg, "mat2str")?;
        if parsed == 0 {
            return Err(compat_error(
                "mat2str",
                "mat2str: precision must be a positive integer scalar",
            ));
        }
        precision = Some(parsed);
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_flow("mat2str"))?;
    Ok(Value::CharArray(CharArray::new_row(&mat2str_value(
        &value,
        precision,
        include_class,
    ))))
}

#[runtime_builtin(
    name = "native2unicode",
    category = "strings/core",
    summary = "Decode native byte values into Unicode text.",
    keywords = "native2unicode,unicode,encoding,text,uint8",
    accel = "sink",
    type_resolver(string_type),
    descriptor(crate::builtins::strings::core::compat::NATIVE2UNICODE_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::strings::core::compat::NATIVE2UNICODE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::strings::core::compat"
)]
async fn native2unicode_builtin(bytes: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(compat_error(
            "native2unicode",
            "native2unicode: expected bytes and optional encoding",
        ));
    }
    reject_explicit_gpu_text_sink(&bytes, "native2unicode", "byte vector")?;
    let bytes = gather_if_needed_async(&bytes)
        .await
        .map_err(map_flow("native2unicode"))?;
    if matches!(&bytes, Value::String(_) | Value::CharArray(_)) {
        return Ok(bytes);
    }
    let encoding = if let Some(value) = rest.first() {
        reject_explicit_gpu_text_sink(value, "native2unicode", "encoding")?;
        let value = gather_if_needed_async(value)
            .await
            .map_err(map_flow("native2unicode"))?;
        scalar_text(&value, "native2unicode")?
    } else {
        "UTF-8".to_string()
    };
    let shape = byte_vector_shape(&bytes, "native2unicode")?;
    let bytes = bytes_from_value(&bytes, "native2unicode")?;
    decode_bytes(&bytes, &encoding, shape)
}

fn reject_explicit_gpu_text_sink(value: &Value, name: &str, role: &str) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    {
        return Err(compat_error(
            name,
            format!("{name}: explicit gpuArray {role} is not supported"),
        ));
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

#[runtime_builtin(
    name = "sscanf",
    category = "strings/core",
    summary = "Parse formatted numeric values from text.",
    keywords = "sscanf,scan,format,text,numeric",
    accel = "sink",
    type_resolver(tensor_type),
    descriptor(crate::builtins::strings::core::compat::SSCANF_DESCRIPTOR),
    integer_capabilities(crate::builtins::strings::core::compat::SSCANF_INTEGER_CAPABILITIES),
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
    pub(crate) fn from_value(value: Value, fn_name: &str) -> BuiltinResult<Self> {
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
    let extended = matlab_broadcast::align_shape(source_shape, shape.len());
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

fn mat2str_value(value: &Value, precision: Option<usize>, include_class: bool) -> String {
    let body = match value {
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
    };
    if !include_class {
        return body;
    }
    match mat2str_class_name(value) {
        Some(class) => format!("{class}({body})"),
        None => body,
    }
}

fn mat2str_class_name(value: &Value) -> Option<&'static str> {
    match value {
        Value::Num(_) => Some("double"),
        Value::Int(value) => Some(value.class_name()),
        Value::Bool(_) | Value::LogicalArray(_) => Some("logical"),
        Value::Tensor(tensor) => Some(tensor.numeric_dtype().class_name()),
        _ => None,
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
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value == f64::INFINITY {
        return "Inf".to_string();
    }
    if value == f64::NEG_INFINITY {
        return "-Inf".to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let digits = precision.unwrap_or(15).max(1);
    let exponent = value.abs().log10().floor() as i32;
    if exponent < -4 || exponent >= digits as i32 {
        let decimals = digits.saturating_sub(1);
        return normalize_scientific(format!("{value:.decimals$e}"));
    }
    let decimals = (digits as i32 - 1 - exponent).max(0) as usize;
    let fixed = trim_decimal(format!("{value:.decimals$}"));
    if fixed
        .parse::<f64>()
        .ok()
        .is_some_and(|rounded| rounded != 0.0 && rounded.abs().log10().floor() >= digits as f64)
    {
        let decimals = digits.saturating_sub(1);
        normalize_scientific(format!("{value:.decimals$e}"))
    } else {
        fixed
    }
}

fn trim_decimal(mut value: String) -> String {
    if value.contains('.') {
        while value.ends_with('0') {
            value.pop();
        }
        if value.ends_with('.') {
            value.pop();
        }
    }
    value
}

fn normalize_scientific(value: String) -> String {
    let Some((mantissa, exponent)) = value.split_once('e') else {
        return value;
    };
    let mantissa = trim_decimal(mantissa.to_string());
    let exponent = exponent.parse::<i32>().unwrap_or(0);
    format!("{mantissa}e{exponent:+03}")
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
                    integer => byte_from_intvalue(
                        &integer
                            .into_int_value()
                            .expect("non-floating numeric scalar is integer"),
                        fn_name,
                    ),
                }
            })
            .collect(),
        Value::Int(i) => Ok(vec![byte_from_intvalue(i, fn_name)?]),
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

fn byte_from_intvalue(value: &IntValue, fn_name: &str) -> BuiltinResult<u8> {
    value
        .try_to_u64()
        .filter(|value| *value <= u8::MAX as u64)
        .map(|value| value as u8)
        .ok_or_else(|| {
            compat_error(
                fn_name,
                format!("{fn_name}: byte values must be in the range 0 through 255"),
            )
        })
}

fn byte_from_f64(value: f64, fn_name: &str) -> BuiltinResult<u8> {
    if !value.is_finite() {
        return Err(compat_error(
            fn_name,
            format!("{fn_name}: byte values must be finite"),
        ));
    }
    if !(0.0..=255.0).contains(&value) {
        return Err(compat_error(
            fn_name,
            format!("{fn_name}: byte values must be in the range 0 through 255"),
        ));
    }
    Ok(value.round() as u8)
}

fn decode_bytes(bytes: &[u8], encoding: &str, shape: Vec<usize>) -> BuiltinResult<Value> {
    let encoding = Encoding::for_label(encoding.as_bytes()).ok_or_else(|| {
        compat_error(
            "native2unicode",
            format!("native2unicode: unsupported encoding '{encoding}'"),
        )
    })?;
    let (text, _, _) = encoding.decode(bytes);
    let chars = text.chars().collect::<Vec<_>>();
    let output = if shape.iter().product::<usize>() == chars.len() {
        CharArray::from_column_major(chars, shape)
    } else {
        Ok(CharArray::new_row(text.as_ref()))
    }
    .map_err(|error| compat_error("native2unicode", error))?;
    Ok(Value::CharArray(output))
}

fn byte_vector_shape(value: &Value, fn_name: &str) -> BuiltinResult<Vec<usize>> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape.clone(),
        Value::Int(_) | Value::Num(_) => vec![1, 1],
        other => {
            return Err(compat_error(
                fn_name,
                format!("{fn_name}: expected a numeric byte vector, got {other:?}"),
            ))
        }
    };
    if shape.iter().filter(|&&extent| extent > 1).count() > 1 {
        return Err(compat_error(
            fn_name,
            format!("{fn_name}: byte input must be a vector"),
        ));
    }
    Ok(match shape.as_slice() {
        [] => vec![1, 1],
        [length] => vec![1, *length],
        _ => shape,
    })
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
    SignedInteger {
        radix: IntegerRadix,
        exact_i64: bool,
    },
    UnsignedInteger {
        radix: IntegerRadix,
        exact_u64: bool,
    },
    String,
    Char,
}

#[derive(Clone, Copy)]
enum IntegerRadix {
    Decimal,
    Auto,
    Octal,
    Hex,
}

#[derive(Clone, Copy)]
enum ScanValue {
    F64(f64),
    I64(i64),
    U64(u64),
}

#[derive(Clone, Copy)]
enum ScanOutputClass {
    F64,
    I64,
    U64,
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
    let storage = scan_values_into_storage(values, scan_output_class(&tokens))?;
    let value = Tensor::from_numeric_storage(storage, shape)
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
        let long = if chars.peek() == Some(&'l') {
            chars.next();
            true
        } else {
            false
        };
        let Some(specifier) = chars.next() else {
            return Err(compat_error(
                "sscanf",
                "sscanf: incomplete format specifier",
            ));
        };
        let kind = match specifier {
            'f' | 'e' | 'E' | 'g' | 'G' => ScanKind::Float,
            'd' => ScanKind::SignedInteger {
                radix: IntegerRadix::Decimal,
                exact_i64: long,
            },
            'i' => ScanKind::SignedInteger {
                radix: IntegerRadix::Auto,
                exact_i64: long,
            },
            'u' => ScanKind::UnsignedInteger {
                radix: IntegerRadix::Decimal,
                exact_u64: long,
            },
            'o' => ScanKind::UnsignedInteger {
                radix: IntegerRadix::Octal,
                exact_u64: long,
            },
            'x' | 'X' => ScanKind::UnsignedInteger {
                radix: IntegerRadix::Hex,
                exact_u64: long,
            },
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
) -> Option<(Vec<ScanValue>, usize)> {
    if pos > text.len() {
        return None;
    }
    let end_limit = width
        .and_then(|w| byte_index_after_n_chars(&text[pos..], w).map(|idx| pos + idx))
        .unwrap_or(text.len());
    match kind {
        ScanKind::Float => {
            let fragment = &text[pos..end_limit];
            let len = numeric_prefix_len(fragment, false)?;
            let token = &fragment[..len];
            Some((vec![ScanValue::F64(token.parse::<f64>().ok()?)], pos + len))
        }
        ScanKind::SignedInteger { radix, exact_i64 } => {
            let fragment = &text[pos..end_limit];
            let (value, len) = scan_signed_integer(fragment, radix)?;
            let value = if exact_i64 {
                ScanValue::I64(value)
            } else {
                ScanValue::F64(value as f64)
            };
            Some((vec![value], pos + len))
        }
        ScanKind::UnsignedInteger { radix, exact_u64 } => {
            let fragment = &text[pos..end_limit];
            let (value, len) = scan_unsigned_integer(fragment, radix)?;
            let value = if exact_u64 {
                ScanValue::U64(value)
            } else {
                ScanValue::F64(value as f64)
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
                    fragment[..len]
                        .chars()
                        .map(|ch| ScanValue::F64(ch as u32 as f64))
                        .collect(),
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
                    .map(|ch| ScanValue::F64(ch as u32 as f64))
                    .collect(),
                pos + len,
            ))
        }
    }
}

fn scan_output_class(tokens: &[ScanToken]) -> ScanOutputClass {
    let mut class = None;
    for token in tokens {
        let ScanToken::Spec {
            kind,
            suppress: false,
            ..
        } = token
        else {
            continue;
        };
        let next = match kind {
            ScanKind::SignedInteger {
                exact_i64: true, ..
            } => ScanOutputClass::I64,
            ScanKind::UnsignedInteger {
                exact_u64: true, ..
            } => ScanOutputClass::U64,
            _ => return ScanOutputClass::F64,
        };
        match (class, next) {
            (None, next) => class = Some(next),
            (Some(ScanOutputClass::I64), ScanOutputClass::I64)
            | (Some(ScanOutputClass::U64), ScanOutputClass::U64) => {}
            _ => return ScanOutputClass::F64,
        }
    }
    class.unwrap_or(ScanOutputClass::F64)
}

fn scan_values_into_storage(
    values: Vec<ScanValue>,
    class: ScanOutputClass,
) -> BuiltinResult<NumericStorage> {
    match class {
        ScanOutputClass::F64 => Ok(NumericStorage::F64(
            values
                .into_iter()
                .map(|value| match value {
                    ScanValue::F64(value) => value,
                    ScanValue::I64(value) => value as f64,
                    ScanValue::U64(value) => value as f64,
                })
                .collect(),
        )),
        ScanOutputClass::I64 => values
            .into_iter()
            .map(|value| match value {
                ScanValue::I64(value) => Ok(value),
                _ => Err(compat_error(
                    "sscanf",
                    "sscanf: incompatible conversions in int64 scan format",
                )),
            })
            .collect::<BuiltinResult<Vec<_>>>()
            .map(NumericStorage::I64),
        ScanOutputClass::U64 => values
            .into_iter()
            .map(|value| match value {
                ScanValue::U64(value) => Ok(value),
                _ => Err(compat_error(
                    "sscanf",
                    "sscanf: incompatible conversions in uint64 scan format",
                )),
            })
            .collect::<BuiltinResult<Vec<_>>>()
            .map(NumericStorage::U64),
    }
}

fn scan_signed_integer(text: &str, radix: IntegerRadix) -> Option<(i64, usize)> {
    let (negative, digits, base, len) = integer_token_parts(text, radix, true)?;
    let magnitude = u64::from_str_radix(digits, base).ok()?;
    let value = if negative {
        if magnitude == (1_u64 << 63) {
            i64::MIN
        } else {
            -i64::try_from(magnitude).ok()?
        }
    } else {
        i64::try_from(magnitude).ok()?
    };
    Some((value, len))
}

fn scan_unsigned_integer(text: &str, radix: IntegerRadix) -> Option<(u64, usize)> {
    let (negative, digits, base, len) = integer_token_parts(text, radix, false)?;
    if negative {
        return None;
    }
    Some((u64::from_str_radix(digits, base).ok()?, len))
}

fn integer_token_parts(
    text: &str,
    radix: IntegerRadix,
    allow_negative: bool,
) -> Option<(bool, &str, u32, usize)> {
    let bytes = text.as_bytes();
    let mut start = 0usize;
    let mut negative = false;
    if let Some(sign) = bytes.first() {
        if *sign == b'+' {
            start = 1;
        } else if *sign == b'-' && allow_negative {
            start = 1;
            negative = true;
        }
    }
    let remaining = &text[start..];
    let (base, prefix_len) = match radix {
        IntegerRadix::Decimal => (10, 0),
        IntegerRadix::Octal => (8, 0),
        IntegerRadix::Hex => (
            16,
            usize::from(remaining.starts_with("0x") || remaining.starts_with("0X")) * 2,
        ),
        IntegerRadix::Auto if remaining.starts_with("0x") || remaining.starts_with("0X") => (16, 2),
        IntegerRadix::Auto if remaining.starts_with('0') => (8, 0),
        IntegerRadix::Auto => (10, 0),
    };
    let digit_start = start + prefix_len;
    let digit_len = text[digit_start..]
        .char_indices()
        .take_while(|(_, ch)| ch.is_digit(base))
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()?;
    let end = digit_start + digit_len;
    Some((negative, &text[digit_start..end], base, end))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::strings::core::{
        blanks::blanks_builtin, convert_chars_to_strings::convert_chars_to_strings_builtin,
        convert_contained_strings_to_chars::convert_contained_strings_to_chars_builtin,
        convert_strings_to_chars::convert_strings_to_chars_builtin,
        digits_pattern::digits_pattern_builtin, is_string_scalar::is_string_scalar_builtin,
        isletter::isletter_builtin, isspace::isspace_builtin, isstrprop::isstrprop_builtin,
        letters_pattern::letters_pattern_builtin, newline::newline_builtin,
        pattern::pattern_builtin, regexp_pattern::regexp_pattern_builtin,
        strncmpi::strncmpi_builtin, text_boundary::text_boundary_builtin,
        wildcard_pattern::wildcard_pattern_builtin,
    };
    use runmat_builtins::{
        BLANKS_GPU_INPUT_EXTENSION, DIGITS_PATTERN_DESCRIPTOR, DIGITS_PATTERN_ERROR_ARGUMENT_COUNT,
        DIGITS_PATTERN_ERROR_INVALID_COUNT, DIGITS_PATTERN_ERROR_INVALID_RANGE,
        DIGITS_PATTERN_ERROR_TOO_MANY_OUTPUTS, DIGITS_PATTERN_INTEGER_CAPABILITIES,
        ISLETTER_ERROR_TOO_MANY_OUTPUTS, ISSPACE_ERROR_TOO_MANY_OUTPUTS,
        ISSTRPROP_ERROR_TOO_MANY_OUTPUTS, ISSTRPROP_RESIDENT_INPUT_EXTENSION,
        LETTERS_PATTERN_ERROR_TOO_MANY_OUTPUTS, PATTERN_ERROR_TOO_MANY_OUTPUTS,
        REGEXP_PATTERN_ERROR_TOO_MANY_OUTPUTS, STRNCMPI_ERROR_TOO_MANY_OUTPUTS,
        TEXT_BOUNDARY_ERROR_TOO_MANY_OUTPUTS, WILDCARD_PATTERN_ERROR_TOO_MANY_OUTPUTS,
    };
    use runmat_value::{CellArray, IntValue, IntegerStorage, NumericDType, StructValue};

    #[test]
    fn text_broadcast_helpers_append_trailing_singletons() {
        assert_eq!(
            broadcast_shape(&[2, 1], &[2, 1, 3], "test").unwrap(),
            vec![2, 1, 3]
        );
        assert!(broadcast_shape(&[2, 3], &[1, 2, 3], "test").is_err());
        assert_eq!(
            (0..6)
                .map(|linear| broadcast_flat_index(linear, &[2, 1, 3], &[2, 1]))
                .collect::<Vec<_>>(),
            vec![0, 1, 0, 1, 0, 1]
        );
    }

    #[test]
    fn mat2str_preserves_exact_uint64_scalar_text() {
        assert_eq!(
            mat2str_value(
                &Value::Int(runmat_value::IntValue::U64(u64::MAX)),
                None,
                false,
            ),
            "18446744073709551615"
        );
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            vec![1, 2],
        )
        .expect("typed integer matrix");
        assert_eq!(
            mat2str_value(&Value::Tensor(tensor), None, false),
            "[18446744073709551615 9007199254740993]"
        );

        let typed = Tensor::new_integer(IntegerStorage::U16(vec![256, 512]), vec![1, 2])
            .expect("typed integer matrix");
        assert_eq!(
            mat2str_value(&Value::Tensor(typed), None, true),
            "uint16([256 512])"
        );
    }

    #[test]
    fn mat2str_precision_is_significant_digits_and_positive() {
        assert_eq!(format_number(std::f64::consts::PI, Some(3)), "3.14");
        assert_eq!(format_number(12_345.0, Some(3)), "1.23e+04");
        assert_eq!(format_number(0.000_012_345, Some(3)), "1.23e-05");
        assert_eq!(format_number(999.9, Some(3)), "1e+03");
        let error = block(mat2str_builtin(Value::Num(1.0), vec![Value::Num(0.0)]))
            .expect_err("zero precision must reject");
        assert!(error.message().contains("positive"));
    }

    #[test]
    fn mat2str_typed_precision_and_text_input_follow_extension_policy() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block(mat2str_builtin(
            Value::Int(IntValue::U16(12)),
            vec![Value::Int(IntValue::U8(3))],
        ))
        .expect_err("typed precision is extension-gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:Mat2strIntegerPrecisionExtension")
        );
        let error = block(mat2str_builtin(Value::String("x".into()), Vec::new()))
            .expect_err("text input is extension-gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:Mat2strTextInputExtension")
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
    fn sscanf_long_integer_formats_preserve_full_width_values() {
        let signed = sscanf_scan("-9223372036854775808 9223372036854775807", "%ld", None)
            .expect("signed long scan");
        let Value::Tensor(signed) = signed.value else {
            panic!("expected signed tensor");
        };
        assert_eq!(signed.numeric_dtype(), NumericDType::I64);
        assert_eq!(
            signed.integer_storage(),
            Some(&IntegerStorage::I64(vec![i64::MIN, i64::MAX]))
        );

        let unsigned = sscanf_scan("18446744073709551615 ff 177", "%lu %lx %lo", None)
            .expect("unsigned long scan");
        let Value::Tensor(unsigned) = unsigned.value else {
            panic!("expected unsigned tensor");
        };
        assert_eq!(unsigned.numeric_dtype(), NumericDType::U64);
        assert_eq!(
            unsigned.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 255, 127]))
        );
    }

    #[test]
    fn sscanf_mixed_long_integer_formats_follow_double_output_rule() {
        let result = sscanf_scan("42 7.5", "%ld %f", None).expect("mixed scan");
        let Value::Tensor(result) = result.value else {
            panic!("expected tensor");
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F64);
        assert_eq!(result.materialize_f64(), vec![42.0, 7.5]);
    }

    #[test]
    fn native2unicode_reads_typed_integer_byte_storage_exactly() {
        let bytes = Tensor::new_integer(IntegerStorage::U8(vec![104, 105]), vec![1, 2]).unwrap();
        assert_eq!(
            block(native2unicode_builtin(Value::Tensor(bytes), Vec::new())).unwrap(),
            Value::CharArray(CharArray::new_row("hi"))
        );
        let single = Tensor::from_f32(vec![111.0, 107.0], vec![1, 2]).expect("single bytes");
        assert_eq!(
            block(native2unicode_builtin(Value::Tensor(single), Vec::new())).unwrap(),
            Value::CharArray(CharArray::new_row("ok"))
        );

        for value in [IntValue::I16(-1), IntValue::U16(256)] {
            let error = block(native2unicode_builtin(Value::Int(value), Vec::new()))
                .expect_err("out-of-range byte");
            assert!(error.message().contains("0 through 255"));
        }

        let column = Tensor::new_integer(IntegerStorage::U8(vec![104, 105]), vec![2, 1])
            .expect("column bytes");
        let Value::CharArray(decoded) =
            block(native2unicode_builtin(Value::Tensor(column), Vec::new())).unwrap()
        else {
            panic!("expected character vector");
        };
        assert_eq!(decoded.shape, vec![2, 1]);
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
    fn blanks_accepts_every_integer_class_and_clamps_negative_lengths() {
        for value in [
            IntValue::I8(3),
            IntValue::I16(3),
            IntValue::I32(3),
            IntValue::I64(3),
            IntValue::U8(3),
            IntValue::U16(3),
            IntValue::U32(3),
            IntValue::U64(3),
        ] {
            assert_eq!(
                block(blanks_builtin(Value::Int(value))).expect("integer length"),
                Value::CharArray(CharArray::new_row("   "))
            );
        }
        for value in [
            Value::Num(-3.0),
            Value::Int(IntValue::I8(-3)),
            Value::Int(IntValue::I16(-3)),
            Value::Int(IntValue::I32(-3)),
            Value::Int(IntValue::I64(-3)),
        ] {
            assert_eq!(
                block(blanks_builtin(value)).expect("negative length"),
                Value::CharArray(CharArray::new_row(""))
            );
        }
        let single = Tensor::from_f32(vec![3.0], vec![1, 1]).expect("single scalar");
        assert_eq!(
            block(blanks_builtin(Value::Tensor(single))).expect("single length"),
            Value::CharArray(CharArray::new_row("   "))
        );
        assert!(block(blanks_builtin(Value::Num(1.5))).is_err());
        assert!(block(blanks_builtin(Value::Int(IntValue::U64(u64::MAX)))).is_err());
    }

    #[test]
    fn blanks_gpu_input_follows_compatibility_mode() {
        use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U16(&[3]),
                    shape: &[1, 1],
                })
                .expect("upload integer length");
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = block(blanks_builtin(Value::GpuTensor(handle.clone())))
                    .expect_err("strict mode rejects resident length");
                assert_eq!(
                    error.identifier(),
                    BLANKS_GPU_INPUT_EXTENSION.error_identifier
                );
            }
            {
                let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
                assert_eq!(
                    block(blanks_builtin(Value::GpuTensor(handle))).expect("RunMat extension"),
                    Value::CharArray(CharArray::new_row("   "))
                );
            }
        });
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
        assert_eq!(
            block(isletter_builtin(Value::Int(runmat_value::IntValue::U64(
                u64::MAX,
            ))))
            .unwrap(),
            Value::Bool(false)
        );
        let integer = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::from(b'A'), u64::from(b'1'), u64::MAX]),
            vec![1, 3],
        )
        .unwrap();
        assert_eq!(
            block(isstrprop_builtin(
                Value::Tensor(integer),
                Value::String("alpha".into()),
                Vec::new(),
            ))
            .unwrap(),
            Value::LogicalArray(LogicalArray::new(vec![1, 0, 0], vec![1, 3]).unwrap())
        );
        assert_eq!(
            block(isspace_builtin(Value::Int(IntValue::U16(32)))).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn migrated_classifiers_reject_excess_outputs_before_execution() {
        let _outputs = crate::output_count::push_output_count(Some(2));
        let strncmpi_error = block(strncmpi_builtin(
            Value::String("ab".into()),
            Value::String("AB".into()),
            Value::Num(2.0),
        ))
        .expect_err("strncmpi has one output");
        assert_eq!(
            strncmpi_error.identifier(),
            STRNCMPI_ERROR_TOO_MANY_OUTPUTS.identifier
        );
        let isletter_error = block(isletter_builtin(Value::String("a".into())))
            .expect_err("isletter has one output");
        assert_eq!(
            isletter_error.identifier(),
            ISLETTER_ERROR_TOO_MANY_OUTPUTS.identifier
        );
        let isspace_error =
            block(isspace_builtin(Value::String(" ".into()))).expect_err("isspace has one output");
        assert_eq!(
            isspace_error.identifier(),
            ISSPACE_ERROR_TOO_MANY_OUTPUTS.identifier
        );
        let isstrprop_error = block(isstrprop_builtin(
            Value::String("a".into()),
            Value::String("alpha".into()),
            Vec::new(),
        ))
        .expect_err("isstrprop has one output");
        assert_eq!(
            isstrprop_error.identifier(),
            ISSTRPROP_ERROR_TOO_MANY_OUTPUTS.identifier
        );
    }

    #[test]
    fn strncmpi_short_prefix_and_missing_behavior_remain_explicit() {
        assert_eq!(
            block(strncmpi_builtin(
                Value::String("ab".into()),
                Value::String("AB".into()),
                Value::Num(3.0),
            ))
            .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            block(strncmpi_builtin(
                Value::String("<missing>".into()),
                Value::String("x".into()),
                Value::Num(0.0),
            ))
            .unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn strncmpi_gathers_automatic_resident_integer_count() {
        use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

        test_support::with_test_provider(|provider| {
            let count = [3_u64];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&count),
                    shape: &[1, 1],
                })
                .expect("upload exact count")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("automatic input");
            let result = block(strncmpi_builtin(
                Value::String("RunMat".into()),
                Value::String("runway".into()),
                Value::GpuTensor(handle.clone()),
            ))
            .expect("gather N");
            assert_eq!(result, Value::Bool(true));
            handle
                .allocation_lease()
                .validate_live()
                .expect("input remains live");
        });
    }

    #[test]
    fn isstrprop_only_gates_explicit_resident_integer_input() {
        use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U16(&[u16::from(b'A'), u16::from(b'1')]),
                    shape: &[1, 2],
                })
                .unwrap();
            {
                let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                assert_eq!(
                    block(isstrprop_builtin(
                        Value::GpuTensor(handle.clone()),
                        Value::String("alpha".into()),
                        Vec::new(),
                    ))
                    .unwrap(),
                    Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap())
                );
            }
            let handle = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("admit provenance view");
            {
                let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = block(isstrprop_builtin(
                    Value::GpuTensor(handle.clone()),
                    Value::String("alpha".into()),
                    Vec::new(),
                ))
                .unwrap_err();
                assert_eq!(
                    error.identifier(),
                    ISSTRPROP_RESIDENT_INPUT_EXTENSION.error_identifier
                );
            }
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert_eq!(
                block(isstrprop_builtin(
                    Value::GpuTensor(handle),
                    Value::String("alpha".into()),
                    Vec::new(),
                ))
                .unwrap(),
                Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap())
            );
        });
    }

    #[test]
    fn isstrprop_force_cell_output_wraps_numeric_result() {
        let result = block(isstrprop_builtin(
            Value::Int(IntValue::U16(u16::from(b'A'))),
            Value::String("alpha".into()),
            vec![Value::String("ForceCellOutput".into()), Value::Bool(true)],
        ))
        .expect("documented ForceCellOutput form");
        assert!(matches!(
            result,
            Value::Cell(cell)
                if cell.shape == vec![1, 1]
                    && cell.data == vec![Value::Bool(true)]
        ));
    }

    #[test]
    fn isstrprop_numeric_codes_use_unicode_categories_and_reject_surrogates() {
        let codes = Tensor::new_integer(
            IntegerStorage::U16(vec![0x0661, 0x2014, 0xd800]),
            vec![1, 3],
        )
        .expect("Unicode code units");
        let digit = block(isstrprop_builtin(
            Value::Tensor(codes.clone()),
            Value::String("digit".into()),
            Vec::new(),
        ))
        .expect("Unicode decimal classification");
        assert_eq!(
            digit,
            Value::LogicalArray(LogicalArray::new(vec![1, 0, 0], vec![1, 3]).unwrap())
        );
        let punctuation = block(isstrprop_builtin(
            Value::Tensor(codes),
            Value::String("punct".into()),
            Vec::new(),
        ))
        .expect("Unicode punctuation classification");
        assert_eq!(
            punctuation,
            Value::LogicalArray(LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap())
        );
    }

    #[test]
    fn isspace_accepts_string_scalars_and_rejects_string_arrays() {
        let scalar = StringArray::new(vec!["a b".into()], vec![1, 1]).expect("string scalar");
        assert_eq!(
            block(isspace_builtin(Value::StringArray(scalar))).expect("string scalar"),
            Value::LogicalArray(LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap())
        );

        let array = StringArray::new(vec![" ".into(), "x".into()], vec![1, 2])
            .expect("nonscalar string array");
        let error = block(isspace_builtin(Value::StringArray(array)))
            .expect_err("nonscalar string arrays reject");
        assert!(error.message().contains("string scalar"));
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
        let _guard = crate::output_count::push_output_count(Some(2));
        assert!(matches!(
            block(convert_chars_to_strings_builtin(
                Value::CharArray(CharArray::new_row("a")),
                vec![Value::CharArray(CharArray::new_row("b"))],
            ))
            .unwrap(),
            Value::OutputList(outputs) if outputs == vec![Value::String("a".into()), Value::String("b".into())]
        ));
        assert!(matches!(
            block(convert_contained_strings_to_chars_builtin(
                Value::String("a".into()),
                vec![Value::String("b".into())],
            ))
            .unwrap(),
            Value::OutputList(outputs) if outputs == vec![Value::CharArray(CharArray::new_row("a")), Value::CharArray(CharArray::new_row("b"))]
        ));
        drop(_guard);
        assert_eq!(
            block(convert_chars_to_strings_builtin(
                Value::CharArray(CharArray::new_row("abc")),
                Vec::new(),
            ))
            .unwrap(),
            Value::String("abc".into())
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
            Value::CharArray(CharArray::new_row("[1 2;3 4]"))
        );
    }

    #[test]
    fn conversion_output_selection_preserves_existing_padding_and_zero_output_behavior() {
        {
            let _outputs = crate::output_count::push_output_count(Some(3));
            let result = block(convert_strings_to_chars_builtin(
                Value::String("a".into()),
                vec![Value::String("b".into())],
            ))
            .unwrap();
            assert_eq!(
                result,
                Value::OutputList(vec![
                    Value::CharArray(CharArray::new_row("a")),
                    Value::CharArray(CharArray::new_row("b")),
                    Value::Num(0.0),
                ])
            );
        }
        {
            let _outputs = crate::output_count::push_output_count(Some(0));
            let result = block(convert_chars_to_strings_builtin(
                Value::CharArray(CharArray::new_row("a")),
                vec![Value::CharArray(CharArray::new_row("b"))],
            ))
            .unwrap();
            assert_eq!(result, Value::OutputList(Vec::new()));
        }
    }

    #[test]
    fn integer_conversion_passthrough_is_exact_for_every_class() {
        for int in [
            IntValue::I8(i8::MIN),
            IntValue::I16(i16::MIN),
            IntValue::I32(i32::MIN),
            IntValue::I64(i64::MIN),
            IntValue::U8(u8::MAX),
            IntValue::U16(u16::MAX),
            IntValue::U32(u32::MAX),
            IntValue::U64(u64::MAX),
        ] {
            let value = Value::Int(int);
            assert_eq!(
                block(convert_strings_to_chars_builtin(value.clone(), Vec::new())).unwrap(),
                value
            );
            assert_eq!(
                block(convert_chars_to_strings_builtin(value.clone(), Vec::new())).unwrap(),
                value
            );
            assert_eq!(
                block(convert_contained_strings_to_chars_builtin(
                    value.clone(),
                    Vec::new()
                ))
                .unwrap(),
                value
            );
        }
    }

    #[test]
    fn integer_conversion_passthrough_preserves_provider_ownership_without_gather() {
        use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[u64::MAX]),
                    shape: &[1, 1],
                })
                .expect("upload integer");
            for output in [
                block(convert_strings_to_chars_builtin(
                    Value::GpuTensor(handle.clone()),
                    Vec::new(),
                ))
                .unwrap(),
                block(convert_chars_to_strings_builtin(
                    Value::GpuTensor(handle.clone()),
                    Vec::new(),
                ))
                .unwrap(),
                block(convert_contained_strings_to_chars_builtin(
                    Value::GpuTensor(handle.clone()),
                    Vec::new(),
                ))
                .unwrap(),
            ] {
                let Value::GpuTensor(returned) = output else {
                    panic!("expected unchanged resident integer handle");
                };
                assert_eq!(returned.buffer_id(), handle.buffer_id());
                assert_eq!(returned.device_id(), handle.device_id());
                assert_eq!(returned.shape(), handle.shape());
                assert_eq!(returned.view_identity(), handle.view_identity());
                assert!(returned.shares_allocation_with(&handle));
                handle.allocation_lease().validate_live().unwrap();
            }
        });
    }

    #[test]
    fn conversion_scope_and_container_shapes_match_their_contracts() {
        let matrix = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        assert_eq!(
            block(convert_chars_to_strings_builtin(
                Value::CharArray(matrix),
                Vec::new(),
            ))
            .unwrap(),
            Value::String("acbd".into())
        );

        let cellstr = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("a")),
                Value::CharArray(CharArray::new_row("b")),
            ],
            1,
            2,
        )
        .unwrap();
        assert_eq!(
            block(convert_chars_to_strings_builtin(
                Value::Cell(cellstr),
                Vec::new(),
            ))
            .unwrap(),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap())
        );

        let mixed = Value::Cell(
            CellArray::new(
                vec![
                    Value::String("keep".into()),
                    Value::Int(IntValue::U64(u64::MAX)),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        assert_eq!(
            block(convert_strings_to_chars_builtin(mixed.clone(), Vec::new())).unwrap(),
            mixed
        );

        assert_eq!(
            block(convert_contained_strings_to_chars_builtin(
                Value::String("top-level".into()),
                Vec::new(),
            ))
            .unwrap(),
            Value::CharArray(CharArray::new_row("top-level"))
        );
        for text in ["", "<missing>"] {
            assert_eq!(
                block(convert_strings_to_chars_builtin(
                    Value::String(text.into()),
                    Vec::new()
                ))
                .unwrap(),
                Value::CharArray(CharArray::new(Vec::new(), 0, 0).unwrap())
            );
        }
        let converted = block(convert_contained_strings_to_chars_builtin(
            mixed,
            Vec::new(),
        ))
        .unwrap();
        let Value::Cell(cell) = converted else {
            panic!("expected cell");
        };
        assert_eq!(
            cell.data,
            vec![
                Value::CharArray(CharArray::new_row("keep")),
                Value::Int(IntValue::U64(u64::MAX))
            ]
        );

        let resident = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("admit unregistered resident fixture")
                .handle,
        );
        let nested = Value::Cell(CellArray::new(vec![resident.clone()], 1, 1).unwrap());
        let Value::Cell(preserved) = block(convert_contained_strings_to_chars_builtin(
            nested,
            Vec::new(),
        ))
        .unwrap() else {
            panic!("expected cell");
        };
        assert_eq!(preserved.data, vec![resident]);
    }

    #[test]
    fn convert_contained_strings_maps_typed_structure_arrays() {
        let mut first = StructValue::new();
        first.insert("label", Value::String("first".into()));
        let mut second = StructValue::new();
        second.insert("label", Value::String("second".into()));
        let array = runmat_value::StructArray::new(vec![first, second], vec![2, 1]).unwrap();

        let converted = block(convert_contained_strings_to_chars_builtin(
            Value::StructArray(array),
            Vec::new(),
        ))
        .unwrap();
        let Value::StructArray(converted) = converted else {
            panic!("expected structure array");
        };
        assert_eq!(converted.shape(), [2, 1]);
        assert!(matches!(
            converted.get_linear(0).unwrap().fields.get("label"),
            Some(Value::CharArray(value)) if value.data.iter().collect::<String>() == "first"
        ));
        assert!(matches!(
            converted.get_linear(1).unwrap().fields.get("label"),
            Some(Value::CharArray(value)) if value.data.iter().collect::<String>() == "second"
        ));
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
            Value::CharArray(CharArray::new_row("hi"))
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
        assert_eq!(
            pattern_regex(&block(digits_pattern_builtin(Vec::new())).unwrap(), "test").unwrap(),
            r"\d+"
        );
        let value = block(digits_pattern_builtin(vec![Value::Num(2.0)])).unwrap();
        assert_eq!(pattern_regex(&value, "test").unwrap(), "\\d{2}");
        let bounded = block(digits_pattern_builtin(vec![
            Value::Int(IntValue::U8(2)),
            Value::Int(IntValue::U16(4)),
        ]))
        .unwrap();
        assert_eq!(pattern_regex(&bounded, "test").unwrap(), r"\d{2,4}");
        let unbounded = block(digits_pattern_builtin(vec![
            Value::Int(IntValue::I8(3)),
            Value::Num(f64::INFINITY),
        ]))
        .unwrap();
        assert_eq!(pattern_regex(&unbounded, "test").unwrap(), r"\d{3,}");
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
            ".*?"
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
    fn pattern_constructors_reject_extra_outputs_before_reading_inputs() {
        let _outputs = crate::output_count::push_output_count(Some(2));
        for (result, identifier) in [
            (
                block(pattern_builtin(Value::Num(7.0))),
                PATTERN_ERROR_TOO_MANY_OUTPUTS.identifier,
            ),
            (
                block(regexp_pattern_builtin(Value::Num(7.0))),
                REGEXP_PATTERN_ERROR_TOO_MANY_OUTPUTS.identifier,
            ),
            (
                block(digits_pattern_builtin(vec![Value::Num(-1.0)])),
                DIGITS_PATTERN_ERROR_TOO_MANY_OUTPUTS.identifier,
            ),
            (
                block(letters_pattern_builtin(vec![Value::Num(-1.0)])),
                LETTERS_PATTERN_ERROR_TOO_MANY_OUTPUTS.identifier,
            ),
            (
                block(wildcard_pattern_builtin(vec![Value::Num(-1.0)])),
                WILDCARD_PATTERN_ERROR_TOO_MANY_OUTPUTS.identifier,
            ),
            (
                block(text_boundary_builtin(vec![Value::Num(7.0)])),
                TEXT_BOUNDARY_ERROR_TOO_MANY_OUTPUTS.identifier,
            ),
        ] {
            assert_eq!(result.expect_err("excess outputs").identifier(), identifier);
        }
    }

    #[test]
    fn digits_pattern_validates_range_controls_and_argument_count() {
        let zero = block(digits_pattern_builtin(vec![Value::Int(IntValue::U64(0))])).unwrap();
        assert_eq!(pattern_regex(&zero, "test").unwrap(), r"\d{0}");
        for args in [
            vec![Value::Num(-1.0)],
            vec![Value::Num(1.5)],
            vec![Value::Num(1.0), Value::Num(f64::NEG_INFINITY)],
            vec![Value::Bool(true)],
        ] {
            let error = block(digits_pattern_builtin(args)).unwrap_err();
            assert_eq!(
                error.identifier(),
                DIGITS_PATTERN_ERROR_INVALID_COUNT.identifier
            );
        }
        let range_error = block(digits_pattern_builtin(vec![
            Value::Num(4.0),
            Value::Num(3.0),
        ]))
        .unwrap_err();
        assert_eq!(
            range_error.identifier(),
            DIGITS_PATTERN_ERROR_INVALID_RANGE.identifier
        );
        let arity_error = block(digits_pattern_builtin(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
        ]))
        .unwrap_err();
        assert_eq!(
            arity_error.identifier(),
            DIGITS_PATTERN_ERROR_ARGUMENT_COUNT.identifier
        );
        let resident = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("admit unregistered resident fixture")
                .handle,
        );
        let resident_error = block(digits_pattern_builtin(vec![resident])).unwrap_err();
        assert_eq!(
            resident_error.identifier(),
            DIGITS_PATTERN_ERROR_INVALID_COUNT.identifier
        );
    }

    #[test]
    fn digits_pattern_range_reads_all_integer_classes_exactly() {
        for (minimum, maximum) in [
            (IntValue::I8(2), IntValue::I8(4)),
            (IntValue::I16(2), IntValue::I16(4)),
            (IntValue::I32(2), IntValue::I32(4)),
            (IntValue::I64(2), IntValue::I64(4)),
            (IntValue::U8(2), IntValue::U8(4)),
            (IntValue::U16(2), IntValue::U16(4)),
            (IntValue::U32(2), IntValue::U32(4)),
            (IntValue::U64(2), IntValue::U64(4)),
        ] {
            let pattern = block(digits_pattern_builtin(vec![
                Value::Int(minimum),
                Value::Int(maximum),
            ]))
            .unwrap();
            assert_eq!(pattern_regex(&pattern, "test").unwrap(), r"\d{2,4}");
        }
    }

    #[test]
    fn digits_pattern_descriptor_declares_all_public_forms() {
        assert_eq!(DIGITS_PATTERN_DESCRIPTOR.signatures.len(), 3);
        assert_eq!(DIGITS_PATTERN_DESCRIPTOR.errors.len(), 4);
        assert_eq!(DIGITS_PATTERN_INTEGER_CAPABILITIES.len(), 2);
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
