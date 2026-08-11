//! MATLAB text transformation compatibility helpers.

use regex::Regex;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CharArray, IntValue, NumericScalar, ResolveContext, StringArray,
    Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::tensor;
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::builtins::strings::core::compat::{
    broadcast_flat_index, broadcast_shape, logical_value, pattern_regex, scalar_text, text_items,
};
use crate::builtins::strings::text_analytics::documents::{
    erase_punctuation_tokenized_document, erase_urls_tokenized_document,
};
use crate::{build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult};

const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output value.",
}];

const OUT_NEW_STR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newStr",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text with matching content erased.",
}];

const OUT_NEW_DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newDocuments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated tokenized documents.",
}];

const OUT_BOOL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical result.",
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

const IN_TEXT_BOUNDARY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "text",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input string array, character vector, or cell array of character vectors.",
    },
    BuiltinParamDescriptor {
        name: "boundary",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Text/pattern boundary or positive numeric position, scalar or the same size as text.",
    },
];

const REST: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "arg",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Values to append.",
}];

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

descriptor!(APPEND_DESCRIPTOR, "s = append(arg1, ...)", &REST, &OUT_ANY);

pub const APPEND_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "append combines string arrays, character vectors, or cell arrays of character vectors. Numeric and integer values are not text inputs; all eight integer classes and resident numeric handles reject without implicit string conversion or provider gather.",
};
descriptor!(REVERSE_DESCRIPTOR, "s = reverse(text)", &IN_TEXT, &OUT_ANY);
descriptor!(DEBLANK_DESCRIPTOR, "s = deblank(text)", &IN_TEXT, &OUT_ANY);
pub const DEBLANK_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "deblank accepts string arrays, character arrays, or cell arrays of character vectors. Numeric and integer inputs reject without implicit text conversion or provider gather.",
};
descriptor!(
    STRJUST_DESCRIPTOR,
    "s = strjust(text, side)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    SPLITLINES_DESCRIPTOR,
    "s = splitlines(text)",
    &IN_TEXT,
    &OUT_ANY
);
const EXTRACT_BEFORE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.EXTRACT_BEFORE.INVALID_INPUT",
        identifier: Some("RunMat:extractBefore:InvalidInput"),
        when: "The call does not contain exactly one documented text input and one boundary.",
        message: "extractBefore: invalid input",
    },
    BuiltinErrorDescriptor {
        code: "RM.EXTRACT_BEFORE.INVALID_BOUNDARY",
        identifier: Some("RunMat:extractBefore:InvalidBoundary"),
        when: "A numeric boundary is not a positive integer or a text/pattern boundary is invalid.",
        message: "extractBefore: invalid boundary",
    },
    BuiltinErrorDescriptor {
        code: "RM.EXTRACT_BEFORE.SIZE_MISMATCH",
        identifier: Some("RunMat:extractBefore:SizeMismatch"),
        when: "A nonscalar boundary does not have the same shape as the text input.",
        message: "extractBefore: boundary size must match text",
    },
];
pub const EXTRACT_BEFORE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "s = extractBefore(text, boundary)",
        inputs: &IN_TEXT_BOUNDARY,
        outputs: &OUT_ANY,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EXTRACT_BEFORE_ERRORS,
};
const EXTRACT_AFTER_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.EXTRACT_AFTER.INVALID_INPUT",
        identifier: Some("RunMat:extractAfter:InvalidInput"),
        when: "The call does not contain exactly one documented text input and one boundary.",
        message: "extractAfter: invalid input",
    },
    BuiltinErrorDescriptor {
        code: "RM.EXTRACT_AFTER.INVALID_BOUNDARY",
        identifier: Some("RunMat:extractAfter:InvalidBoundary"),
        when: "A numeric boundary is not a positive integer or a text/pattern boundary is invalid.",
        message: "extractAfter: invalid boundary",
    },
    BuiltinErrorDescriptor {
        code: "RM.EXTRACT_AFTER.SIZE_MISMATCH",
        identifier: Some("RunMat:extractAfter:SizeMismatch"),
        when: "A nonscalar boundary does not have the same shape as the text input.",
        message: "extractAfter: boundary size must match text",
    },
];
pub const EXTRACT_AFTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "s = extractAfter(text, boundary)",
        inputs: &IN_TEXT_BOUNDARY,
        outputs: &OUT_ANY,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EXTRACT_AFTER_ERRORS,
};

const EXTRACT_BOUNDARY_POSITION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "pos",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The numeric position is a one-based structural control and accepts scalar or same-size arrays from every built-in integer class.",
    }];
const EXTRACT_BOUNDARY_TEXT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "str",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The first argument is text; integer data is rejected before provider access.",
    }];
pub const EXTRACT_BEFORE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    extract_boundary_position_capability("newStr = extractBefore(str, integer_pos)"),
    extract_boundary_text_capability("newStr = extractBefore(integer_str, boundary)"),
];
pub const EXTRACT_AFTER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    extract_boundary_position_capability("newStr = extractAfter(str, integer_pos)"),
    extract_boundary_text_capability("newStr = extractAfter(integer_str, boundary)"),
];

const fn extract_boundary_position_capability(
    form: &'static str,
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs: &EXTRACT_BOUNDARY_POSITION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Positions are read exactly with one-based indexing; the output preserves the input text container class and shape.",
    }
}

const fn extract_boundary_text_capability(
    form: &'static str,
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs: &EXTRACT_BOUNDARY_TEXT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer text input is outside the public text domain and rejects without numeric-to-text conversion.",
    }
}

const EXTRACT_BEFORE_RESIDENT_POSITION_EXTENSION: BuiltinExtensionDescriptor =
    extract_boundary_extension(
        "extractbefore-resident-position",
        "extractBefore with a resident numeric position is a RunMat extension",
        "RunMat:compatibility:ExtractBeforeResidentPositionExtension",
    );
const EXTRACT_BEFORE_CHAR_MATRIX_EXTENSION: BuiltinExtensionDescriptor = extract_boundary_extension(
    "extractbefore-char-matrix",
    "extractBefore row-wise character-matrix input is a RunMat extension",
    "RunMat:compatibility:ExtractBeforeCharMatrixExtension",
);
const EXTRACT_BEFORE_STRING_CELL_EXTENSION: BuiltinExtensionDescriptor = extract_boundary_extension(
    "extractbefore-string-cell",
    "extractBefore cells containing string scalars or nested cells are a RunMat extension",
    "RunMat:compatibility:ExtractBeforeStringCellExtension",
);
const EXTRACT_BEFORE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    EXTRACT_BEFORE_RESIDENT_POSITION_EXTENSION,
    EXTRACT_BEFORE_CHAR_MATRIX_EXTENSION,
    EXTRACT_BEFORE_STRING_CELL_EXTENSION,
];
const EXTRACT_AFTER_RESIDENT_POSITION_EXTENSION: BuiltinExtensionDescriptor =
    extract_boundary_extension(
        "extractafter-resident-position",
        "extractAfter with a resident numeric position is a RunMat extension",
        "RunMat:compatibility:ExtractAfterResidentPositionExtension",
    );
const EXTRACT_AFTER_CHAR_MATRIX_EXTENSION: BuiltinExtensionDescriptor = extract_boundary_extension(
    "extractafter-char-matrix",
    "extractAfter row-wise character-matrix input is a RunMat extension",
    "RunMat:compatibility:ExtractAfterCharMatrixExtension",
);
const EXTRACT_AFTER_STRING_CELL_EXTENSION: BuiltinExtensionDescriptor = extract_boundary_extension(
    "extractafter-string-cell",
    "extractAfter cells containing string scalars or nested cells are a RunMat extension",
    "RunMat:compatibility:ExtractAfterStringCellExtension",
);
const EXTRACT_AFTER_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    EXTRACT_AFTER_RESIDENT_POSITION_EXTENSION,
    EXTRACT_AFTER_CHAR_MATRIX_EXTENSION,
    EXTRACT_AFTER_STRING_CELL_EXTENSION,
];

const fn extract_boundary_extension(
    id: &'static str,
    description: &'static str,
    error_identifier: &'static str,
) -> BuiltinExtensionDescriptor {
    BuiltinExtensionDescriptor {
        id,
        mode: BuiltinExtensionMode::RunMatOnly,
        description,
        error_identifier: Some(error_identifier),
    }
}
descriptor!(
    INSERT_BEFORE_DESCRIPTOR,
    "s = insertBefore(text, boundary, newText)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    INSERT_AFTER_DESCRIPTOR,
    "s = insertAfter(text, boundary, newText)",
    &IN_TEXT_REST,
    &OUT_ANY
);
descriptor!(
    REPLACE_BETWEEN_DESCRIPTOR,
    "s = replaceBetween(text, start, end, newText)",
    &IN_TEXT_REST,
    &OUT_ANY
);
const ERASE_URLS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "newStr = eraseURLs(str)",
        inputs: &IN_TEXT,
        outputs: &OUT_NEW_STR,
    },
    BuiltinSignatureDescriptor {
        label: "newDocuments = eraseURLs(documents)",
        inputs: &IN_TEXT,
        outputs: &OUT_NEW_DOCUMENTS,
    },
];
const ERASE_PUNCTUATION_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "newStr = erasePunctuation(str)",
        inputs: &IN_TEXT,
        outputs: &OUT_NEW_STR,
    },
    BuiltinSignatureDescriptor {
        label: "newDocuments = erasePunctuation(documents)",
        inputs: &IN_TEXT,
        outputs: &OUT_NEW_DOCUMENTS,
    },
    BuiltinSignatureDescriptor {
        label: "newDocuments = erasePunctuation(documents, 'TokenTypes', types)",
        inputs: &IN_TEXT_REST,
        outputs: &OUT_NEW_DOCUMENTS,
    },
];
const ERASE_URLS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_URLS.INVALID_INPUT",
    identifier: Some("RunMat:eraseURLs:InvalidInput"),
    when: "Input is not documented host text or a tokenizedDocument object.",
    message: "eraseURLs: input must be text or tokenizedDocument",
};
const ERASE_PUNCTUATION_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_PUNCTUATION.INVALID_INPUT",
    identifier: Some("RunMat:erasePunctuation:InvalidInput"),
    when: "Input is not documented host text or a tokenizedDocument object.",
    message: "erasePunctuation: input must be text or tokenizedDocument",
};
const ERASE_PUNCTUATION_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_PUNCTUATION.INVALID_OPTION",
    identifier: Some("RunMat:erasePunctuation:InvalidOption"),
    when: "TokenTypes is malformed, empty, or used with non-document text.",
    message: "erasePunctuation: invalid TokenTypes option",
};
const ERASE_URLS_ERRORS: [BuiltinErrorDescriptor; 1] = [ERASE_URLS_ERROR_INVALID_INPUT];
const ERASE_PUNCTUATION_ERRORS: [BuiltinErrorDescriptor; 2] = [
    ERASE_PUNCTUATION_ERROR_INVALID_INPUT,
    ERASE_PUNCTUATION_ERROR_INVALID_OPTION,
];
pub const ERASE_URLS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ERASE_URLS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERASE_URLS_ERRORS,
};
pub const ERASE_PUNCTUATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ERASE_PUNCTUATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERASE_PUNCTUATION_ERRORS,
};
pub const ERASE_URLS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "eraseURLs accepts host text or tokenizedDocument input only. All eight integer classes, logical values, and resident numeric handles reject without numeric-to-text conversion or provider access.",
    };
pub const ERASE_PUNCTUATION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "erasePunctuation accepts host text, tokenizedDocument, and textual TokenTypes values only. All eight integer classes, logical values, and resident numeric handles reject without numeric-to-text conversion or provider access.",
    };
const ERASE_URLS_CHAR_MATRIX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "erase-urls-char-matrix-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "eraseURLs with a multirow character matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EraseURLsCharMatrixExtension"),
};
const ERASE_URLS_BROAD_CELL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "erase-urls-broad-cell-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "eraseURLs with nested cells or non-character cell elements is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EraseURLsBroadCellExtension"),
};
const ERASE_URLS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    ERASE_URLS_CHAR_MATRIX_EXTENSION,
    ERASE_URLS_BROAD_CELL_EXTENSION,
];
const ERASE_PUNCTUATION_CHAR_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "erase-punctuation-char-matrix-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "erasePunctuation with a multirow character matrix is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ErasePunctuationCharMatrixExtension"),
    };
const ERASE_PUNCTUATION_BROAD_CELL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "erase-punctuation-broad-cell-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "erasePunctuation with nested cells or non-character cell elements is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ErasePunctuationBroadCellExtension"),
    };
const ERASE_PUNCTUATION_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    ERASE_PUNCTUATION_CHAR_MATRIX_EXTENSION,
    ERASE_PUNCTUATION_BROAD_CELL_EXTENSION,
];
descriptor!(
    MATCHES_DESCRIPTOR,
    "tf = matches(text, pattern)",
    &IN_TEXT_REST,
    &OUT_BOOL
);

fn any_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

fn bool_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

fn transform_error(name: &str, message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

#[derive(Clone, Copy)]
enum ExtractErrorKind {
    InvalidInput,
    InvalidBoundary,
    SizeMismatch,
}

fn extract_error(
    name: &str,
    kind: ExtractErrorKind,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let descriptor = match (name, kind) {
        ("extractBefore", ExtractErrorKind::InvalidInput) => &EXTRACT_BEFORE_ERRORS[0],
        ("extractBefore", ExtractErrorKind::InvalidBoundary) => &EXTRACT_BEFORE_ERRORS[1],
        ("extractBefore", ExtractErrorKind::SizeMismatch) => &EXTRACT_BEFORE_ERRORS[2],
        ("extractAfter", ExtractErrorKind::InvalidInput) => &EXTRACT_AFTER_ERRORS[0],
        ("extractAfter", ExtractErrorKind::InvalidBoundary) => &EXTRACT_AFTER_ERRORS[1],
        ("extractAfter", ExtractErrorKind::SizeMismatch) => &EXTRACT_AFTER_ERRORS[2],
        _ => return transform_error(name, message),
    };
    let mut builder = build_runtime_error(message).with_builtin(name);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_flow(name: &'static str) -> impl Fn(crate::RuntimeError) -> crate::RuntimeError {
    move |err| map_control_flow_with_builtin(err, name)
}

#[runtime_builtin(
    name = "append",
    category = "strings/transform",
    summary = "Append text inputs elementwise.",
    keywords = "append,string,char,text,concatenate",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::APPEND_DESCRIPTOR),
    integer_audit(crate::builtins::strings::transform::compat::APPEND_INTEGER_AUDIT),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn append_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.is_empty() {
        return Ok(Value::String(String::new()));
    }
    let mut lists = Vec::with_capacity(rest.len());
    let mut output_kind = AppendOutputKind::Char;
    for value in rest {
        if matches!(value, Value::GpuTensor(_)) {
            return Err(transform_error(
                "append",
                "append: expected string, character vector, or cell array of character vectors",
            ));
        }
        let (list, kind) = append_text_items(value)?;
        output_kind = output_kind.combine(kind);
        lists.push(list);
    }
    let shape = lists
        .iter()
        .skip(1)
        .try_fold(lists[0].shape.clone(), |shape, list| {
            broadcast_shape(&shape, &list.shape, "append")
        })?;
    let total: usize = shape.iter().product();
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let mut text = String::new();
        for list in &lists {
            let source = broadcast_flat_index(idx, &shape, &list.shape);
            if let Some(part) = &list.items[source] {
                text.push_str(part);
            }
        }
        out.push(text);
    }
    append_output(out, shape, output_kind)
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum AppendOutputKind {
    Char,
    Cell,
    String,
}

impl AppendOutputKind {
    fn combine(self, other: Self) -> Self {
        self.max(other)
    }
}

fn append_text_items(
    value: Value,
) -> BuiltinResult<(
    crate::builtins::strings::core::compat::TextList,
    AppendOutputKind,
)> {
    match value {
        Value::String(text) => Ok((
            text_items(Value::String(text), "append")?,
            AppendOutputKind::String,
        )),
        Value::StringArray(array) => Ok((
            text_items(Value::StringArray(array), "append")?,
            AppendOutputKind::String,
        )),
        Value::Cell(cell) => {
            let mut items = Vec::with_capacity(cell.data.len());
            for value in cell.data {
                match value {
                    Value::CharArray(array) if array.rows <= 1 || array.cols <= 1 => {
                        items.push(Some(array.data.into_iter().collect::<String>()));
                    }
                    _ => {
                        return Err(transform_error(
                            "append",
                            "append: cell inputs must contain only character vectors",
                        ));
                    }
                }
            }
            Ok((
                crate::builtins::strings::core::compat::TextList {
                    items,
                    shape: cell.shape,
                },
                AppendOutputKind::Cell,
            ))
        }
        Value::CharArray(array) if array.rows <= 1 || array.cols <= 1 => {
            let text = array.data.into_iter().collect::<String>();
            Ok((
                text_items(Value::String(text), "append")?,
                AppendOutputKind::Char,
            ))
        }
        Value::CharArray(_) => Err(transform_error(
            "append",
            "append: character inputs must be vectors",
        )),
        other => Err(transform_error(
            "append",
            format!(
                "append: expected string, character vector, or cell array of character vectors, got {other:?}"
            ),
        )),
    }
}

fn append_output(
    values: Vec<String>,
    shape: Vec<usize>,
    kind: AppendOutputKind,
) -> BuiltinResult<Value> {
    match kind {
        AppendOutputKind::String => string_array_or_scalar(values, shape, "append"),
        AppendOutputKind::Cell => {
            let values = values
                .into_iter()
                .map(|text| char_rows(vec![text], "append"))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, shape).map_err(|error| transform_error("append", error))
        }
        AppendOutputKind::Char => {
            let text = values.into_iter().next().unwrap_or_default();
            char_rows(vec![text], "append")
        }
    }
}

#[runtime_builtin(
    name = "reverse",
    category = "strings/transform",
    summary = "Reverse characters in text values.",
    keywords = "reverse,string,char,text",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::REVERSE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn reverse_builtin(text: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("reverse"))?;
    map_text_preserve(text, "reverse", |s| s.chars().rev().collect())
}

#[runtime_builtin(
    name = "deblank",
    category = "strings/transform",
    summary = "Remove trailing whitespace from text.",
    keywords = "deblank,trailing whitespace,string,char,text",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::DEBLANK_DESCRIPTOR),
    integer_audit(crate::builtins::strings::transform::compat::DEBLANK_INTEGER_AUDIT),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn deblank_builtin(text: Value) -> BuiltinResult<Value> {
    if crate::dispatcher::value_contains_gpu(&text) {
        return Err(transform_error(
            "deblank",
            "deblank: expected string, character array, or cell array of character vectors",
        ));
    }
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("deblank"))?;
    map_text_preserve(text, "deblank", |s| {
        s.trim_end_matches(char::is_whitespace).to_string()
    })
}

#[runtime_builtin(
    name = "strjust",
    category = "strings/transform",
    summary = "Justify character rows left, right, or center.",
    keywords = "strjust,justify,char,text",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::STRJUST_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn strjust_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("strjust"))?;
    let side = if let Some(value) = rest.first() {
        let value = gather_if_needed_async(value)
            .await
            .map_err(map_flow("strjust"))?;
        scalar_text(&value, "strjust")?.to_ascii_lowercase()
    } else {
        "right".to_string()
    };
    map_text_preserve(text, "strjust", |s| justify(s, &side))
}

#[runtime_builtin(
    name = "splitlines",
    category = "strings/transform",
    summary = "Split text at newline boundaries.",
    keywords = "splitlines,newline,string,text",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::SPLITLINES_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn splitlines_builtin(text: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("splitlines"))?;
    match text {
        Value::String(text) => strings_from_lines(&text),
        Value::StringArray(array) if array.data.len() == 1 => strings_from_lines(&array.data[0]),
        Value::CharArray(array) if array.rows <= 1 => {
            strings_from_lines(&char_row_to_string_slice(&array.data, array.cols, 0))
        }
        other => {
            let list = text_items(other, "splitlines")?;
            let values = list
                .items
                .into_iter()
                .map(|text| strings_from_lines(&text.unwrap_or_default()))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, list.shape).map_err(|e| transform_error("splitlines", e))
        }
    }
}

#[runtime_builtin(
    name = "extractBefore",
    category = "strings/transform",
    summary = "Extract text before a position or boundary.",
    keywords = "extractBefore,string,text,boundary",
    accel = "sink",
    extensions(EXTRACT_BEFORE_EXTENSIONS),
    integer_capabilities(EXTRACT_BEFORE_INTEGER_CAPABILITIES),
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::EXTRACT_BEFORE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn extract_before_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    boundary_transform(
        text,
        rest,
        "extractBefore",
        &EXTRACT_BEFORE_RESIDENT_POSITION_EXTENSION,
        &EXTRACT_BEFORE_CHAR_MATRIX_EXTENSION,
        &EXTRACT_BEFORE_STRING_CELL_EXTENSION,
        |s, boundary| {
            let (start, _) = locate_boundary(s, boundary)?;
            Ok(s[..start].to_string())
        },
    )
    .await
}

#[runtime_builtin(
    name = "extractAfter",
    category = "strings/transform",
    summary = "Extract text after a position or boundary.",
    keywords = "extractAfter,string,text,boundary",
    accel = "sink",
    extensions(EXTRACT_AFTER_EXTENSIONS),
    integer_capabilities(EXTRACT_AFTER_INTEGER_CAPABILITIES),
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::EXTRACT_AFTER_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn extract_after_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    boundary_transform(
        text,
        rest,
        "extractAfter",
        &EXTRACT_AFTER_RESIDENT_POSITION_EXTENSION,
        &EXTRACT_AFTER_CHAR_MATRIX_EXTENSION,
        &EXTRACT_AFTER_STRING_CELL_EXTENSION,
        |s, boundary| {
            let (_, end) = locate_boundary(s, boundary)?;
            Ok(s[end..].to_string())
        },
    )
    .await
}

#[runtime_builtin(
    name = "insertBefore",
    category = "strings/transform",
    summary = "Insert text before a position or boundary.",
    keywords = "insertBefore,string,text,boundary",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::INSERT_BEFORE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn insert_before_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    insert_transform(text, rest, "insertBefore", false).await
}

#[runtime_builtin(
    name = "insertAfter",
    category = "strings/transform",
    summary = "Insert text after a position or boundary.",
    keywords = "insertAfter,string,text,boundary",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::INSERT_AFTER_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn insert_after_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    insert_transform(text, rest, "insertAfter", true).await
}

#[runtime_builtin(
    name = "replaceBetween",
    category = "strings/transform",
    summary = "Replace text between positions or boundary markers.",
    keywords = "replaceBetween,string,text,boundary",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::REPLACE_BETWEEN_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn replace_between_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() < 3 {
        return Err(transform_error(
            "replaceBetween",
            "replaceBetween: expected start, end, and replacement text",
        ));
    }
    let start = gather_if_needed_async(&rest[0])
        .await
        .map_err(map_flow("replaceBetween"))?;
    let stop = gather_if_needed_async(&rest[1])
        .await
        .map_err(map_flow("replaceBetween"))?;
    let replacement = gather_if_needed_async(&rest[2])
        .await
        .map_err(map_flow("replaceBetween"))?;
    let replacement = scalar_text(&replacement, "replaceBetween")?;
    let start = Boundary::from_value(&start, "replaceBetween")?;
    let stop = Boundary::from_value(&stop, "replaceBetween")?;
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("replaceBetween"))?;
    map_text_try_preserve(text, "replaceBetween", |s| {
        let (replace_start, replace_end) = replacement_span_between(s, &start, &stop)?;
        Ok(format!(
            "{}{}{}",
            &s[..replace_start],
            replacement,
            &s[replace_end..]
        ))
    })
}

#[runtime_builtin(
    name = "eraseURLs",
    category = "strings/transform",
    summary = "Remove URL substrings from text.",
    keywords = "eraseURLs,url,text analytics,string",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::ERASE_URLS_DESCRIPTOR),
    extensions(ERASE_URLS_EXTENSIONS),
    integer_audit(crate::builtins::strings::transform::compat::ERASE_URLS_INTEGER_AUDIT),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn erase_urls_builtin(text: Value) -> BuiltinResult<Value> {
    reject_resident_text_input(&text, "eraseURLs", &ERASE_URLS_ERROR_INVALID_INPUT)?;
    ensure_erase_text_extensions(
        &text,
        "eraseURLs",
        &ERASE_URLS_CHAR_MATRIX_EXTENSION,
        &ERASE_URLS_BROAD_CELL_EXTENSION,
    )?;
    if let Value::Object(object) = text {
        return erase_urls_tokenized_document(object);
    }
    if !is_host_text_tree(&text) {
        return Err(descriptor_transform_error(
            "eraseURLs",
            &ERASE_URLS_ERROR_INVALID_INPUT,
        ));
    }
    let regex = Regex::new(r"(?i)https?://[^\s]+")
        .map_err(|e| transform_error("eraseURLs", e.to_string()))?;
    map_text_preserve(text, "eraseURLs", |s| regex.replace_all(s, "").to_string())
}

#[runtime_builtin(
    name = "erasePunctuation",
    category = "strings/transform",
    summary = "Remove Unicode punctuation and symbol characters from text.",
    keywords = "erasePunctuation,punctuation,symbol,text analytics,string",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::transform::compat::ERASE_PUNCTUATION_DESCRIPTOR),
    extensions(ERASE_PUNCTUATION_EXTENSIONS),
    integer_audit(crate::builtins::strings::transform::compat::ERASE_PUNCTUATION_INTEGER_AUDIT),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn erase_punctuation_builtin(text: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    reject_resident_text_input(
        &text,
        "erasePunctuation",
        &ERASE_PUNCTUATION_ERROR_INVALID_INPUT,
    )?;
    for value in &rest {
        reject_resident_text_input(
            value,
            "erasePunctuation",
            &ERASE_PUNCTUATION_ERROR_INVALID_OPTION,
        )?;
    }
    ensure_erase_text_extensions(
        &text,
        "erasePunctuation",
        &ERASE_PUNCTUATION_CHAR_MATRIX_EXTENSION,
        &ERASE_PUNCTUATION_BROAD_CELL_EXTENSION,
    )?;
    if let Value::Object(object) = text {
        return erase_punctuation_tokenized_document(object, rest);
    }
    if !is_host_text_tree(&text) {
        return Err(descriptor_transform_error(
            "erasePunctuation",
            &ERASE_PUNCTUATION_ERROR_INVALID_INPUT,
        ));
    }
    if !rest.is_empty() {
        return Err(transform_error(
            "erasePunctuation",
            "erasePunctuation: name-value options are only supported for tokenizedDocument input",
        ));
    }
    let regex = Regex::new(r"[\p{P}\p{S}]")
        .map_err(|e| transform_error("erasePunctuation", e.to_string()))?;
    map_text_preserve(text, "erasePunctuation", |s| {
        regex.replace_all(s, "").to_string()
    })
}

fn reject_resident_text_input(
    value: &Value,
    name: &str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<()> {
    if crate::dispatcher::value_contains_gpu(value) {
        let mut builder = build_runtime_error(error.message).with_builtin(name);
        if let Some(identifier) = error.identifier {
            builder = builder.with_identifier(identifier);
        }
        return Err(builder.build());
    }
    Ok(())
}

fn descriptor_transform_error(
    name: &str,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(name);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn is_host_text_tree(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => true,
        Value::Cell(cell) => cell.data.iter().all(is_host_text_tree),
        _ => false,
    }
}

fn ensure_erase_text_extensions(
    value: &Value,
    builtin: &str,
    char_matrix: &'static BuiltinExtensionDescriptor,
    broad_cell: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    match value {
        Value::CharArray(array) if array.rows > 1 && array.cols > 1 => {
            crate::compatibility::ensure_builtin_extension_enabled(char_matrix, builtin)
        }
        Value::Cell(cell) if !cell.data.iter().all(is_cellstr_element) => {
            crate::compatibility::ensure_builtin_extension_enabled(broad_cell, builtin)
        }
        _ => Ok(()),
    }
}

fn is_cellstr_element(value: &Value) -> bool {
    matches!(value, Value::CharArray(array) if array.rows <= 1 || array.cols <= 1)
}

#[runtime_builtin(
    name = "matches",
    category = "strings/search",
    summary = "Return true when text fully matches a pattern.",
    keywords = "matches,string pattern,text,regular expression",
    accel = "sink",
    type_resolver(bool_type),
    descriptor(crate::builtins::strings::transform::compat::MATCHES_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::transform::compat"
)]
async fn matches_builtin(text: Value, pattern: Value) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow("matches"))?;
    let pattern = gather_if_needed_async(&pattern)
        .await
        .map_err(map_flow("matches"))?;
    let pattern = pattern_regex(&pattern, "matches")?;
    let regex = Regex::new(&format!("^(?:{pattern})$"))
        .map_err(|e| transform_error("matches", e.to_string()))?;
    let list = text_items(text, "matches")?;
    let out = list
        .items
        .iter()
        .map(|text| u8::from(text.as_ref().is_some_and(|text| regex.is_match(text))))
        .collect::<Vec<_>>();
    logical_value(out, list.shape, "matches")
}

async fn boundary_transform(
    text: Value,
    rest: Vec<Value>,
    fn_name: &'static str,
    resident_extension: &'static BuiltinExtensionDescriptor,
    char_matrix_extension: &'static BuiltinExtensionDescriptor,
    string_cell_extension: &'static BuiltinExtensionDescriptor,
    op: impl Fn(&str, &Boundary) -> BuiltinResult<String> + Copy,
) -> BuiltinResult<Value> {
    if rest.len() != 1 {
        return Err(extract_error(
            fn_name,
            ExtractErrorKind::InvalidInput,
            format!("{fn_name}: expected exactly text and one boundary argument"),
        ));
    }
    if numeric_or_resident_value(&text) || contains_numeric_or_resident(&text) {
        return Err(extract_error(
            fn_name,
            ExtractErrorKind::InvalidInput,
            format!("{fn_name}: expected text input"),
        ));
    }
    if contains_resident_value(&rest[0]) {
        crate::compatibility::ensure_builtin_extension_enabled(resident_extension, fn_name)?;
    }
    if is_multirow_char(&text) || is_multirow_char(&rest[0]) {
        crate::compatibility::ensure_builtin_extension_enabled(char_matrix_extension, fn_name)?;
    }
    if contains_string_or_nested_cell(&text) || contains_string_or_nested_cell(&rest[0]) {
        crate::compatibility::ensure_builtin_extension_enabled(string_cell_extension, fn_name)?;
    }
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow(fn_name))?;
    if !valid_extract_text_input(&text, false) {
        return Err(extract_error(
            fn_name,
            ExtractErrorKind::InvalidInput,
            format!("{fn_name}: expected a supported text container"),
        ));
    }
    let boundary = gather_if_needed_async(&rest[0])
        .await
        .map_err(map_flow(fn_name))?;
    let boundaries = BoundaryList::from_value(&boundary, fn_name).map_err(|error| {
        extract_error(
            fn_name,
            ExtractErrorKind::InvalidBoundary,
            error.to_string(),
        )
    })?;
    let text_shape = boundary_text_shape(&text).ok_or_else(|| {
        extract_error(
            fn_name,
            ExtractErrorKind::InvalidInput,
            format!("{fn_name}: expected text input"),
        )
    })?;
    if !shape_is_scalar_or_same(&boundaries.shape, &text_shape) {
        return Err(extract_error(
            fn_name,
            ExtractErrorKind::SizeMismatch,
            format!("{fn_name}: boundary size must be scalar or match the text input"),
        ));
    }
    map_text_with_boundaries(text, &boundaries, fn_name, op).map_err(|error| {
        extract_error(
            fn_name,
            ExtractErrorKind::InvalidBoundary,
            error.to_string(),
        )
    })
}

async fn insert_transform(
    text: Value,
    rest: Vec<Value>,
    fn_name: &'static str,
    after: bool,
) -> BuiltinResult<Value> {
    if rest.len() < 2 {
        return Err(transform_error(
            fn_name,
            format!("{fn_name}: expected boundary and new text"),
        ));
    }
    let boundary = gather_if_needed_async(&rest[0])
        .await
        .map_err(map_flow(fn_name))?;
    let new_text = gather_if_needed_async(&rest[1])
        .await
        .map_err(map_flow(fn_name))?;
    let boundary = Boundary::from_value(&boundary, fn_name)?;
    let new_text = scalar_text(&new_text, fn_name)?;
    let text = gather_if_needed_async(&text)
        .await
        .map_err(map_flow(fn_name))?;
    map_text_try_preserve(text, fn_name, |s| {
        let (start, end) = locate_boundary(s, &boundary)?;
        let idx = if after { end } else { start };
        Ok(format!("{}{}{}", &s[..idx], new_text, &s[idx..]))
    })
}

#[derive(Clone)]
enum Boundary {
    Position(usize),
    Text(String),
    Pattern(String),
}

#[derive(Clone)]
struct BoundaryList {
    data: Vec<Boundary>,
    shape: Vec<usize>,
}

impl BoundaryList {
    fn from_value(value: &Value, fn_name: &str) -> BuiltinResult<Self> {
        match value {
            Value::Num(value) => Ok(Self {
                data: vec![Boundary::Position(parse_boundary_float(*value, fn_name)?)],
                shape: vec![1, 1],
            }),
            Value::Int(value) => Ok(Self {
                data: vec![Boundary::Position(parse_boundary_integer(
                    value.clone(),
                    fn_name,
                )?)],
                shape: vec![1, 1],
            }),
            Value::Tensor(tensor) => {
                let mut data = Vec::with_capacity(tensor.len());
                for idx in 0..tensor.len() {
                    let value = tensor.numeric_value_at(idx).ok_or_else(|| {
                        transform_error(
                            fn_name,
                            format!("{fn_name}: numeric boundaries must be positive integers"),
                        )
                    })?;
                    let position = match value {
                        NumericScalar::F64(value) => parse_boundary_float(value, fn_name)?,
                        NumericScalar::F32(value) => {
                            parse_boundary_float(f64::from(value), fn_name)?
                        }
                        integer => parse_boundary_integer(
                            integer
                                .into_int_value()
                                .expect("non-floating numeric scalar is integer"),
                            fn_name,
                        )?,
                    };
                    data.push(Boundary::Position(position));
                }
                Ok(Self {
                    data,
                    shape: tensor.shape.clone(),
                })
            }
            Value::String(text) => Ok(Self {
                data: vec![Boundary::Text(text.clone())],
                shape: vec![1, 1],
            }),
            Value::StringArray(array) => Ok(Self {
                data: array.data.iter().cloned().map(Boundary::Text).collect(),
                shape: array.shape.clone(),
            }),
            Value::CharArray(array) => {
                let data = if array.rows == 0 {
                    vec![Boundary::Text(String::new())]
                } else {
                    (0..array.rows)
                        .map(|row| {
                            Boundary::Text(char_row_to_string_slice(&array.data, array.cols, row))
                        })
                        .collect()
                };
                Ok(Self {
                    data,
                    shape: if array.rows <= 1 {
                        vec![1, 1]
                    } else {
                        vec![array.rows, 1]
                    },
                })
            }
            Value::Cell(cell) => {
                let mut data = Vec::with_capacity(cell.data.len());
                for value in &cell.data {
                    data.push(Boundary::Text(scalar_text(value, fn_name)?));
                }
                Ok(Self {
                    data,
                    shape: cell.shape.clone(),
                })
            }
            Value::Object(_) => Ok(Self {
                data: vec![Boundary::Pattern(pattern_regex(value, fn_name)?)],
                shape: vec![1, 1],
            }),
            other => Err(transform_error(
                fn_name,
                format!("{fn_name}: expected a text, pattern, or numeric boundary, got {other:?}"),
            )),
        }
    }

    fn at(&self, idx: usize) -> &Boundary {
        if self.data.len() == 1 {
            &self.data[0]
        } else {
            &self.data[idx]
        }
    }
}

impl Boundary {
    fn from_value(value: &Value, fn_name: &str) -> BuiltinResult<Self> {
        if let Some(integer) = tensor::scalar_integer_value(value) {
            return parse_boundary_integer(integer, fn_name).map(Self::Position);
        }
        match value {
            Value::Num(n) if n.is_finite() && *n > 0.0 && n.fract() == 0.0 => {
                parse_boundary_float(*n, fn_name).map(Self::Position)
            }
            Value::Num(_) => Err(transform_error(
                fn_name,
                format!("{fn_name}: numeric boundaries must be positive integer scalars"),
            )),
            Value::Object(_) => Ok(Self::Pattern(pattern_regex(value, fn_name)?)),
            _ => Ok(Self::Text(scalar_text(value, fn_name)?)),
        }
    }
}

fn parse_boundary_integer(value: IntValue, fn_name: &str) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .filter(|position| *position > 0)
        .ok_or_else(|| {
            transform_error(
                fn_name,
                format!("{fn_name}: numeric boundaries must be positive integer scalars"),
            )
        })
}

fn parse_boundary_float(value: f64, fn_name: &str) -> BuiltinResult<usize> {
    if value > usize::MAX.saturating_sub(1) as f64 {
        return Err(transform_error(
            fn_name,
            format!("{fn_name}: numeric boundaries must be positive integer scalars"),
        ));
    }
    let parsed = value as usize;
    if parsed == 0 || parsed as f64 != value || parsed == usize::MAX {
        return Err(transform_error(
            fn_name,
            format!("{fn_name}: numeric boundaries must be positive integer scalars"),
        ));
    }
    Ok(parsed)
}

fn locate_boundary(text: &str, boundary: &Boundary) -> BuiltinResult<(usize, usize)> {
    match boundary {
        Boundary::Position(pos) => {
            let char_len = text.chars().count();
            if *pos > char_len.saturating_add(1) {
                return Err(transform_error(
                    "text boundary",
                    format!("boundary position {pos} exceeds text length {char_len}"),
                ));
            }
            let start = byte_index_for_char_position(text, *pos);
            let end = if *pos > char_len {
                text.len()
            } else {
                byte_index_after_char_position(text, *pos)
            };
            Ok((start, end))
        }
        Boundary::Text(needle) => text
            .find(needle)
            .map(|idx| (idx, idx + needle.len()))
            .ok_or_else(|| transform_error("text boundary", "boundary not found")),
        Boundary::Pattern(pattern) => Regex::new(pattern)
            .map_err(|e| transform_error("text boundary", e.to_string()))?
            .find(text)
            .map(|m| (m.start(), m.end()))
            .ok_or_else(|| transform_error("text boundary", "boundary not found")),
    }
}

fn numeric_or_resident_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::Complex(_, _)
            | Value::ComplexTensor(_)
            | Value::GpuTensor(_)
    )
}

fn contains_numeric_or_resident(value: &Value) -> bool {
    match value {
        Value::Cell(cell) => cell
            .data
            .iter()
            .any(|value| numeric_or_resident_value(value) || contains_numeric_or_resident(value)),
        _ => false,
    }
}

fn contains_resident_value(value: &Value) -> bool {
    match value {
        Value::GpuTensor(_) => true,
        Value::Cell(cell) => cell.data.iter().any(contains_resident_value),
        _ => false,
    }
}

fn is_multirow_char(value: &Value) -> bool {
    matches!(value, Value::CharArray(array) if array.rows > 1)
}

fn contains_string_or_nested_cell(value: &Value) -> bool {
    match value {
        Value::Cell(cell) => cell.data.iter().any(|value| {
            matches!(
                value,
                Value::String(_) | Value::StringArray(_) | Value::Cell(_)
            ) || contains_string_or_nested_cell(value)
        }),
        _ => false,
    }
}

fn valid_extract_text_input(value: &Value, nested: bool) -> bool {
    match value {
        Value::String(_) => true,
        Value::StringArray(array) => !nested || array.data.len() == 1,
        Value::CharArray(array) => !nested || array.rows <= 1,
        Value::Cell(cell) => cell
            .data
            .iter()
            .all(|value| valid_extract_text_input(value, true)),
        _ => false,
    }
}

fn boundary_text_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::String(_) => Some(vec![1, 1]),
        Value::StringArray(array) => Some(array.shape.clone()),
        Value::CharArray(array) if array.rows <= 1 => Some(vec![1, 1]),
        Value::CharArray(array) => Some(vec![array.rows, 1]),
        Value::Cell(cell) => Some(cell.shape.clone()),
        _ => None,
    }
}

fn shape_is_scalar_or_same(shape: &[usize], text_shape: &[usize]) -> bool {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        == Some(1)
        || shape == text_shape
}

fn map_text_with_boundaries(
    value: Value,
    boundaries: &BoundaryList,
    fn_name: &str,
    map: impl Fn(&str, &Boundary) -> BuiltinResult<String> + Copy,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => {
            if is_missing_string(&text) {
                Ok(Value::String(text))
            } else {
                Ok(Value::String(map(&text, boundaries.at(0))?))
            }
        }
        Value::StringArray(array) => StringArray::new(
            array
                .data
                .into_iter()
                .enumerate()
                .map(|(idx, text)| {
                    if is_missing_string(&text) {
                        Ok(text)
                    } else {
                        map(&text, boundaries.at(idx))
                    }
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
            array.shape,
        )
        .map(Value::StringArray)
        .map_err(|e| transform_error(fn_name, e)),
        Value::CharArray(array) => {
            let rows = (0..array.rows)
                .map(|row| {
                    map(
                        &char_row_to_string_slice(&array.data, array.cols, row),
                        boundaries.at(row),
                    )
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            char_rows(rows, fn_name)
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .enumerate()
                .map(|(idx, value)| map_text_cell_element(value, boundaries.at(idx), fn_name, map))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape).map_err(|e| transform_error(fn_name, e))
        }
        other => Err(transform_error(
            fn_name,
            format!("{fn_name}: expected text input, got {other:?}"),
        )),
    }
}

fn map_text_cell_element(
    value: Value,
    boundary: &Boundary,
    fn_name: &str,
    map: impl Fn(&str, &Boundary) -> BuiltinResult<String> + Copy,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => map(&text, boundary).map(Value::String),
        Value::StringArray(array) if array.data.len() == 1 => {
            map(&array.data[0], boundary).map(Value::String)
        }
        Value::CharArray(array) if array.rows <= 1 => {
            let text = if array.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(&array.data, array.cols, 0)
            };
            map(&text, boundary).map(|text| Value::CharArray(CharArray::new_row(&text)))
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .map(|value| map_text_cell_element(value, boundary, fn_name, map))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape).map_err(|e| transform_error(fn_name, e))
        }
        other => Err(transform_error(
            fn_name,
            format!("{fn_name}: expected character vectors in cell input, got {other:?}"),
        )),
    }
}

fn replacement_span_between(
    text: &str,
    start: &Boundary,
    stop: &Boundary,
) -> BuiltinResult<(usize, usize)> {
    match (start, stop) {
        (Boundary::Position(start), Boundary::Position(stop)) => {
            if start > stop {
                return Err(transform_error(
                    "replaceBetween",
                    "replaceBetween: start position must be less than or equal to end position",
                ));
            }
            let char_len = text.chars().count();
            if *stop > char_len {
                return Err(transform_error(
                    "replaceBetween",
                    format!("replaceBetween: end position {stop} exceeds text length {char_len}"),
                ));
            }
            Ok((
                byte_index_for_char_position(text, *start),
                byte_index_after_char_position(text, *stop),
            ))
        }
        _ => {
            let (_, start_end) = locate_boundary(text, start)?;
            let (stop_start, _) = locate_boundary(&text[start_end..], stop)
                .map(|(a, b)| (a + start_end, b + start_end))?;
            Ok((start_end, stop_start))
        }
    }
}

fn byte_index_for_char_position(text: &str, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    text.char_indices()
        .nth(pos.saturating_sub(1))
        .map(|(idx, _)| idx)
        .unwrap_or(text.len())
}

fn byte_index_after_char_position(text: &str, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    text.char_indices()
        .nth(pos)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len())
}

fn map_text_preserve(
    value: Value,
    fn_name: &str,
    map: impl Fn(&str) -> String + Copy,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => {
            if is_missing_string(&text) {
                Ok(Value::String(text))
            } else {
                Ok(Value::String(map(&text)))
            }
        }
        Value::StringArray(array) => StringArray::new(
            array
                .data
                .into_iter()
                .map(|text| {
                    if is_missing_string(&text) {
                        text
                    } else {
                        map(&text)
                    }
                })
                .collect(),
            array.shape,
        )
        .map(Value::StringArray)
        .map_err(|e| transform_error(fn_name, e)),
        Value::CharArray(array) => {
            let rows = (0..array.rows)
                .map(|row| map(&char_row_to_string_slice(&array.data, array.cols, row)))
                .collect::<Vec<_>>();
            char_rows(rows, fn_name)
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .map(|value| map_text_preserve(value, fn_name, map))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape).map_err(|e| transform_error(fn_name, e))
        }
        other => Err(transform_error(
            fn_name,
            format!("{fn_name}: expected text input, got {other:?}"),
        )),
    }
}

fn map_text_try_preserve(
    value: Value,
    fn_name: &str,
    map: impl Fn(&str) -> BuiltinResult<String> + Copy,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => {
            if is_missing_string(&text) {
                Ok(Value::String(text))
            } else {
                Ok(Value::String(map(&text)?))
            }
        }
        Value::StringArray(array) => StringArray::new(
            array
                .data
                .into_iter()
                .map(|text| {
                    if is_missing_string(&text) {
                        Ok(text)
                    } else {
                        map(&text)
                    }
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
            array.shape,
        )
        .map(Value::StringArray)
        .map_err(|e| transform_error(fn_name, e)),
        Value::CharArray(array) => {
            let rows = (0..array.rows)
                .map(|row| map(&char_row_to_string_slice(&array.data, array.cols, row)))
                .collect::<BuiltinResult<Vec<_>>>()?;
            char_rows(rows, fn_name)
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .map(|value| map_text_try_preserve(value, fn_name, map))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape).map_err(|e| transform_error(fn_name, e))
        }
        other => Err(transform_error(
            fn_name,
            format!("{fn_name}: expected text input, got {other:?}"),
        )),
    }
}

fn char_rows(rows: Vec<String>, fn_name: &str) -> BuiltinResult<Value> {
    let row_count = rows.len();
    let cols = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    let mut data = Vec::with_capacity(row_count * cols);
    for row in rows {
        let mut chars = row.chars().collect::<Vec<_>>();
        chars.resize(cols, ' ');
        data.extend(chars);
    }
    CharArray::new(data, row_count, cols)
        .map(Value::CharArray)
        .map_err(|e| transform_error(fn_name, e))
}

fn string_array_or_scalar(
    values: Vec<String>,
    shape: Vec<usize>,
    fn_name: &str,
) -> BuiltinResult<Value> {
    if values.len() == 1 {
        Ok(Value::String(values.into_iter().next().unwrap()))
    } else {
        StringArray::new(values, shape)
            .map(Value::StringArray)
            .map_err(|e| transform_error(fn_name, e))
    }
}

fn strings_from_lines(text: &str) -> BuiltinResult<Value> {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let lines = normalized
        .split('\n')
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = lines.len();
    StringArray::new(lines, vec![rows, 1])
        .map(Value::StringArray)
        .map_err(|e| transform_error("splitlines", e))
}

fn justify(text: &str, side: &str) -> String {
    let width = text.chars().count();
    let trimmed = text.trim().to_string();
    let pad = width.saturating_sub(trimmed.chars().count());
    match side {
        "left" => format!("{trimmed}{}", " ".repeat(pad)),
        "center" | "centre" => {
            let left = pad / 2;
            let right = pad - left;
            format!("{}{}{}", " ".repeat(left), trimmed, " ".repeat(right))
        }
        _ => format!("{}{}", " ".repeat(pad), trimmed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{
        BuiltinIntegerAuditKind, CellArray, IntValue, IntegerStorage, StructValue, Tensor,
    };

    fn block(
        value: impl std::future::Future<Output = BuiltinResult<Value>>,
    ) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
    }

    #[test]
    fn append_extract_insert_and_replace_work() {
        assert_eq!(
            block(append_builtin(vec![
                Value::String("run".into()),
                Value::String("mat".into())
            ]))
            .unwrap(),
            Value::String("runmat".into())
        );
        assert_eq!(
            block(extract_before_builtin(
                Value::String("alpha.beta".into()),
                vec![Value::String(".".into())],
            ))
            .unwrap(),
            Value::String("alpha".into())
        );
        assert_eq!(
            block(insert_after_builtin(
                Value::String("run".into()),
                vec![Value::Num(3.0), Value::String("mat".into())],
            ))
            .unwrap(),
            Value::String("runmat".into())
        );
        assert_eq!(
            block(extract_after_builtin(
                Value::String("alpha.beta".into()),
                vec![Value::String(".".into())],
            ))
            .unwrap(),
            Value::String("beta".into())
        );
        assert_eq!(
            block(insert_before_builtin(
                Value::String("run".into()),
                vec![Value::Num(1.0), Value::String("pre".into())],
            ))
            .unwrap(),
            Value::String("prerun".into())
        );
        assert_eq!(
            block(replace_between_builtin(
                Value::String("a[old]b".into()),
                vec![
                    Value::String("[".into()),
                    Value::String("]".into()),
                    Value::String("new".into()),
                ],
            ))
            .unwrap(),
            Value::String("a[new]b".into())
        );
    }

    #[test]
    fn append_integer_audit_is_explicitly_inapplicable() {
        assert_eq!(
            APPEND_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        assert!(APPEND_INTEGER_AUDIT.canonical_builtin.is_none());
        assert!(APPEND_INTEGER_AUDIT.notes.contains("all eight integer"));
    }

    #[test]
    fn append_preserves_documented_text_output_precedence_and_whitespace() {
        assert_eq!(
            block(append_builtin(vec![
                Value::CharArray(CharArray::new_row("Hello ")),
                Value::CharArray(CharArray::new_row("World")),
            ]))
            .expect("all-char append"),
            Value::CharArray(CharArray::new_row("Hello World"))
        );

        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("alpha")),
                Value::CharArray(CharArray::new_row("beta")),
            ],
            1,
            2,
        )
        .expect("cellstr");
        let result = block(append_builtin(vec![
            Value::Cell(cell.clone()),
            Value::CharArray(CharArray::new_row(" ")),
        ]))
        .expect("cell append");
        let Value::Cell(cell_result) = result else {
            panic!("expected cellstr output");
        };
        assert_eq!(cell_result.shape, vec![1, 2]);
        assert_eq!(
            cell_result.data,
            vec![
                Value::CharArray(CharArray::new_row("alpha ")),
                Value::CharArray(CharArray::new_row("beta ")),
            ]
        );

        let strings = StringArray::new(vec!["A".into(), "B".into()], vec![2, 1]).expect("strings");
        let result = block(append_builtin(vec![
            Value::StringArray(strings),
            Value::Cell(cell),
        ]))
        .expect("string-cell append");
        let Value::StringArray(string_result) = result else {
            panic!("expected string-array output");
        };
        assert_eq!(string_result.shape, vec![2, 2]);
        assert_eq!(
            string_result.data,
            vec!["Aalpha", "Balpha", "Abeta", "Bbeta"]
        );
    }

    #[test]
    fn append_rejects_every_integer_class_without_numeric_conversion() {
        let cases = [
            ("int8", IntValue::I8(-1), IntegerStorage::I8(vec![-1, 2])),
            ("int16", IntValue::I16(-2), IntegerStorage::I16(vec![-2, 3])),
            ("int32", IntValue::I32(-3), IntegerStorage::I32(vec![-3, 4])),
            (
                "int64",
                IntValue::I64(i64::MIN),
                IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            ),
            (
                "uint8",
                IntValue::U8(5),
                IntegerStorage::U8(vec![5, u8::MAX]),
            ),
            (
                "uint16",
                IntValue::U16(6),
                IntegerStorage::U16(vec![6, u16::MAX]),
            ),
            (
                "uint32",
                IntValue::U32(7),
                IntegerStorage::U32(vec![7, u32::MAX]),
            ),
            (
                "uint64",
                IntValue::U64(u64::MAX),
                IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            ),
        ];
        for (class, scalar, storage) in cases {
            for value in [
                Value::Int(scalar),
                Value::Tensor(Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor")),
            ] {
                let error = block(append_builtin(vec![
                    value,
                    Value::CharArray(CharArray::new_row("suffix")),
                ]))
                .expect_err("numeric append input must reject");
                assert!(
                    error.message().contains(
                        "expected string, character vector, or cell array of character vectors"
                    ),
                    "{class}: {error}"
                );
            }
        }

        for value in [Value::Num(1.0), Value::Bool(true)] {
            let error = block(append_builtin(vec![
                value,
                Value::CharArray(CharArray::new_row("suffix")),
            ]))
            .expect_err("nontext append input must reject");
            assert!(error.message().contains("expected string"), "{error}");
        }

        let invalid_cell =
            CellArray::new(vec![Value::String("not a character vector".into())], 1, 1)
                .expect("cell");
        let error = block(append_builtin(vec![Value::Cell(invalid_cell)]))
            .expect_err("non-cellstr input must reject");
        assert!(
            error
                .message()
                .contains("must contain only character vectors"),
            "{error}"
        );
    }

    #[test]
    fn append_rejects_resident_integer_handles_before_provider_gather() {
        for (offset, element_type) in [
            (0, runmat_accelerate_api::IntegerElementType::I8),
            (1, runmat_accelerate_api::IntegerElementType::I16),
            (2, runmat_accelerate_api::IntegerElementType::I32),
            (3, runmat_accelerate_api::IntegerElementType::I64),
            (4, runmat_accelerate_api::IntegerElementType::U8),
            (5, runmat_accelerate_api::IntegerElementType::U16),
            (6, runmat_accelerate_api::IntegerElementType::U32),
            (7, runmat_accelerate_api::IntegerElementType::U64),
        ] {
            let handle = runmat_accelerate_api::GpuTensorHandle {
                shape: vec![1, 1],
                device_id: u32::MAX,
                buffer_id: u64::MAX - 350 - offset,
            };
            runmat_accelerate_api::set_handle_integer_type(&handle, element_type);
            let result = block(append_builtin(vec![
                Value::GpuTensor(handle.clone()),
                Value::String("suffix".into()),
            ]));
            runmat_accelerate_api::clear_handle_integer_type(&handle);
            let error = result.expect_err("resident numeric append input must reject");
            assert!(error.message().contains("expected string"), "{error}");
        }
    }

    #[test]
    fn text_boundary_positions_read_typed_integer_storage_exactly() {
        let boundary =
            Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).expect("boundary");

        assert_eq!(
            block(insert_after_builtin(
                Value::String("run".into()),
                vec![Value::Tensor(boundary), Value::String("mat".into())],
            ))
            .unwrap(),
            Value::String("runmat".into())
        );

        let start = Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("start");
        let stop = Tensor::new_integer(IntegerStorage::U16(vec![4]), vec![1, 1]).expect("stop");

        assert_eq!(
            block(replace_between_builtin(
                Value::String("abcde".into()),
                vec![
                    Value::Tensor(start),
                    Value::Tensor(stop),
                    Value::String("X".into()),
                ],
            ))
            .unwrap(),
            Value::String("aXe".into())
        );
    }

    #[test]
    fn text_boundary_positions_reject_invalid_integer_and_double_values() {
        let zero = Tensor::new_integer(IntegerStorage::U16(vec![0]), vec![1, 1]).expect("boundary");
        assert!(block(insert_after_builtin(
            Value::String("run".into()),
            vec![Value::Tensor(zero), Value::String("mat".into())],
        ))
        .is_err());

        assert!(block(insert_after_builtin(
            Value::String("run".into()),
            vec![Value::Num(1.0e300), Value::String("mat".into())],
        ))
        .is_err());
    }

    #[test]
    fn splitlines_reverse_deblank_and_matches_work() {
        assert_eq!(
            block(reverse_builtin(Value::String("abc".into()))).unwrap(),
            Value::String("cba".into())
        );
        assert_eq!(
            block(deblank_builtin(Value::CharArray(CharArray::new_row(
                "abc   "
            ))))
            .unwrap(),
            Value::CharArray(CharArray::new_row("abc"))
        );
        assert_eq!(
            block(matches_builtin(
                Value::String("A12".into()),
                crate::builtins::strings::core::compat::pattern_object(r"A\d+"),
            ))
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            block(erase_urls_builtin(Value::String(
                "see https://example.com now".into()
            )))
            .unwrap(),
            Value::String("see  now".into())
        );
        assert_eq!(
            block(erase_punctuation_builtin(
                Value::String("it's one and/or two.".into()),
                vec![]
            ))
            .unwrap(),
            Value::String("its one andor two".into())
        );
        assert_eq!(
            block(erase_punctuation_builtin(
                Value::CharArray(CharArray::new_row("cost: $5.00!")),
                vec![]
            ))
            .unwrap(),
            Value::CharArray(CharArray::new_row("cost 500"))
        );
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("alpha,beta!")),
                Value::String("C++ and C#".into()),
            ],
            1,
            2,
        )
        .unwrap();
        match block(erase_punctuation_builtin(Value::Cell(cell), vec![])).unwrap() {
            Value::Cell(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(
                    out.data[0],
                    Value::CharArray(CharArray::new_row("alphabeta"))
                );
                assert_eq!(out.data[1], Value::String("C and C".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
        assert_eq!(
            block(strjust_builtin(
                Value::CharArray(CharArray::new_row("  x")),
                vec![Value::String("left".into())],
            ))
            .unwrap(),
            Value::CharArray(CharArray::new_row("x  "))
        );
    }

    #[test]
    fn erase_descriptors_and_integer_audits_are_settled() {
        assert_eq!(ERASE_URLS_DESCRIPTOR.signatures.len(), 2);
        assert_eq!(ERASE_PUNCTUATION_DESCRIPTOR.signatures.len(), 3);
        for name in ["eraseURLs", "erasePunctuation"] {
            let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered");
            assert_eq!(
                builtin.integer_audit.expect("integer audit").kind,
                BuiltinIntegerAuditKind::NotApplicable
            );
        }
    }

    #[test]
    fn erase_text_builtins_reject_all_integer_classes_and_logical_values() {
        for value in [
            IntValue::I8(-1),
            IntValue::I16(-1),
            IntValue::I32(-1),
            IntValue::I64(-1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(u64::MAX),
        ] {
            assert!(block(erase_urls_builtin(Value::Int(value.clone()))).is_err());
            assert!(block(erase_punctuation_builtin(Value::Int(value), vec![])).is_err());
        }
        assert!(block(erase_urls_builtin(Value::Bool(true))).is_err());
        assert!(block(erase_punctuation_builtin(Value::Bool(true), vec![])).is_err());
    }

    #[test]
    fn erase_text_broad_container_extensions_are_independently_gated() {
        let matrix = Value::CharArray(
            CharArray::new(vec!['a', '!', 'b', '?'], 2, 2).expect("character matrix"),
        );
        let broad_cell = Value::Cell(
            CellArray::new(vec![Value::String("https://example.com!".into())], 1, 1)
                .expect("broad text cell"),
        );
        let nested_cell = Value::Cell(
            CellArray::new(
                vec![Value::Cell(
                    CellArray::new(vec![Value::CharArray(CharArray::new_row("nested!"))], 1, 1)
                        .expect("inner cell"),
                )],
                1,
                1,
            )
            .expect("outer cell"),
        );
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for (value, identifier) in [
            (
                matrix.clone(),
                "RunMat:compatibility:EraseURLsCharMatrixExtension",
            ),
            (
                broad_cell.clone(),
                "RunMat:compatibility:EraseURLsBroadCellExtension",
            ),
        ] {
            let error = block(erase_urls_builtin(value)).expect_err("strict eraseURLs gate");
            assert_eq!(error.identifier(), Some(identifier));
        }
        for (value, identifier) in [
            (
                matrix,
                "RunMat:compatibility:ErasePunctuationCharMatrixExtension",
            ),
            (
                broad_cell,
                "RunMat:compatibility:ErasePunctuationBroadCellExtension",
            ),
            (
                nested_cell,
                "RunMat:compatibility:ErasePunctuationBroadCellExtension",
            ),
        ] {
            let error = block(erase_punctuation_builtin(value, vec![]))
                .expect_err("strict erasePunctuation gate");
            assert_eq!(error.identifier(), Some(identifier));
        }
    }

    #[test]
    fn erase_text_builtins_reject_resident_numeric_without_provider_access() {
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let url_error = block(erase_urls_builtin(resident.clone())).unwrap_err();
        assert_eq!(
            url_error.identifier(),
            ERASE_URLS_ERROR_INVALID_INPUT.identifier
        );
        let nested = Value::Cell(CellArray::new(vec![resident], 1, 1).unwrap());
        let punctuation_error = block(erase_punctuation_builtin(nested, vec![])).unwrap_err();
        assert_eq!(
            punctuation_error.identifier(),
            ERASE_PUNCTUATION_ERROR_INVALID_INPUT.identifier
        );
    }

    #[test]
    fn extract_before_and_after_accept_every_integer_scalar_class_exactly() {
        assert_eq!(EXTRACT_BEFORE_DESCRIPTOR.errors.len(), 3);
        assert_eq!(EXTRACT_AFTER_DESCRIPTOR.errors.len(), 3);
        for position in [
            IntValue::I8(2),
            IntValue::I16(2),
            IntValue::I32(2),
            IntValue::I64(2),
            IntValue::U8(2),
            IntValue::U16(2),
            IntValue::U32(2),
            IntValue::U64(2),
        ] {
            assert_eq!(
                block(extract_before_builtin(
                    Value::String("abcdef".into()),
                    vec![Value::Int(position.clone())],
                ))
                .expect("extractBefore"),
                Value::String("a".into())
            );
            assert_eq!(
                block(extract_after_builtin(
                    Value::String("abcdef".into()),
                    vec![Value::Int(position)],
                ))
                .expect("extractAfter"),
                Value::String("cdef".into())
            );
        }
    }

    #[test]
    fn extract_before_and_after_apply_same_size_native_integer_positions() {
        let text = Value::StringArray(
            StringArray::new(vec!["abcd".into(), "wxyz".into()], vec![2, 1]).unwrap(),
        );
        for storage in [
            IntegerStorage::I8(vec![2, 3]),
            IntegerStorage::I16(vec![2, 3]),
            IntegerStorage::I32(vec![2, 3]),
            IntegerStorage::I64(vec![2, 3]),
            IntegerStorage::U8(vec![2, 3]),
            IntegerStorage::U16(vec![2, 3]),
            IntegerStorage::U32(vec![2, 3]),
            IntegerStorage::U64(vec![2, 3]),
        ] {
            let positions = Value::Tensor(Tensor::new_integer(storage, vec![2, 1]).unwrap());
            assert_eq!(
                block(extract_before_builtin(
                    text.clone(),
                    vec![positions.clone()]
                ))
                .unwrap(),
                Value::StringArray(
                    StringArray::new(vec!["a".into(), "wx".into()], vec![2, 1]).unwrap()
                )
            );
            assert_eq!(
                block(extract_after_builtin(text.clone(), vec![positions])).unwrap(),
                Value::StringArray(
                    StringArray::new(vec!["cd".into(), "z".into()], vec![2, 1]).unwrap()
                )
            );
        }
    }

    #[test]
    fn extract_before_and_after_enforce_strict_extensions_before_gather() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let matrix = Value::CharArray(
            CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).expect("character matrix"),
        );
        let before = block(extract_before_builtin(
            matrix,
            vec![Value::Int(IntValue::U8(1))],
        ))
        .expect_err("strict char matrix gate");
        assert_eq!(
            before.identifier(),
            EXTRACT_BEFORE_CHAR_MATRIX_EXTENSION.error_identifier
        );
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let after = block(extract_after_builtin(
            Value::String("abc".into()),
            vec![resident],
        ))
        .expect_err("strict resident gate");
        assert_eq!(
            after.identifier(),
            EXTRACT_AFTER_RESIDENT_POSITION_EXTENSION.error_identifier
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn extract_boundary_preserves_wide_u64_position_without_f64_conversion() {
        let position = (1_u64 << 53) + 1;
        let boundaries =
            BoundaryList::from_value(&Value::Int(IntValue::U64(position)), "extractBefore")
                .unwrap();
        assert!(matches!(
            boundaries.data.as_slice(),
            [Boundary::Position(value)] if *value == position as usize
        ));
    }

    #[test]
    fn extract_before_and_after_reject_extra_arguments() {
        assert!(block(extract_before_builtin(
            Value::String("abc".into()),
            vec![Value::Num(2.0), Value::Num(3.0)],
        ))
        .is_err());
        assert!(block(extract_after_builtin(
            Value::String("abc".into()),
            vec![Value::Num(2.0), Value::Num(3.0)],
        ))
        .is_err());
    }

    #[test]
    fn extract_before_and_after_reject_zero_floating_positions() {
        for position in [
            Value::Num(0.0),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
        ] {
            let before = block(extract_before_builtin(
                Value::String("abc".into()),
                vec![position.clone()],
            ))
            .expect_err("zero extractBefore position must reject");
            assert_eq!(
                before.identifier(),
                Some("RunMat:extractBefore:InvalidBoundary")
            );
            let after = block(extract_after_builtin(
                Value::String("abc".into()),
                vec![position],
            ))
            .expect_err("zero extractAfter position must reject");
            assert_eq!(
                after.identifier(),
                Some("RunMat:extractAfter:InvalidBoundary")
            );
        }
    }

    #[test]
    fn extract_before_and_after_accept_empty_character_boundary_without_panicking() {
        let boundary = Value::CharArray(CharArray::new(Vec::new(), 0, 0).unwrap());
        assert_eq!(
            block(extract_before_builtin(
                Value::String("abc".into()),
                vec![boundary.clone()],
            ))
            .expect("empty character boundary"),
            Value::String(String::new())
        );
        assert_eq!(
            block(extract_after_builtin(
                Value::String("abc".into()),
                vec![boundary],
            ))
            .expect("empty character boundary"),
            Value::String("abc".into())
        );
    }

    #[test]
    fn extract_before_and_after_classify_unsupported_text_cell_elements_as_invalid_input() {
        let text = Value::Cell(
            CellArray::new(vec![Value::Struct(StructValue::new())], 1, 1).expect("text cell"),
        );
        let before = block(extract_before_builtin(
            text.clone(),
            vec![Value::Int(IntValue::U8(1))],
        ))
        .expect_err("unsupported extractBefore text cell");
        assert_eq!(
            before.identifier(),
            Some("RunMat:extractBefore:InvalidInput")
        );
        let after = block(extract_after_builtin(
            text,
            vec![Value::Int(IntValue::U8(1))],
        ))
        .expect_err("unsupported extractAfter text cell");
        assert_eq!(after.identifier(), Some("RunMat:extractAfter:InvalidInput"));
    }

    #[test]
    fn deblank_rejects_every_integer_class_without_text_conversion() {
        for value in [
            IntValue::I8(-1),
            IntValue::I16(-1),
            IntValue::I32(-1),
            IntValue::I64(-1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(u64::MAX),
        ] {
            assert!(block(deblank_builtin(Value::Int(value))).is_err());
        }
        assert!(block(deblank_builtin(Value::Tensor(
            runmat_builtins::Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                .unwrap()
        )))
        .is_err());
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let nested = Value::Cell(CellArray::new(vec![resident], 1, 1).unwrap());
        let err = block(deblank_builtin(nested)).unwrap_err();
        assert!(!err.to_string().contains("provider"));
    }
}
