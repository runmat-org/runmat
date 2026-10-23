use crate::*;

pub const EXTRACT_HTML_TEXT_EXTRACT_HTML_CHAR_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "extracthtmltext-char-matrix",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "extractHTMLText row-wise character-matrix input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ExtractHTMLTextCharMatrixExtension"),
    };

pub const EXTRACT_HTML_TEXT_EXTRACT_HTML_BROAD_CELL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "extracthtmltext-broad-cell",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "extractHTMLText with string-valued or mixed htmlTree/text cells is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ExtractHTMLTextBroadCellExtension"),
    };

pub const EXTRACT_HTML_TEXT_EXTRACT_HTML_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    EXTRACT_HTML_TEXT_EXTRACT_HTML_CHAR_MATRIX_EXTENSION,
    EXTRACT_HTML_TEXT_EXTRACT_HTML_BROAD_CELL_EXTENSION,
];

pub const EXTRACT_HTML_TEXT_EXTRACT_HTML_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "extractHTMLText accepts host text or htmlTree input and textual ExtractionMethod values only. All eight integer classes, logical and complex values, and resident numeric handles reject without numeric-to-text conversion or provider access.",
    };

const OUT_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Extracted text.",
}];

const IN_CODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "code",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HTML code or htmlTree object.",
}];

const IN_CODE_METHOD: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "code",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HTML code or htmlTree object.",
    },
    BuiltinParamDescriptor {
        name: "Name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("ExtractionMethod"),
        description: "Extraction method option name.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("tree"),
        description: "Extraction method: tree, article, or all-text.",
    },
];

pub const EXTRACT_HTML_TEXT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HTML.INVALID_INPUT",
    identifier: Some("RunMat:html:InvalidInput"),
    when: "Inputs are not a supported htmlTree or extractHTMLText form.",
    message: "HTML Text Analytics helper received invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [
    EXTRACT_HTML_TEXT_ERROR_INVALID_INPUT,
    EXTRACT_HTML_TEXT_ERROR_TOO_MANY_OUTPUTS,
];

pub const EXTRACT_HTML_TEXT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "str = extractHTMLText(code)",
            inputs: &IN_CODE,
            outputs: &OUT_TEXT,
        },
        BuiltinSignatureDescriptor {
            label: "str = extractHTMLText(___, 'ExtractionMethod', method)",
            inputs: &IN_CODE_METHOD,
            outputs: &OUT_TEXT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const EXTRACT_HTML_TEXT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.EXTRACT_HTML_TEXT.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:extractHTMLText:TooManyOutputs"),
        when: "More than one output is requested.",
        message: "extractHTMLText: too many output arguments",
    };
