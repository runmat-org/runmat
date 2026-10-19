use crate::*;

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text.",
}];
const DOCUMENT_INPUTS_WITH_TYPES: [BuiltinParamDescriptor; 2] = [
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
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "newStr = erasePunctuation(str)",
        inputs: &INPUTS,
        outputs: &OUT_NEW_STR,
    },
    BuiltinSignatureDescriptor {
        label: "newDocuments = erasePunctuation(documents)",
        inputs: &INPUTS,
        outputs: &OUT_NEW_DOCUMENTS,
    },
    BuiltinSignatureDescriptor {
        label: "newDocuments = erasePunctuation(documents, 'TokenTypes', types)",
        inputs: &DOCUMENT_INPUTS_WITH_TYPES,
        outputs: &OUT_NEW_DOCUMENTS,
    },
];
pub const ERASE_PUNCTUATION_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_PUNCTUATION.INVALID_INPUT",
    identifier: Some("RunMat:erasePunctuation:InvalidInput"),
    when: "Input is not documented host text or a tokenizedDocument object.",
    message: "erasePunctuation: input must be text or tokenizedDocument",
};
pub const ERASE_PUNCTUATION_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_PUNCTUATION.INVALID_OPTION",
    identifier: Some("RunMat:erasePunctuation:InvalidOption"),
    when: "TokenTypes is malformed, empty, or used with non-document text.",
    message: "erasePunctuation: invalid TokenTypes option",
};
pub const ERASE_PUNCTUATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        ERASE_PUNCTUATION_ERROR_INVALID_INPUT,
        ERASE_PUNCTUATION_ERROR_INVALID_OPTION,
    ],
};
pub const ERASE_PUNCTUATION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "erasePunctuation accepts host text, tokenizedDocument, and textual TokenTypes values only. All eight integer classes, logical values, and resident numeric handles reject without numeric-to-text conversion or provider access.",
};
pub const ERASE_PUNCTUATION_CHAR_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "erase-punctuation-char-matrix-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "erasePunctuation with a multirow character matrix is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ErasePunctuationCharMatrixExtension"),
    };
pub const ERASE_PUNCTUATION_BROAD_CELL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "erase-punctuation-broad-cell-input", mode: BuiltinExtensionMode::RunMatOnly,
    description: "erasePunctuation with nested cells or non-character cell elements is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ErasePunctuationBroadCellExtension"),
};
pub const ERASE_PUNCTUATION_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    ERASE_PUNCTUATION_CHAR_MATRIX_EXTENSION,
    ERASE_PUNCTUATION_BROAD_CELL_EXTENSION,
];
