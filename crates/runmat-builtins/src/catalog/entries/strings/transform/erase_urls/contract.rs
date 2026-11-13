use crate::*;

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text.",
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
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "newStr = eraseURLs(str)",
        inputs: &INPUTS,
        outputs: &OUT_NEW_STR,
    },
    BuiltinSignatureDescriptor {
        label: "newDocuments = eraseURLs(documents)",
        inputs: &INPUTS,
        outputs: &OUT_NEW_DOCUMENTS,
    },
];
pub const ERASE_URLS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_URLS.INVALID_INPUT",
    identifier: Some("RunMat:eraseURLs:InvalidInput"),
    when: "Input is not documented host text or a tokenizedDocument object.",
    message: "eraseURLs: input must be text or tokenizedDocument",
};
pub const ERASE_URLS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[ERASE_URLS_ERROR_INVALID_INPUT],
};
pub const ERASE_URLS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "eraseURLs accepts host text or tokenizedDocument input only. All eight integer classes, logical values, and resident numeric handles reject without numeric-to-text conversion or provider access.",
};
pub const ERASE_URLS_CHAR_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "erase-urls-char-matrix-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "eraseURLs with a multirow character matrix is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EraseURLsCharMatrixExtension"),
    };
pub const ERASE_URLS_BROAD_CELL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "erase-urls-broad-cell-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "eraseURLs with nested cells or non-character cell elements is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EraseURLsBroadCellExtension"),
    };
pub const ERASE_URLS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    ERASE_URLS_CHAR_MATRIX_EXTENSION,
    ERASE_URLS_BROAD_CELL_EXTENSION,
];
