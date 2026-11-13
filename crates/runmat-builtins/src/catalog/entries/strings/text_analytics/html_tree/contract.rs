use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tree",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Parsed htmlTree object, or a shape-preserving cell for nonscalar input.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "code",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host HTML string array, character vector, or cell array of character vectors.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tree = htmlTree(code)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const HTML_TREE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HTML.INVALID_INPUT",
    identifier: Some("RunMat:html:InvalidInput"),
    when: "The input is not a supported host HTML text form.",
    message: "htmlTree received invalid input",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [HTML_TREE_ERROR_INVALID_INPUT];
pub const HTML_TREE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const HTML_TREE_CELL_OBJECT_ARRAY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "htmltree-cell-object-array",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "htmlTree nonscalar input returns a shape-preserving cell array of scalar htmlTree objects until native object arrays cover this form",
        error_identifier: Some("RunMat:compatibility:HtmlTreeCellObjectArrayExtension"),
    };
pub const HTML_TREE_BROAD_CELL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "htmltree-broad-cell-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "htmlTree accepts string-valued cells only as a RunMat extension; the public cell form is a cell array of character vectors",
    error_identifier: Some("RunMat:compatibility:HtmlTreeBroadCellExtension"),
};
pub const HTML_TREE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    HTML_TREE_CELL_OBJECT_ARRAY_EXTENSION,
    HTML_TREE_BROAD_CELL_EXTENSION,
];
pub const HTML_TREE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "htmlTree accepts host string arrays, character vectors, and cell arrays of character vectors. All integer, logical, complex, and resident numeric values reject before parsing or provider access.",
};
