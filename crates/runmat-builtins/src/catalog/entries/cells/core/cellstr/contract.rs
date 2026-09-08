use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};

const OUTPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array of character vectors.",
}];
const INPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Character array, string array, or supported RunMat text value.",
}];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "C = cellstr(A)",
    inputs: INPUTS,
    outputs: OUTPUTS,
}];

pub const CELLSTR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLSTR.INVALID_INPUT",
    identifier: Some("RunMat:cellstr:InvalidInput"),
    when: "The input is not a supported text value.",
    message: "cellstr: input must be text-compatible",
};
pub const CELLSTR_ERROR_INVALID_CONTENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLSTR.INVALID_CONTENTS",
    identifier: Some("RunMat:cellstr:InvalidContents"),
    when: "A cell input contains something other than a character vector or string scalar.",
    message: "cellstr: cell array elements must be text scalars",
};
pub const CELLSTR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELLSTR.INTERNAL",
    identifier: None,
    when: "The supplied array shape does not match its storage.",
    message: "cellstr: internal shape error",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[
    CELLSTR_ERROR_INVALID_INPUT,
    CELLSTR_ERROR_INVALID_CONTENTS,
    CELLSTR_ERROR_INTERNAL,
];

pub const CELLSTR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};

pub const CELLSTR_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "cellstr has no numeric input form; all fixed-width integer classes reject before provider lookup or conversion.",
};

pub const CELLSTR_CELL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cellstr-cell-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cellstr with a cell-array input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CellstrCellInputExtension"),
};
pub const CELLSTR_SYMBOLIC_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cellstr-symbolic-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cellstr with symbolic input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CellstrSymbolicInputExtension"),
    };
pub const CELLSTR_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    CELLSTR_CELL_INPUT_EXTENSION,
    CELLSTR_SYMBOLIC_INPUT_EXTENSION,
];
