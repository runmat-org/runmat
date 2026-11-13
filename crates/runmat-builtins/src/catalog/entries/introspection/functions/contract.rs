use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function-handle metadata struct.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fh",
    ty: BuiltinParamType::Callable,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle or closure.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "info = functions(fh)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];
pub const FUNCTIONS_ERROR_HANDLE_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FUNCTIONS.HANDLE_UNSUPPORTED",
    identifier: Some("RunMat:FunctionsHandleUnsupported"),
    when: "The input is not a supported function handle value.",
    message: "functions: expected a function handle",
};
pub const FUNCTIONS_ERRORS: [BuiltinErrorDescriptor; 1] = [FUNCTIONS_ERROR_HANDLE_UNSUPPORTED];
pub const FUNCTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FUNCTIONS_ERRORS,
};
pub const FUNCTIONS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Only function handles are accepted. Integer, numeric, and resident inputs reject without provider access; captured closure values are opaque state and are not converted.",
};
