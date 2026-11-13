use crate::*;
const SUPERCLASSES_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell row of visible superclass names as character vectors.",
}];

const SUPERCLASSES_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj_or_class",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Object instance or class name string/character vector.",
}];

const SUPERCLASSES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "s = superclasses(obj_or_class)",
    inputs: &SUPERCLASSES_INPUTS,
    outputs: &SUPERCLASSES_OUTPUT,
}];

pub const SUPERCLASSES_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "superclasses accepts class-name text or class objects. Integer primitives are neither and reject from type metadata without numeric conversion or provider payload access.",
    };

pub const SUPERCLASSES_ERROR_CLASS_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUPERCLASSES.CLASS_INVALID",
    identifier: Some("RunMat:SuperclassesClassInvalid"),
    when: "The argument is neither an object nor a class-name string scalar/row character vector.",
    message: "superclasses: input must be an object or class name",
};

pub const SUPERCLASSES_ERROR_CLASS_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUPERCLASSES.CLASS_UNKNOWN",
    identifier: Some("RunMat:SuperclassesClassUnknown"),
    when: "The requested class is not present in RunMat class metadata.",
    message: "superclasses: class metadata not found",
};

pub const SUPERCLASSES_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SUPERCLASSES_ERROR_OUTPUTS,
    SUPERCLASSES_ERROR_CLASS_INVALID,
    SUPERCLASSES_ERROR_CLASS_UNKNOWN,
];

pub const SUPERCLASSES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUPERCLASSES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUPERCLASSES_ERRORS,
};

pub const SUPERCLASSES_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUPERCLASSES.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:superclasses:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "superclasses: expected at most one output",
};
