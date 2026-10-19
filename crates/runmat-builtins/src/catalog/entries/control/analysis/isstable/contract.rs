use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "True when every pole lies inside the stability region for the model's time domain.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO transfer-function model.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isstable(sys)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const ISSTABLE_ERROR_INVALID_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSTABLE.INVALID_MODEL",
    identifier: Some("RunMat:isstable:InvalidModel"),
    when: "Input system is not a valid SISO tf object.",
    message: "isstable: invalid model",
};
pub const ISSTABLE_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSTABLE.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:isstable:UnsupportedModel"),
    when: "Model form is unsupported.",
    message: "isstable: unsupported model",
};
const ERRORS: [BuiltinErrorDescriptor; 6] = [
    ISSTABLE_ERROR_INVALID_MODEL,
    ISSTABLE_ERROR_UNSUPPORTED_MODEL,
    BuiltinErrorDescriptor {
        code: "RM.ISSTABLE.INVALID_ARGUMENT",
        identifier: Some("RunMat:isstable:InvalidArgument"),
        when: "A transfer-function scalar property or Variable property has the wrong value type.",
        message: "isstable: invalid transfer-function property",
    },
    BuiltinErrorDescriptor {
        code: "RM.ISSTABLE.INVALID_VARIABLE",
        identifier: Some("RunMat:tf:InvalidVariable"),
        when: "The shared transfer-function parser rejects a Variable value or its time-domain pairing.",
        message: "tf: invalid Variable option",
    },
    BuiltinErrorDescriptor {
        code: "RM.ISSTABLE.INVALID_SAMPLE_TIME",
        identifier: Some("RunMat:tf:InvalidSampleTime"),
        when: "The shared transfer-function parser rejects the Variable and sample-time pairing.",
        message: "isstable: invalid transfer-function sample time",
    },
    BuiltinErrorDescriptor {
        code: "RM.ISSTABLE.INTERNAL",
        identifier: Some("RunMat:isstable:Internal"),
        when: "Pole calculation failed.",
        message: "isstable: internal error",
    },
];

pub const ISSTABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ISSTABLE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "A fundamental integer sys is not a dynamic-system model and rejects before provider access. This direct-input audit does not cover typed integer properties assigned to an existing tf object; consumed properties undergo separate extension and exactness admission.",
};
