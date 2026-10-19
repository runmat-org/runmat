use crate::*;

const SPRINTF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Formatted character row vector output.",
}];

const SPRINTF_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "formatSpec",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Format template text.",
    },
    BuiltinParamDescriptor {
        name: "A...",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Values substituted by conversion specifiers.",
    },
];

const SPRINTF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "txt = sprintf(formatSpec, A...)",
    inputs: &SPRINTF_INPUTS,
    outputs: &SPRINTF_OUTPUT,
}];

pub const SPRINTF_ERROR_INVALID_FORMAT_SPEC: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPRINTF.INVALID_FORMAT_SPEC",
    identifier: Some("RunMat:sprintf:InvalidFormatSpec"),
    when: "formatSpec is invalid or unsupported.",
    message: "sprintf: invalid formatSpec",
};

pub const SPRINTF_ERROR_ARGUMENT_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPRINTF.ARGUMENT_MISMATCH",
    identifier: Some("RunMat:sprintf:ArgumentMismatch"),
    when: "Conversion specifier count does not match provided arguments.",
    message: "sprintf: format arguments do not match conversion specifiers",
};

pub const SPRINTF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPRINTF.INTERNAL",
    identifier: Some("RunMat:sprintf:InternalError"),
    when: "Internal char-array construction failed.",
    message: "sprintf: internal error",
};

const SPRINTF_ERRORS: [BuiltinErrorDescriptor; 4] = [
    SPRINTF_ERROR_INVALID_FORMAT_SPEC,
    SPRINTF_ERROR_ARGUMENT_MISMATCH,
    SPRINTF_ERROR_INTERNAL,
    SPRINTF_ERROR_TOO_MANY_OUTPUTS,
];

pub const SPRINTF_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPRINTF.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:sprintf:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "sprintf: too many output arguments",
};

pub const SPRINTF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPRINTF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPRINTF_ERRORS,
};

const SPRINTF_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A...",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target explicitly lists all eight integer classes. Integer conversions read authoritative values directly, including signed, unsigned, octal, hexadecimal, character-code, width, and precision roles.",
    }];
pub const SPRINTF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "txt = sprintf(formatSpec, integer_A...)",
        inputs: &SPRINTF_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Formatting preserves full-width integer text without an f64 intermediary and consumes array elements in column order. Resident numeric arguments gather authoritatively because formatting executes on the client.",
    }];
