use crate::*;
const SYM_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "value",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The current RunMat symbolic implementation accepts integer scalars from every native class.",
}];
pub const SYM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "S = sym(integer_scalar)",
        inputs: &SYM_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The scalar is converted directly from authoritative integer storage to an arbitrary-precision symbolic integer; no binary floating-point boundary is crossed. Numeric-array symbolic construction remains outside the current scalar implementation.",
    }];

const SYM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar symbolic variable or constant.",
}];

const SYM_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Variable name, numeric scalar, or existing symbolic scalar.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Reserved symbolic options.",
    },
];

const SYM_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "S = sym(value)",
    inputs: &SYM_INPUTS,
    outputs: &SYM_OUTPUT,
}];

pub const SYM_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.SYM.ARG_COUNT",
        identifier: Some("RunMat:sym:ArgCount"),
        when: "Options beyond the scalar input are supplied.",
        message: "sym: unsupported options",
    },
    BuiltinErrorDescriptor {
        code: "RM.SYM.INVALID_TEXT",
        identifier: Some("RunMat:sym:InvalidText"),
        when: "Text input is neither a valid identifier nor a numeric scalar literal.",
        message: "sym: invalid symbolic text",
    },
    BuiltinErrorDescriptor {
        code: "RM.SYM.INVALID_INPUT",
        identifier: Some("RunMat:sym:InvalidInput"),
        when: "Input cannot be represented as a scalar symbolic value.",
        message: "sym: expected a scalar text, numeric, logical, or symbolic input",
    },
    SYM_ERROR_TOO_MANY_OUTPUTS,
];

pub const SYM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SYM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SYM_ERRORS,
};

pub const SYM_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SYM.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:sym:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "sym: too many output arguments",
};
