use crate::*;
pub const METACLASS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mc",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Meta-class reference for the input value's runtime class.",
}];

pub const METACLASS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "object",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Object or value whose runtime class metadata is requested.",
}];

pub const METACLASS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "mc = metaclass(object)",
    inputs: &METACLASS_INPUTS,
    outputs: &METACLASS_OUTPUT,
}];

pub const METACLASS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &METACLASS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &METACLASS_PUBLIC_ERRORS,
};

pub const METACLASS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "object",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer scalars and arrays are ordinary MATLAB values whose runtime class metadata preserves exact signedness and width; payload data is not inspected.",
    }];

pub const METACLASS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "mc = metaclass(integer_object)",
        inputs: &METACLASS_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "RunMat returns its ClassRef representation of matlab.metadata.Class. Explicit resident values report gpuArray metadata while automatic residency remains transparent, both without gathering payload bytes.",
    }];

pub const METACLASS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.METACLASS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:metaclass:TooManyOutputs"),
    when: "More outputs are requested than this form supports.",
    message: "metaclass: too many output arguments",
};
const METACLASS_PUBLIC_ERRORS: [BuiltinErrorDescriptor; 1] = [METACLASS_ERROR_TOO_MANY_OUTPUTS];
