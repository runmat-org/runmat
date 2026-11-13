use crate::*;

const FLIPUD_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are reordered through authoritative storage without numeric conversion.",
    }];
pub const FLIPUD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "B = flipud(integer_A)",
        inputs: &FLIPUD_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Dimension 1 is reversed exactly; GPU fallback restores the result to the owning provider.",
    }];

const FLIPUD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array mirrored up-to-down (dimension 1).",
}];

const FLIPUD_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array/value to reverse along rows.",
}];

const FLIPUD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = flipud(A)",
    inputs: &FLIPUD_INPUTS,
    outputs: &FLIPUD_OUTPUT,
}];

pub const FLIPUD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FLIPUD.INVALID_INPUT",
    identifier: Some("RunMat:flipud:InvalidInput"),
    when: "Input type or conversion path is invalid for flipud.",
    message: "flipud: invalid input argument",
};

pub const FLIPUD_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FLIPUD.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:flipud:UnsupportedInput"),
    when: "Input type is unsupported for flipud.",
    message: "flipud: unsupported input type",
};

const FLIPUD_ERRORS: [BuiltinErrorDescriptor; 2] =
    [FLIPUD_ERROR_INVALID_INPUT, FLIPUD_ERROR_UNSUPPORTED_INPUT];

pub const FLIPUD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FLIPUD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FLIPUD_ERRORS,
};
