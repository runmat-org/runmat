use crate::*;

const FLIPLR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are reordered through authoritative storage without numeric conversion.",
    }];
pub const FLIPLR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "B = fliplr(integer_A)",
        inputs: &FLIPLR_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Dimension 2 is reversed exactly; GPU fallback restores the result to the owning provider.",
    }];

const FLIPLR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array mirrored left-to-right (dimension 2).",
}];

const FLIPLR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array/value to reverse along columns.",
}];

const FLIPLR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = fliplr(A)",
    inputs: &FLIPLR_INPUTS,
    outputs: &FLIPLR_OUTPUT,
}];

pub const FLIPLR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FLIPLR.INVALID_INPUT",
    identifier: Some("RunMat:fliplr:InvalidInput"),
    when: "Input type or conversion path is unsupported for fliplr.",
    message: "fliplr: invalid input argument",
};

pub const FLIPLR_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FLIPLR.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:fliplr:UnsupportedInput"),
    when: "Input type is unsupported for fliplr.",
    message: "fliplr: unsupported input type",
};

const FLIPLR_ERRORS: [BuiltinErrorDescriptor; 2] =
    [FLIPLR_ERROR_INVALID_INPUT, FLIPLR_ERROR_UNSUPPORTED_INPUT];

pub const FLIPLR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FLIPLR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FLIPLR_ERRORS,
};
