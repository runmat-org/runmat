use crate::*;

pub const SYMRCM_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "Integer matrices contribute their exact zero/nonzero structure to the graph ordering.",
    }];
pub const SYMRCM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "p = symrcm(integer_A)",
        inputs: &SYMRCM_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Adjacency is derived with exact integer zero tests and the result is a one-based double permutation vector. Resident inputs use the provider hook or gather fallback.",
    }];

pub const SYMRCM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Permutation row vector (1-based indices) from reverse Cuthill-McKee ordering.",
}];

pub const SYMRCM_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];

pub const SYMRCM_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = symrcm(A)",
    inputs: &SYMRCM_INPUTS,
    outputs: &SYMRCM_OUTPUT,
}];

pub const SYMRCM_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SYMRCM.INVALID_INPUT",
    identifier: Some("RunMat:symrcm:InvalidInput"),
    when: "Input shape/type cannot be interpreted as a numeric or logical square matrix.",
    message: "symrcm: invalid input",
};

pub const SYMRCM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SYMRCM.INTERNAL",
    identifier: Some("RunMat:symrcm:Internal"),
    when: "Runtime fails while constructing intermediate tensors or adjacency structures.",
    message: "symrcm: internal runtime failure",
};

pub const SYMRCM_ERRORS: [BuiltinErrorDescriptor; 2] =
    [SYMRCM_ERROR_INVALID_INPUT, SYMRCM_ERROR_INTERNAL];

pub const SYMRCM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SYMRCM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SYMRCM_ERRORS,
};
