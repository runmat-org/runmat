use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "q",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Integrated polynomial coefficient row vector.",
}];

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single- or double-precision polynomial coefficients.",
}];

const INPUTS_WITH_CONSTANT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Single- or double-precision polynomial coefficients.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Real or complex scalar integration constant; defaults to zero.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "q = polyint(p)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "q = polyint(p, k)",
        inputs: &INPUTS_WITH_CONSTANT,
        outputs: &OUTPUT,
    },
];

pub const POLYINT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYINT.INVALID_ARGUMENT",
    identifier: Some("RunMat:polyint:InvalidArgument"),
    when: "Input arity or integration-constant argument is malformed.",
    message: "polyint: invalid argument",
};

pub const POLYINT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYINT.INVALID_INPUT",
    identifier: Some("RunMat:polyint:InvalidInput"),
    when: "Inputs are not single- or double-precision coefficient/constant values, or coefficients do not form a vector.",
    message: "polyint: invalid input",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLYINT.INTERNAL",
    identifier: Some("RunMat:polyint:Internal"),
    when: "Runtime fails while building output tensors or provider fallback paths.",
    message: "polyint: internal runtime failure",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    POLYINT_ERROR_INVALID_ARGUMENT,
    POLYINT_ERROR_INVALID_INPUT,
    ERROR_INTERNAL,
];

pub const POLYINT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "p",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer and logical coefficient vectors are rejected before floating host/provider dispatch.",
    },
    BuiltinIntegerInputCapability {
        name: "k",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer and logical integration constants are rejected; only single/double real or complex values are supported.",
    },
];

pub const POLYINT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "q = polyint(p, k)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "No integer overload; empty accepted-class masks prevent generic numeric coercion from admitting integers.",
    }];
