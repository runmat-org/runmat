use crate::*;

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r_or_A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite real or complex single/double root vector or square host matrix.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Monic descending-power coefficient row of the input class and domain.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = poly(r_or_A)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const POLY_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLY.INVALID_INPUT",
    identifier: Some("RunMat:poly:InvalidInput"),
    when: "The operand is not a finite host single/double vector or square matrix, or output arity is invalid.",
    message: "poly: invalid input",
};
pub const POLY_ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLY.NUMERICAL",
    identifier: Some("RunMat:poly:Numerical"),
    when: "The Schur iteration does not converge or the coefficient/output allocation fails.",
    message: "poly: numerical or allocation failure",
};
pub const POLY_ERROR_UNSUPPORTED_PROVIDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLY.UNSUPPORTED_PROVIDER",
    identifier: Some("RunMat:poly:UnsupportedProvider"),
    when: "The input is resident on a provider without a certified result-restoration owner.",
    message: "poly: unsupported input provider",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    POLY_ERROR_INVALID_INPUT,
    POLY_ERROR_NUMERICAL,
    POLY_ERROR_UNSUPPORTED_PROVIDER,
];
pub const POLY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "r_or_A",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer and logical operands are not coerced into a polynomial computation.",
}];
pub const POLY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "p = poly(r_or_A)",
    inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "Only floating single/double operands are supported; current implementation is host-only.",
}];
