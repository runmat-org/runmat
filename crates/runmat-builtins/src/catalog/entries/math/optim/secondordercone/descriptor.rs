use crate::*;

const INTEGER_INPUTS: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "A, b, d, or gamma", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Existing RunMat constructor forms read native integer storage then normalize to binary64, rounding values outside the exact range. MATLAB integer-overload parity is not asserted.",
}];
pub const SECONDORDERCONE_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor {
    form: "secondordercone(integer-valued data)", inputs: INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::Double,
    overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "Host constraint metadata retains the existing floating conversion policy, not an exact-integer cone algebra.",
}];

const SOC_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "socConstraint",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second-order cone constraint descriptor.",
}];

const SOC_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cone matrix in norm(A*x - b) <= d'*x - gamma.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cone center vector, subtracted from A*x.",
    },
    BuiltinParamDescriptor {
        name: "d",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Linear cone right-hand-side vector.",
    },
    BuiltinParamDescriptor {
        name: "gamma",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar bound subtracted from d'*x; smaller gamma loosens the cone.",
    },
];

const SOC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "socConstraint = secondordercone(A, b, d, gamma)",
    inputs: &SOC_INPUTS,
    outputs: &SOC_OUTPUT,
}];

pub const SECONDORDERCONE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SOC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONEPROG_ERRORS,
};
