//! Signatures, parameters, and integer facts for `fea.Direction` constructors.
//! The runtime binding re-exports these facts; it does not redefine them.

use crate::fea_contract::descriptors::ERRORS;
use crate::*;

const fn required(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

const VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "components",
    BuiltinParamType::NumericArray,
    "Finite, nonzero 1-by-3 or 3-by-1 real vector.",
)];
const VECTOR_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "direction",
    BuiltinParamType::Any,
    "fea.Direction object.",
)];
const VECTOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "direction = fea.Direction.vector(components)",
    inputs: &VECTOR_INPUTS,
    outputs: &VECTOR_OUTPUTS,
}];
pub const FEA_DIRECTION_VECTOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VECTOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const COMPONENTS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "components",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The exact three-component integer vector is validated before explicit conversion to canonical normalized binary64 components.",
    }];
pub const FEA_DIRECTION_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "three-component direction",
        inputs: &COMPONENTS_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Direction normalization uses binary64 arithmetic; resident values are rejected without provider execution.",
    }];
