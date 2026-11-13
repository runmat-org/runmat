use crate::fea_contract::{descriptors::ERRORS, parameters::OUT_ANY};
use crate::*;

const DIRECTION_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "components",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite, nonzero 1-by-3 or 3-by-1 direction vector.",
}];
const DIRECTION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "direction = fea.Direction.vector(components)",
    inputs: &DIRECTION_INPUTS,
    outputs: &OUT_ANY,
}];
pub const FEA_DIRECTION_VECTOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DIRECTION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const DIRECTION_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "components",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The exact three-component integer vector is validated before explicit conversion to canonical normalized binary64 components.",
    }];
pub const FEA_DIRECTION_VECTOR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "three-component direction",
        inputs: &DIRECTION_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Direction normalization uses binary64 arithmetic; resident values are rejected without provider execution.",
    }];
pub const FEA_TYPED_CONSTRAINT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Concrete constraint constructors accept identifiers, immutable geometry regions, and already-dimensioned values; they have no direct integer input role.",
    };

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const TOLERANCE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tolerance",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Nonnegative typed fea.Length matching tolerance.",
}];
const MATCHING_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "pairing = fea.PeriodicPairing.matching()",
    inputs: &NO_INPUTS,
    outputs: &OUT_ANY,
}];
const GEOMETRIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "pairing = fea.PeriodicPairing.geometric(tolerance)",
    inputs: &TOLERANCE_INPUTS,
    outputs: &OUT_ANY,
}];
pub const FEA_PERIODIC_PAIRING_MATCHING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATCHING_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_PERIODIC_PAIRING_GEOMETRIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GEOMETRIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const TRANSLATION_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "translation",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Typed fea.Translation vector from primary to secondary geometry.",
}];
const ROTATION_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "origin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.Position on the rotation axis.",
    },
    BuiltinParamDescriptor {
        name: "axis",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.Direction rotation axis.",
    },
    BuiltinParamDescriptor {
        name: "angle",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.Rotation angle from primary to secondary geometry.",
    },
];
const IDENTITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "transform = fea.PeriodicTransform.identity()",
    inputs: &NO_INPUTS,
    outputs: &OUT_ANY,
}];
const TRANSLATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "transform = fea.PeriodicTransform.translation(translation)",
    inputs: &TRANSLATION_INPUTS,
    outputs: &OUT_ANY,
}];
const ROTATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "transform = fea.PeriodicTransform.rotation(origin, axis, angle)",
    inputs: &ROTATION_INPUTS,
    outputs: &OUT_ANY,
}];
pub const FEA_PERIODIC_TRANSFORM_IDENTITY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IDENTITY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_PERIODIC_TRANSFORM_TRANSLATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRANSLATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_PERIODIC_TRANSFORM_ROTATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ROTATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
