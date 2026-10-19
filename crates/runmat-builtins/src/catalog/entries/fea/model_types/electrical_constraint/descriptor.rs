//! Signatures, parameters, and integer facts for `fea.ElectricalConstraint` constructors.
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

pub const FEA_ELECTRICAL_CONSTRAINT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Concrete constraint constructors accept identifiers, immutable geometry regions, and already-dimensioned values; they have no direct integer input role.",
};

const EQUIPOTENTIAL_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Constraint identifier (nonempty text).",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "geometry.Region selected from the model geometry.",
    ),
];
const EQUIPOTENTIAL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.ElectricalConstraint object.",
)];
const EQUIPOTENTIAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.ElectricalConstraint.equipotential(id, region)",
    inputs: &EQUIPOTENTIAL_INPUTS,
    outputs: &EQUIPOTENTIAL_OUTPUTS,
}];
pub const FEA_ELECTRICAL_CONSTRAINT_EQUIPOTENTIAL_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &EQUIPOTENTIAL_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const FLOATING_CONDUCTOR_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Constraint identifier (nonempty text).",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "geometry.Region selected from the model geometry.",
    ),
];
const FLOATING_CONDUCTOR_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.ElectricalConstraint object.",
)];
const FLOATING_CONDUCTOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "constraint = fea.ElectricalConstraint.floatingConductor(id, region)",
        inputs: &FLOATING_CONDUCTOR_INPUTS,
        outputs: &FLOATING_CONDUCTOR_OUTPUTS,
    }];
pub const FEA_ELECTRICAL_CONSTRAINT_FLOATING_CONDUCTOR_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &FLOATING_CONDUCTOR_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const POTENTIAL_REFERENCE_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Constraint identifier (nonempty text).",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "geometry.Region selected from the model geometry.",
    ),
];
const POTENTIAL_REFERENCE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.ElectricalConstraint object.",
)];
const POTENTIAL_REFERENCE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "constraint = fea.ElectricalConstraint.potentialReference(id, region)",
        inputs: &POTENTIAL_REFERENCE_INPUTS,
        outputs: &POTENTIAL_REFERENCE_OUTPUTS,
    }];
pub const FEA_ELECTRICAL_CONSTRAINT_POTENTIAL_REFERENCE_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &POTENTIAL_REFERENCE_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const PRESCRIBED_POTENTIAL_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Constraint identifier (nonempty text).",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "geometry.Region selected from the model geometry.",
    ),
    required(
        "potential",
        BuiltinParamType::Any,
        "Electric potential as fea.ElectricPotential (V).",
    ),
];
const PRESCRIBED_POTENTIAL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.ElectricalConstraint object.",
)];
const PRESCRIBED_POTENTIAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "constraint = fea.ElectricalConstraint.prescribedPotential(id, region, potential)",
        inputs: &PRESCRIBED_POTENTIAL_INPUTS,
        outputs: &PRESCRIBED_POTENTIAL_OUTPUTS,
    }];
pub const FEA_ELECTRICAL_CONSTRAINT_PRESCRIBED_POTENTIAL_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &PRESCRIBED_POTENTIAL_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const INTEGRAL_CURRENT_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Constraint identifier (nonempty text).",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "geometry.Region selected from the model geometry.",
    ),
    required(
        "current",
        BuiltinParamType::Any,
        "Net current as fea.ElectricCurrent (A).",
    ),
];
const INTEGRAL_CURRENT_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.ElectricalConstraint object.",
)];
const INTEGRAL_CURRENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.ElectricalConstraint.integralCurrent(id, region, current)",
    inputs: &INTEGRAL_CURRENT_INPUTS,
    outputs: &INTEGRAL_CURRENT_OUTPUTS,
}];
pub const FEA_ELECTRICAL_CONSTRAINT_INTEGRAL_CURRENT_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &INTEGRAL_CURRENT_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const PERIODIC_INPUTS: [BuiltinParamDescriptor; 5] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Constraint identifier (nonempty text).",
    ),
    required("primary", BuiltinParamType::Any, "Primary geometry.Region."),
    required(
        "secondary",
        BuiltinParamType::Any,
        "Secondary geometry.Region, distinct from primary.",
    ),
    required(
        "transform",
        BuiltinParamType::Any,
        "fea.PeriodicTransform mapping primary geometry onto secondary geometry.",
    ),
    required(
        "pairing",
        BuiltinParamType::Any,
        "fea.PeriodicPairing policy that matches primary and secondary entities.",
    ),
];
const PERIODIC_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.ElectricalConstraint object.",
)];
const PERIODIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label:
        "constraint = fea.ElectricalConstraint.periodic(id, primary, secondary, transform, pairing)",
    inputs: &PERIODIC_INPUTS,
    outputs: &PERIODIC_OUTPUTS,
}];
pub const FEA_ELECTRICAL_CONSTRAINT_PERIODIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PERIODIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
