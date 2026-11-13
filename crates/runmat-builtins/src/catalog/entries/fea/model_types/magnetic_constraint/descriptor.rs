//! Signatures, parameters, and integer facts for `fea.MagneticConstraint` constructors.
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

pub const FEA_MAGNETIC_CONSTRAINT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Concrete constraint constructors accept identifiers, immutable geometry regions, and already-dimensioned values; they have no direct integer input role.",
};

const VECTOR_POTENTIAL_GROUND_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const VECTOR_POTENTIAL_GROUND_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.MagneticConstraint object.",
)];
const VECTOR_POTENTIAL_GROUND_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "constraint = fea.MagneticConstraint.vectorPotentialGround(id, region)",
        inputs: &VECTOR_POTENTIAL_GROUND_INPUTS,
        outputs: &VECTOR_POTENTIAL_GROUND_OUTPUTS,
    }];
pub const FEA_MAGNETIC_CONSTRAINT_VECTOR_POTENTIAL_GROUND_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &VECTOR_POTENTIAL_GROUND_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const COULOMB_GAUGE_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const COULOMB_GAUGE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.MagneticConstraint object.",
)];
const COULOMB_GAUGE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.MagneticConstraint.coulombGauge(id, region)",
    inputs: &COULOMB_GAUGE_INPUTS,
    outputs: &COULOMB_GAUGE_OUTPUTS,
}];
pub const FEA_MAGNETIC_CONSTRAINT_COULOMB_GAUGE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COULOMB_GAUGE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const PRESCRIBED_VECTOR_POTENTIAL_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "Vector potential as fea.MagneticVectorPotential (Wb/m).",
    ),
];
const PRESCRIBED_VECTOR_POTENTIAL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.MagneticConstraint object.",
)];
const PRESCRIBED_VECTOR_POTENTIAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label:
            "constraint = fea.MagneticConstraint.prescribedVectorPotential(id, region, potential)",
        inputs: &PRESCRIBED_VECTOR_POTENTIAL_INPUTS,
        outputs: &PRESCRIBED_VECTOR_POTENTIAL_OUTPUTS,
    }];
pub const FEA_MAGNETIC_CONSTRAINT_PRESCRIBED_VECTOR_POTENTIAL_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &PRESCRIBED_VECTOR_POTENTIAL_SIGNATURES,
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
    "fea.MagneticConstraint object.",
)];
const PERIODIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label:
        "constraint = fea.MagneticConstraint.periodic(id, primary, secondary, transform, pairing)",
    inputs: &PERIODIC_INPUTS,
    outputs: &PERIODIC_OUTPUTS,
}];
pub const FEA_MAGNETIC_CONSTRAINT_PERIODIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PERIODIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const PHASE_SHIFTED_PERIODIC_INPUTS: [BuiltinParamDescriptor; 6] = [
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
    required(
        "phase",
        BuiltinParamType::Any,
        "Complex phase shift as fea.ComplexPhase.",
    ),
];
const PHASE_SHIFTED_PERIODIC_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.MagneticConstraint object.",
)];
const PHASE_SHIFTED_PERIODIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.MagneticConstraint.phaseShiftedPeriodic(id, primary, secondary, transform, pairing, phase)",
    inputs: &PHASE_SHIFTED_PERIODIC_INPUTS,
    outputs: &PHASE_SHIFTED_PERIODIC_OUTPUTS,
}];
pub const FEA_MAGNETIC_CONSTRAINT_PHASE_SHIFTED_PERIODIC_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &PHASE_SHIFTED_PERIODIC_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
