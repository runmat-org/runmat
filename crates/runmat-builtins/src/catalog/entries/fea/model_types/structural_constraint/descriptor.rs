//! Signatures, parameters, and integer facts for `fea.StructuralConstraint` constructors.
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

pub const FEA_STRUCTURAL_CONSTRAINT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Concrete constraint constructors accept identifiers, immutable geometry regions, and already-dimensioned values; they have no direct integer input role.",
};

pub const FEA_STRUCTURAL_CONSTRAINT_FIXED_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "The fixed structural-constraint constructor accepts an identifier and an immutable geometry.Region; it has no integer input role.",
};

const FIXED_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const FIXED_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.StructuralConstraint object.",
)];
const FIXED_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.StructuralConstraint.fixed(id, region)",
    inputs: &FIXED_INPUTS,
    outputs: &FIXED_OUTPUTS,
}];
pub const FEA_STRUCTURAL_CONSTRAINT_FIXED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIXED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const PRESCRIBED_INPUTS: [BuiltinParamDescriptor; 4] = [
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
        "values",
        BuiltinParamType::Any,
        "One fea.StructuralDofValue or a cell array of them.",
    ),
    required(
        "frame",
        BuiltinParamType::Any,
        "fea.CoordinateFrame in which the components are measured.",
    ),
];
const PRESCRIBED_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.StructuralConstraint object.",
)];
const PRESCRIBED_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.StructuralConstraint.prescribed(id, region, values, frame)",
    inputs: &PRESCRIBED_INPUTS,
    outputs: &PRESCRIBED_OUTPUTS,
}];
pub const FEA_STRUCTURAL_CONSTRAINT_PRESCRIBED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PRESCRIBED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const PRESCRIBED_MOTION_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "motions",
        BuiltinParamType::Any,
        "One fea.StructuralMotion or a cell array of component motions.",
    ),
];
const PRESCRIBED_MOTION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.StructuralConstraint object.",
)];
const PRESCRIBED_MOTION_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "constraint = fea.StructuralConstraint.prescribedMotion(id, region, motions)",
        inputs: &PRESCRIBED_MOTION_INPUTS,
        outputs: &PRESCRIBED_MOTION_OUTPUTS,
    }];
pub const FEA_STRUCTURAL_CONSTRAINT_PRESCRIBED_MOTION_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &PRESCRIBED_MOTION_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const SYMMETRY_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const SYMMETRY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.StructuralConstraint object.",
)];
const SYMMETRY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.StructuralConstraint.symmetry(id, region)",
    inputs: &SYMMETRY_INPUTS,
    outputs: &SYMMETRY_OUTPUTS,
}];
pub const FEA_STRUCTURAL_CONSTRAINT_SYMMETRY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SYMMETRY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const ANTISYMMETRY_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const ANTISYMMETRY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.StructuralConstraint object.",
)];
const ANTISYMMETRY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.StructuralConstraint.antisymmetry(id, region)",
    inputs: &ANTISYMMETRY_INPUTS,
    outputs: &ANTISYMMETRY_OUTPUTS,
}];
pub const FEA_STRUCTURAL_CONSTRAINT_ANTISYMMETRY_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &ANTISYMMETRY_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const NORMAL_DISPLACEMENT_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "displacement",
        BuiltinParamType::Any,
        "Normal displacement as fea.Displacement (m).",
    ),
];
const NORMAL_DISPLACEMENT_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.StructuralConstraint object.",
)];
const NORMAL_DISPLACEMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "constraint = fea.StructuralConstraint.normalDisplacement(id, region, displacement)",
        inputs: &NORMAL_DISPLACEMENT_INPUTS,
        outputs: &NORMAL_DISPLACEMENT_OUTPUTS,
    }];
pub const FEA_STRUCTURAL_CONSTRAINT_NORMAL_DISPLACEMENT_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &NORMAL_DISPLACEMENT_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const RIGID_COUPLING_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "referencePoint",
        BuiltinParamType::Any,
        "fea.ReferencePoint that drives the region.",
    ),
];
const RIGID_COUPLING_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.StructuralConstraint object.",
)];
const RIGID_COUPLING_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.StructuralConstraint.rigidCoupling(id, region, referencePoint)",
    inputs: &RIGID_COUPLING_INPUTS,
    outputs: &RIGID_COUPLING_OUTPUTS,
}];
pub const FEA_STRUCTURAL_CONSTRAINT_RIGID_COUPLING_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &RIGID_COUPLING_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const DISTRIBUTING_COUPLING_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "referencePoint",
        BuiltinParamType::Any,
        "fea.ReferencePoint whose load is distributed to the region.",
    ),
];
const DISTRIBUTING_COUPLING_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.StructuralConstraint object.",
)];
const DISTRIBUTING_COUPLING_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [
    BuiltinSignatureDescriptor {
        label:
            "constraint = fea.StructuralConstraint.distributingCoupling(id, region, referencePoint)",
        inputs: &DISTRIBUTING_COUPLING_INPUTS,
        outputs: &DISTRIBUTING_COUPLING_OUTPUTS,
    },
];
pub const FEA_STRUCTURAL_CONSTRAINT_DISTRIBUTING_COUPLING_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &DISTRIBUTING_COUPLING_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const TIE_INPUTS: [BuiltinParamDescriptor; 4] = [
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
        "pairing",
        BuiltinParamType::Any,
        "fea.PeriodicPairing policy that matches primary and secondary entities.",
    ),
];
const TIE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.StructuralConstraint object.",
)];
const TIE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.StructuralConstraint.tie(id, primary, secondary, pairing)",
    inputs: &TIE_INPUTS,
    outputs: &TIE_OUTPUTS,
}];
pub const FEA_STRUCTURAL_CONSTRAINT_TIE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIE_SIGNATURES,
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
    "fea.StructuralConstraint object.",
)];
const PERIODIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label:
        "constraint = fea.StructuralConstraint.periodic(id, primary, secondary, transform, pairing)",
    inputs: &PERIODIC_INPUTS,
    outputs: &PERIODIC_OUTPUTS,
}];
pub const FEA_STRUCTURAL_CONSTRAINT_PERIODIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PERIODIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
