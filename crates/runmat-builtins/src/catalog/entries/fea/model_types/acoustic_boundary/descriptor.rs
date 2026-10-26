//! Signatures, parameters, and integer facts for `fea.AcousticBoundary` constructors.
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

pub const FEA_ACOUSTIC_BOUNDARY_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA boundary constructors accept identifiers, immutable geometry.Region values, and typed physical quantities; they have no integer input role.",
};

const RIGID_WALL_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Boundary identifier (nonempty text).",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "geometry.Region selected from the model geometry.",
    ),
];
const RIGID_WALL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "boundary",
    BuiltinParamType::Any,
    "fea.AcousticBoundary boundary object.",
)];
const RIGID_WALL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "boundary = fea.AcousticBoundary.rigidWall(id, region)",
    inputs: &RIGID_WALL_INPUTS,
    outputs: &RIGID_WALL_OUTPUTS,
}];
pub const FEA_ACOUSTIC_BOUNDARY_RIGID_WALL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RIGID_WALL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const RADIATION_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Boundary identifier (nonempty text).",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "geometry.Region selected from the model geometry.",
    ),
];
const RADIATION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "boundary",
    BuiltinParamType::Any,
    "fea.AcousticBoundary boundary object.",
)];
const RADIATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "boundary = fea.AcousticBoundary.radiation(id, region)",
    inputs: &RADIATION_INPUTS,
    outputs: &RADIATION_OUTPUTS,
}];
pub const FEA_ACOUSTIC_BOUNDARY_RADIATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RADIATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const IMPEDANCE_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Boundary identifier (nonempty text).",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "geometry.Region selected from the model geometry.",
    ),
    required(
        "impedance",
        BuiltinParamType::Any,
        "Specific acoustic impedance as fea.SpecificAcousticImpedance (Pa s/m).",
    ),
];
const IMPEDANCE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "boundary",
    BuiltinParamType::Any,
    "fea.AcousticBoundary boundary object.",
)];
const IMPEDANCE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "boundary = fea.AcousticBoundary.impedance(id, region, impedance)",
    inputs: &IMPEDANCE_INPUTS,
    outputs: &IMPEDANCE_OUTPUTS,
}];
pub const FEA_ACOUSTIC_BOUNDARY_IMPEDANCE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMPEDANCE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
