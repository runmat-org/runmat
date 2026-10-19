//! Signatures, parameters, and integer facts for `fea.FluidConstraint` constructors.
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

pub const FEA_FLUID_CONSTRAINT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Concrete constraint constructors accept identifiers, immutable geometry regions, and already-dimensioned values; they have no direct integer input role.",
};

const NO_SLIP_WALL_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const NO_SLIP_WALL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.FluidConstraint object.",
)];
const NO_SLIP_WALL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.FluidConstraint.noSlipWall(id, region)",
    inputs: &NO_SLIP_WALL_INPUTS,
    outputs: &NO_SLIP_WALL_OUTPUTS,
}];
pub const FEA_FLUID_CONSTRAINT_NO_SLIP_WALL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NO_SLIP_WALL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const SLIP_WALL_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const SLIP_WALL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.FluidConstraint object.",
)];
const SLIP_WALL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.FluidConstraint.slipWall(id, region)",
    inputs: &SLIP_WALL_INPUTS,
    outputs: &SLIP_WALL_OUTPUTS,
}];
pub const FEA_FLUID_CONSTRAINT_SLIP_WALL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SLIP_WALL_SIGNATURES,
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
    "fea.FluidConstraint object.",
)];
const SYMMETRY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.FluidConstraint.symmetry(id, region)",
    inputs: &SYMMETRY_INPUTS,
    outputs: &SYMMETRY_OUTPUTS,
}];
pub const FEA_FLUID_CONSTRAINT_SYMMETRY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SYMMETRY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const PRESCRIBED_VELOCITY_INPUTS: [BuiltinParamDescriptor; 3] = [
    required("id", BuiltinParamType::StringScalar, "Constraint identifier (nonempty text)."),
    required("region", BuiltinParamType::Any, "geometry.Region selected from the model geometry."),
    required("velocity", BuiltinParamType::Any, "Three fea.Velocity components in a cell vector (global frame), or a framed three-component fea.VelocitySignal."),
];
const PRESCRIBED_VELOCITY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.FluidConstraint object.",
)];
const PRESCRIBED_VELOCITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "constraint = fea.FluidConstraint.prescribedVelocity(id, region, velocity)",
        inputs: &PRESCRIBED_VELOCITY_INPUTS,
        outputs: &PRESCRIBED_VELOCITY_OUTPUTS,
    }];
pub const FEA_FLUID_CONSTRAINT_PRESCRIBED_VELOCITY_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &PRESCRIBED_VELOCITY_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const MOVING_WALL_INPUTS: [BuiltinParamDescriptor; 3] = [
    required("id", BuiltinParamType::StringScalar, "Constraint identifier (nonempty text)."),
    required("region", BuiltinParamType::Any, "geometry.Region selected from the model geometry."),
    required("velocity", BuiltinParamType::Any, "Three fea.Velocity components in a cell vector (global frame), or a framed three-component fea.VelocitySignal."),
];
const MOVING_WALL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.FluidConstraint object.",
)];
const MOVING_WALL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.FluidConstraint.movingWall(id, region, velocity)",
    inputs: &MOVING_WALL_INPUTS,
    outputs: &MOVING_WALL_OUTPUTS,
}];
pub const FEA_FLUID_CONSTRAINT_MOVING_WALL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MOVING_WALL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const PRESSURE_REFERENCE_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "pressure",
        BuiltinParamType::Any,
        "Reference pressure as fea.Pressure (Pa).",
    ),
];
const PRESSURE_REFERENCE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "constraint",
    BuiltinParamType::Any,
    "fea.FluidConstraint object.",
)];
const PRESSURE_REFERENCE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "constraint = fea.FluidConstraint.pressureReference(id, region, pressure)",
        inputs: &PRESSURE_REFERENCE_INPUTS,
        outputs: &PRESSURE_REFERENCE_OUTPUTS,
    }];
pub const FEA_FLUID_CONSTRAINT_PRESSURE_REFERENCE_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &PRESSURE_REFERENCE_SIGNATURES,
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
    "fea.FluidConstraint object.",
)];
const PERIODIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.FluidConstraint.periodic(id, primary, secondary, transform, pairing)",
    inputs: &PERIODIC_INPUTS,
    outputs: &PERIODIC_OUTPUTS,
}];
pub const FEA_FLUID_CONSTRAINT_PERIODIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PERIODIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
