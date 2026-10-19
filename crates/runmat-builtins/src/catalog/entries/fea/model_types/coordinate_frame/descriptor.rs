//! Signatures, parameters, and integer facts for `fea.CoordinateFrame` constructors.
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

pub const FEA_COORDINATE_FRAME_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Concrete constraint constructors accept identifiers, immutable geometry regions, and already-dimensioned values; they have no direct integer input role.",
};

const GLOBAL_FRAME_INPUTS: [BuiltinParamDescriptor; 0] = [];
const GLOBAL_FRAME_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "frame",
    BuiltinParamType::Any,
    "fea.GlobalCoordinateFrame object.",
)];
const GLOBAL_FRAME_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "frame = fea.CoordinateFrame.globalFrame()",
    inputs: &GLOBAL_FRAME_INPUTS,
    outputs: &GLOBAL_FRAME_OUTPUTS,
}];
pub const FEA_COORDINATE_FRAME_GLOBAL_FRAME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GLOBAL_FRAME_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const MATERIAL_INPUTS: [BuiltinParamDescriptor; 0] = [];
const MATERIAL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "frame",
    BuiltinParamType::Any,
    "fea.MaterialCoordinateFrame object.",
)];
const MATERIAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "frame = fea.CoordinateFrame.material()",
    inputs: &MATERIAL_INPUTS,
    outputs: &MATERIAL_OUTPUTS,
}];
pub const FEA_COORDINATE_FRAME_MATERIAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATERIAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const ELEMENT_INPUTS: [BuiltinParamDescriptor; 0] = [];
const ELEMENT_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "frame",
    BuiltinParamType::Any,
    "fea.ElementCoordinateFrame object.",
)];
const ELEMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "frame = fea.CoordinateFrame.element()",
    inputs: &ELEMENT_INPUTS,
    outputs: &ELEMENT_OUTPUTS,
}];
pub const FEA_COORDINATE_FRAME_ELEMENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ELEMENT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const BOUNDARY_NORMAL_TANGENT_INPUTS: [BuiltinParamDescriptor; 0] = [];
const BOUNDARY_NORMAL_TANGENT_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "frame",
    BuiltinParamType::Any,
    "fea.BoundaryNormalTangentFrame object.",
)];
const BOUNDARY_NORMAL_TANGENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "frame = fea.CoordinateFrame.boundaryNormalTangent()",
        inputs: &BOUNDARY_NORMAL_TANGENT_INPUTS,
        outputs: &BOUNDARY_NORMAL_TANGENT_OUTPUTS,
    }];
pub const FEA_COORDINATE_FRAME_BOUNDARY_NORMAL_TANGENT_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &BOUNDARY_NORMAL_TANGENT_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const CARTESIAN_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Coordinate-frame identifier (nonempty text).",
    ),
    required(
        "origin",
        BuiltinParamType::Any,
        "Frame origin as fea.Position (m).",
    ),
    required(
        "xAxis",
        BuiltinParamType::Any,
        "Primary axis as fea.Direction.",
    ),
    required(
        "yAxis",
        BuiltinParamType::Any,
        "Secondary reference direction as fea.Direction, not parallel to the primary axis.",
    ),
];
const CARTESIAN_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "frame",
    BuiltinParamType::Any,
    "fea.CartesianCoordinateFrame object.",
)];
const CARTESIAN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "frame = fea.CoordinateFrame.cartesian(id, origin, xAxis, yAxis)",
    inputs: &CARTESIAN_INPUTS,
    outputs: &CARTESIAN_OUTPUTS,
}];
pub const FEA_COORDINATE_FRAME_CARTESIAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CARTESIAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const CYLINDRICAL_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Coordinate-frame identifier (nonempty text).",
    ),
    required(
        "origin",
        BuiltinParamType::Any,
        "Frame origin as fea.Position (m).",
    ),
    required(
        "axis",
        BuiltinParamType::Any,
        "Primary axis as fea.Direction.",
    ),
    required(
        "radialReference",
        BuiltinParamType::Any,
        "Secondary reference direction as fea.Direction, not parallel to the primary axis.",
    ),
];
const CYLINDRICAL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "frame",
    BuiltinParamType::Any,
    "fea.CylindricalCoordinateFrame object.",
)];
const CYLINDRICAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "frame = fea.CoordinateFrame.cylindrical(id, origin, axis, radialReference)",
    inputs: &CYLINDRICAL_INPUTS,
    outputs: &CYLINDRICAL_OUTPUTS,
}];
pub const FEA_COORDINATE_FRAME_CYLINDRICAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CYLINDRICAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const SPHERICAL_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Coordinate-frame identifier (nonempty text).",
    ),
    required(
        "origin",
        BuiltinParamType::Any,
        "Frame origin as fea.Position (m).",
    ),
    required(
        "polarAxis",
        BuiltinParamType::Any,
        "Primary axis as fea.Direction.",
    ),
    required(
        "azimuthReference",
        BuiltinParamType::Any,
        "Secondary reference direction as fea.Direction, not parallel to the primary axis.",
    ),
];
const SPHERICAL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "frame",
    BuiltinParamType::Any,
    "fea.SphericalCoordinateFrame object.",
)];
const SPHERICAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "frame = fea.CoordinateFrame.spherical(id, origin, polarAxis, azimuthReference)",
    inputs: &SPHERICAL_INPUTS,
    outputs: &SPHERICAL_OUTPUTS,
}];
pub const FEA_COORDINATE_FRAME_SPHERICAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPHERICAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
