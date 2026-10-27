//! Original frame descriptors lifted from the Runtime constructor definitions.
use crate::fea_contract::{descriptors::ERRORS, parameters::OUT_ANY};
use crate::*;

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const FRAME_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Coordinate-frame identifier.",
    },
    BuiltinParamDescriptor {
        name: "origin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.Position origin.",
    },
    BuiltinParamDescriptor {
        name: "primaryAxis",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.Direction primary axis.",
    },
    BuiltinParamDescriptor {
        name: "secondaryAxis",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.Direction nonparallel reference axis.",
    },
];

macro_rules! descriptor {
    ($descriptor:ident, $signatures:ident, $label:literal, $inputs:ident) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &$inputs,
            outputs: &OUT_ANY,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}
descriptor!(
    GLOBAL_FRAME_DESCRIPTOR,
    GLOBAL_FRAME_SIGNATURES,
    "frame = fea.CoordinateFrame.globalFrame()",
    NO_INPUTS
);
descriptor!(
    MATERIAL_FRAME_DESCRIPTOR,
    MATERIAL_FRAME_SIGNATURES,
    "frame = fea.CoordinateFrame.material()",
    NO_INPUTS
);
descriptor!(
    ELEMENT_FRAME_DESCRIPTOR,
    ELEMENT_FRAME_SIGNATURES,
    "frame = fea.CoordinateFrame.element()",
    NO_INPUTS
);
descriptor!(
    BOUNDARY_FRAME_DESCRIPTOR,
    BOUNDARY_FRAME_SIGNATURES,
    "frame = fea.CoordinateFrame.boundaryNormalTangent()",
    NO_INPUTS
);
descriptor!(
    CARTESIAN_FRAME_DESCRIPTOR,
    CARTESIAN_FRAME_SIGNATURES,
    "frame = fea.CoordinateFrame.cartesian(id, origin, xAxis, yAxis)",
    FRAME_INPUTS
);
descriptor!(
    CYLINDRICAL_FRAME_DESCRIPTOR,
    CYLINDRICAL_FRAME_SIGNATURES,
    "frame = fea.CoordinateFrame.cylindrical(id, origin, axis, radialReference)",
    FRAME_INPUTS
);
descriptor!(
    SPHERICAL_FRAME_DESCRIPTOR,
    SPHERICAL_FRAME_SIGNATURES,
    "frame = fea.CoordinateFrame.spherical(id, origin, polarAxis, azimuthReference)",
    FRAME_INPUTS
);
