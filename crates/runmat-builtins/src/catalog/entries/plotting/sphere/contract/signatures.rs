use super::parameters_inputs_none::{
    SPHERE_INPUTS_N, SPHERE_INPUTS_NONE, SPHERE_OUTPUT_X, SPHERE_OUTPUT_XY, SPHERE_OUTPUT_XYZ,
};
use crate::*;

pub(super) const SPHERE_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "sphere()",
        inputs: &SPHERE_INPUTS_NONE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "sphere(n)",
        inputs: &SPHERE_INPUTS_N,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "X = sphere()",
        inputs: &SPHERE_INPUTS_NONE,
        outputs: &SPHERE_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "X = sphere(n)",
        inputs: &SPHERE_INPUTS_N,
        outputs: &SPHERE_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = sphere()",
        inputs: &SPHERE_INPUTS_NONE,
        outputs: &SPHERE_OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = sphere(n)",
        inputs: &SPHERE_INPUTS_N,
        outputs: &SPHERE_OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,Z] = sphere()",
        inputs: &SPHERE_INPUTS_NONE,
        outputs: &SPHERE_OUTPUT_XYZ,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,Z] = sphere(n)",
        inputs: &SPHERE_INPUTS_N,
        outputs: &SPHERE_OUTPUT_XYZ,
    },
];
