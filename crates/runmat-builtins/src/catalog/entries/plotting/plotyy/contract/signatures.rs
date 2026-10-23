use super::parameters_output_ax_h1_h2::{
    INPUTS_WITH_FUN, INPUTS_WITH_FUNS, INPUTS_X1_Y1_X2_Y2, OUTPUT_AX_H1_H2,
};
use crate::*;

pub(super) const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "[ax,h1,h2] = plotyy(X1,Y1,X2,Y2)",
        inputs: &INPUTS_X1_Y1_X2_Y2,
        outputs: &OUTPUT_AX_H1_H2,
    },
    BuiltinSignatureDescriptor {
        label: "[ax,h1,h2] = plotyy(X1,Y1,X2,Y2,fun)",
        inputs: &INPUTS_WITH_FUN,
        outputs: &OUTPUT_AX_H1_H2,
    },
    BuiltinSignatureDescriptor {
        label: "[ax,h1,h2] = plotyy(X1,Y1,X2,Y2,fun1,fun2)",
        inputs: &INPUTS_WITH_FUNS,
        outputs: &OUTPUT_AX_H1_H2,
    },
];
