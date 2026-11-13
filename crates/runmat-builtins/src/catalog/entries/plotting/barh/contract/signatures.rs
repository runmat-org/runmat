use super::parameters_bar_inputs_x_y_width_color::{
    BAR_INPUTS_X_Y_PROPS, BAR_INPUTS_X_Y_WIDTH_COLOR,
};
use super::parameters_bar_output_handle::{
    BAR_INPUTS_X_Y, BAR_INPUTS_X_Y_STYLE, BAR_INPUTS_X_Y_WIDTH, BAR_INPUTS_Y, BAR_INPUTS_Y_PROPS,
    BAR_INPUTS_Y_STYLE, BAR_INPUTS_Y_WIDTH, BAR_INPUTS_Y_WIDTH_COLOR, BAR_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const BARH_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "h = barh(Y)",
        inputs: &BAR_INPUTS_Y,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(Y, width)",
        inputs: &BAR_INPUTS_Y_WIDTH,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(Y, width, color)",
        inputs: &BAR_INPUTS_Y_WIDTH_COLOR,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(Y, style)",
        inputs: &BAR_INPUTS_Y_STYLE,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(Y, Name, Value, ...)",
        inputs: &BAR_INPUTS_Y_PROPS,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(X, Y)",
        inputs: &BAR_INPUTS_X_Y,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(X, Y, width)",
        inputs: &BAR_INPUTS_X_Y_WIDTH,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(X, Y, width, color)",
        inputs: &BAR_INPUTS_X_Y_WIDTH_COLOR,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(X, Y, style)",
        inputs: &BAR_INPUTS_X_Y_STYLE,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = barh(X, Y, Name, Value, ...)",
        inputs: &BAR_INPUTS_X_Y_PROPS,
        outputs: &BAR_OUTPUT_HANDLE,
    },
];
