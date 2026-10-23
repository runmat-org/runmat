use super::parameters_inputs_x_y_width_color::{BAR_INPUTS_X_Y_PROPS, BAR_INPUTS_X_Y_WIDTH_COLOR};
use super::parameters_output_handle::{
    BAR_INPUTS_X_Y, BAR_INPUTS_X_Y_STYLE, BAR_INPUTS_X_Y_WIDTH, BAR_INPUTS_Y, BAR_INPUTS_Y_PROPS,
    BAR_INPUTS_Y_STYLE, BAR_INPUTS_Y_WIDTH, BAR_INPUTS_Y_WIDTH_COLOR, BAR_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const BAR_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "h = bar(Y)",
        inputs: &BAR_INPUTS_Y,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(Y, width)",
        inputs: &BAR_INPUTS_Y_WIDTH,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(Y, width, color)",
        inputs: &BAR_INPUTS_Y_WIDTH_COLOR,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(Y, style)",
        inputs: &BAR_INPUTS_Y_STYLE,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(Y, Name, Value, ...)",
        inputs: &BAR_INPUTS_Y_PROPS,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(X, Y)",
        inputs: &BAR_INPUTS_X_Y,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(X, Y, width)",
        inputs: &BAR_INPUTS_X_Y_WIDTH,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(X, Y, width, color)",
        inputs: &BAR_INPUTS_X_Y_WIDTH_COLOR,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(X, Y, style)",
        inputs: &BAR_INPUTS_X_Y_STYLE,
        outputs: &BAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = bar(X, Y, Name, Value, ...)",
        inputs: &BAR_INPUTS_X_Y_PROPS,
        outputs: &BAR_OUTPUT_HANDLE,
    },
];
