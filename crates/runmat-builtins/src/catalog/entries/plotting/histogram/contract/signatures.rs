use super::parameters_inputs_ax_x_bins_namevalue::HISTOGRAM_INPUTS_AX_X_BINS_NAMEVALUE;
use super::parameters_output_handle::{
    HISTOGRAM_INPUTS_AX_X, HISTOGRAM_INPUTS_AX_X_BINS, HISTOGRAM_INPUTS_AX_X_NAMEVALUE,
    HISTOGRAM_INPUTS_X, HISTOGRAM_INPUTS_X_BINS, HISTOGRAM_INPUTS_X_BINS_NAMEVALUE,
    HISTOGRAM_INPUTS_X_NAMEVALUE, HISTOGRAM_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const HISTOGRAM_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "h = histogram(X)",
        inputs: &HISTOGRAM_INPUTS_X,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram(X, bins)",
        inputs: &HISTOGRAM_INPUTS_X_BINS,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram(X, Name, Value, ...)",
        inputs: &HISTOGRAM_INPUTS_X_NAMEVALUE,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram(X, bins, Name, Value, ...)",
        inputs: &HISTOGRAM_INPUTS_X_BINS_NAMEVALUE,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram(ax, X)",
        inputs: &HISTOGRAM_INPUTS_AX_X,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram(ax, X, bins)",
        inputs: &HISTOGRAM_INPUTS_AX_X_BINS,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram(ax, X, Name, Value, ...)",
        inputs: &HISTOGRAM_INPUTS_AX_X_NAMEVALUE,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = histogram(ax, X, bins, Name, Value, ...)",
        inputs: &HISTOGRAM_INPUTS_AX_X_BINS_NAMEVALUE,
        outputs: &HISTOGRAM_OUTPUT_HANDLE,
    },
];
