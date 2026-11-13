use super::parameters_output_counts::{
    HIST_INPUTS_X, HIST_INPUTS_X_BINS, HIST_INPUTS_X_BINS_NAMEVALUE, HIST_INPUTS_X_NAMEVALUE,
    HIST_INPUTS_X_NORMALIZATION, HIST_OUTPUT_COUNTS, HIST_OUTPUT_COUNTS_CENTERS,
};
use crate::*;

pub(super) const HIST_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "N = hist(X)",
        inputs: &HIST_INPUTS_X,
        outputs: &HIST_OUTPUT_COUNTS,
    },
    BuiltinSignatureDescriptor {
        label: "N = hist(X, bins)",
        inputs: &HIST_INPUTS_X_BINS,
        outputs: &HIST_OUTPUT_COUNTS,
    },
    BuiltinSignatureDescriptor {
        label: "N = hist(X, normalization)",
        inputs: &HIST_INPUTS_X_NORMALIZATION,
        outputs: &HIST_OUTPUT_COUNTS,
    },
    BuiltinSignatureDescriptor {
        label: "N = hist(X, Name, Value, ...)",
        inputs: &HIST_INPUTS_X_NAMEVALUE,
        outputs: &HIST_OUTPUT_COUNTS,
    },
    BuiltinSignatureDescriptor {
        label: "N = hist(X, bins, Name, Value, ...)",
        inputs: &HIST_INPUTS_X_BINS_NAMEVALUE,
        outputs: &HIST_OUTPUT_COUNTS,
    },
    BuiltinSignatureDescriptor {
        label: "[counts, centers] = hist(X)",
        inputs: &HIST_INPUTS_X,
        outputs: &HIST_OUTPUT_COUNTS_CENTERS,
    },
    BuiltinSignatureDescriptor {
        label: "[counts, centers] = hist(X, bins)",
        inputs: &HIST_INPUTS_X_BINS,
        outputs: &HIST_OUTPUT_COUNTS_CENTERS,
    },
];
