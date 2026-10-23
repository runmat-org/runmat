use super::parameters_output_handle::{
    IMSHOW_INPUTS_IMAGE, IMSHOW_INPUTS_IMAGE_RANGE, IMSHOW_OUTPUT_HANDLE,
};
use crate::*;

pub(super) const IMSHOW_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "h = imshow(I)",
        inputs: &IMSHOW_INPUTS_IMAGE,
        outputs: &IMSHOW_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = imshow(I, [low high])",
        inputs: &IMSHOW_INPUTS_IMAGE_RANGE,
        outputs: &IMSHOW_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = imshow(I, [])",
        inputs: &IMSHOW_INPUTS_IMAGE_RANGE,
        outputs: &IMSHOW_OUTPUT_HANDLE,
    },
];
