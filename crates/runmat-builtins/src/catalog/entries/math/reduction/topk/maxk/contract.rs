use super::super::signatures::*;
use crate::*;
const MAXK_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "B = maxk(A, k)",
        inputs: &INPUTS_A_K,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = maxk(A, k)",
        inputs: &INPUTS_A_K,
        outputs: &OUTPUT_VALUE_INDEX,
    },
    BuiltinSignatureDescriptor {
        label: "B = maxk(A, k, dim)",
        inputs: &INPUTS_A_K_DIM,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = maxk(A, k, dim)",
        inputs: &INPUTS_A_K_DIM,
        outputs: &OUTPUT_VALUE_INDEX,
    },
    BuiltinSignatureDescriptor {
        label: "B = maxk(A, k, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_OPTIONS,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = maxk(A, k, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_OPTIONS,
        outputs: &OUTPUT_VALUE_INDEX,
    },
    BuiltinSignatureDescriptor {
        label: "B = maxk(A, k, dim, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_DIM_OPTIONS,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = maxk(A, k, dim, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_DIM_OPTIONS,
        outputs: &OUTPUT_VALUE_INDEX,
    },
];

pub const MAXK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MAXK_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &super::super::contract::TOPK_ERRORS,
};
