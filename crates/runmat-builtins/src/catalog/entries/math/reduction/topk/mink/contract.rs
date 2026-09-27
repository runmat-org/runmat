use super::super::signatures::*;
use crate::*;
const MINK_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "B = mink(A, k)",
        inputs: &INPUTS_A_K,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = mink(A, k)",
        inputs: &INPUTS_A_K,
        outputs: &OUTPUT_VALUE_INDEX,
    },
    BuiltinSignatureDescriptor {
        label: "B = mink(A, k, dim)",
        inputs: &INPUTS_A_K_DIM,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = mink(A, k, dim)",
        inputs: &INPUTS_A_K_DIM,
        outputs: &OUTPUT_VALUE_INDEX,
    },
    BuiltinSignatureDescriptor {
        label: "B = mink(A, k, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_OPTIONS,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = mink(A, k, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_OPTIONS,
        outputs: &OUTPUT_VALUE_INDEX,
    },
    BuiltinSignatureDescriptor {
        label: "B = mink(A, k, dim, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_DIM_OPTIONS,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = mink(A, k, dim, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_DIM_OPTIONS,
        outputs: &OUTPUT_VALUE_INDEX,
    },
];

pub const MINK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MINK_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &super::super::contract::TOPK_ERRORS,
};
