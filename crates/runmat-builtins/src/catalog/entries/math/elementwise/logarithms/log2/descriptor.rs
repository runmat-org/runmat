use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input or a supported table; integer, logical, and character forms are RunMat-only extensions.",
}];
const VALUE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise base-2 logarithm, promoted to complex for negative real input.",
}];
const DISSECTION_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "F",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Real floating-point fraction with the same class and shape as the supported input.",
    },
    BuiltinParamDescriptor {
        name: "E",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real floating-point exponent satisfying X = F .* 2.^E.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = log2(X)",
        inputs: &INPUTS,
        outputs: &VALUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[F,E] = log2(X)",
        inputs: &INPUTS,
        outputs: &DISSECTION_OUTPUTS,
    },
];

pub const LOG2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.INVALID_INPUT",
    identifier: Some("RunMat:log2:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric, logical, character, or table data.",
    message: "log2: invalid input",
};
pub const LOG2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.INTERNAL",
    identifier: Some("RunMat:log2:Internal"),
    when: "Internal tensor construction, table mapping, or provider interaction fails.",
    message: "log2: internal error",
};
pub const LOG2_ERROR_COMPLEX_DISSECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.COMPLEX_DISSECTION",
    identifier: Some("RunMat:log2:ComplexDissection"),
    when: "Complex input is supplied to the two-output floating-point dissection form.",
    message: "log2: two-output dissection requires real input",
};
pub const LOG2_ERROR_GPU_DISSECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.GPU_DISSECTION",
    identifier: Some("RunMat:log2:GpuDissectionUnsupported"),
    when: "A GPU-resident input is supplied to the two-output floating-point dissection form.",
    message: "log2: two-output dissection does not support gpuArray input",
};
pub const LOG2_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:log2:TooManyOutputs"),
    when: "More than two outputs are requested.",
    message: "log2: at most two outputs are available",
};
pub const LOG2_ERROR_PROVIDER_OWNERSHIP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.PROVIDER_OWNERSHIP_MISMATCH",
    identifier: Some("RunMat:gpu:ProviderOwnershipMismatch"),
    when: "A resident input has no exact owning provider.",
    message: "log2: resident input has no exact owning provider",
};
pub const LOG2_ERROR_GPU_COMPLEX_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.GPU_COMPLEX_INPUT_REQUIRED",
    identifier: Some("RunMat:log2:GpuComplexInputRequired"),
    when: "Explicitly resident real input would require a complex result.",
    message: "log2: real gpuArray input must be explicitly complex when the result can be complex",
};

const ERRORS: [BuiltinErrorDescriptor; 7] = [
    LOG2_ERROR_INVALID_INPUT,
    LOG2_ERROR_INTERNAL,
    LOG2_ERROR_COMPLEX_DISSECTION,
    LOG2_ERROR_GPU_DISSECTION,
    LOG2_ERROR_TOO_MANY_OUTPUTS,
    LOG2_ERROR_PROVIDER_OWNERSHIP,
    LOG2_ERROR_GPU_COMPLEX_INPUT,
];
pub const LOG2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
