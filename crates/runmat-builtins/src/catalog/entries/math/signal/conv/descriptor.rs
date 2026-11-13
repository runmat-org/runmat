use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Convolution result.",
}];

const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "First input vector/scalar.",
};
const INPUT_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second input vector/scalar.",
};
const INPUTS: [BuiltinParamDescriptor; 2] = [INPUT_A, INPUT_B];
const SHAPE_INPUTS: [BuiltinParamDescriptor; 3] = [
    INPUT_A,
    INPUT_B,
    BuiltinParamDescriptor {
        name: "shape",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"full\""),
        description: "Output shape: full, same, or valid.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = conv(A, B)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = conv(A, B, shape)",
        inputs: &SHAPE_INPUTS,
        outputs: &OUTPUT,
    },
];

pub const CONV_ERROR_SHAPE_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV.SHAPE_INVALID",
    identifier: Some("RunMat:conv:ShapeInvalid"),
    when: "Third argument is missing or not one of full/same/valid.",
    message: "conv: third argument must be the string 'full', 'same', or 'valid'",
};
pub const CONV_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV.ARG_COUNT",
    identifier: Some("RunMat:conv:ArgCount"),
    when: "More than three input arguments are provided.",
    message: "conv: expected at most three input arguments",
};
pub const CONV_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV.INVALID_INPUT",
    identifier: Some("RunMat:conv:InvalidInput"),
    when: "An input value is not numeric/logical scalar/vector/tensor compatible.",
    message: "conv: unsupported input type",
};
pub const CONV_ERROR_CONVERSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV.CONVERSION",
    identifier: Some("RunMat:conv:Conversion"),
    when: "Input conversion from logical/gpu tensor into host numeric domain fails.",
    message: "conv: input conversion failed",
};
pub const CONV_ERROR_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV.GATHER_FAILED",
    identifier: Some("RunMat:conv:GatherFailed"),
    when: "GPU input cannot be gathered for host fallback normalization.",
    message: "conv: failed to gather GPU input",
};
pub const CONV_ERROR_BUILD_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV.BUILD_OUTPUT",
    identifier: Some("RunMat:conv:BuildOutput"),
    when: "Output tensor allocation fails.",
    message: "conv: failed to build tensor",
};
pub const CONV_ERROR_BUILD_COMPLEX_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV.BUILD_COMPLEX_OUTPUT",
    identifier: Some("RunMat:conv:BuildComplexOutput"),
    when: "Output complex tensor allocation fails.",
    message: "conv: failed to build complex tensor",
};
const ERRORS: [BuiltinErrorDescriptor; 7] = [
    CONV_ERROR_SHAPE_INVALID,
    CONV_ERROR_ARG_COUNT,
    CONV_ERROR_INVALID_INPUT,
    CONV_ERROR_CONVERSION,
    CONV_ERROR_GATHER_FAILED,
    CONV_ERROR_BUILD_OUTPUT,
    CONV_ERROR_BUILD_COMPLEX_OUTPUT,
];

pub const CONV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
