use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "2-D convolution result.",
}];
const DIRECT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First matrix input.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second matrix input.",
    },
];
const DIRECT_SHAPE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First matrix input.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second matrix input.",
    },
    BuiltinParamDescriptor {
        name: "shape",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"full\""),
        description: "Output shape: full, same, or valid.",
    },
];
const SEPARABLE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "hcol",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column-vector kernel component.",
    },
    BuiltinParamDescriptor {
        name: "hrow",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row-vector kernel component.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
];
const SEPARABLE_SHAPE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "hcol",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column-vector kernel component.",
    },
    BuiltinParamDescriptor {
        name: "hrow",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row-vector kernel component.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "shape",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"full\""),
        description: "Output shape: full, same, or valid.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "C = conv2(A, B)",
        inputs: &DIRECT,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = conv2(A, B, shape)",
        inputs: &DIRECT_SHAPE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = conv2(hcol, hrow, A)",
        inputs: &SEPARABLE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = conv2(hcol, hrow, A, shape)",
        inputs: &SEPARABLE_SHAPE,
        outputs: &OUTPUT,
    },
];

pub const CONV2_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.ARG_COUNT",
    identifier: Some("RunMat:conv2:ArgCount"),
    when: "More than four inputs are provided.",
    message: "conv2: expected at most four input arguments",
};
pub const CONV2_ERROR_SHAPE_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.SHAPE_INVALID",
    identifier: Some("RunMat:conv2:ShapeInvalid"),
    when: "Shape argument is not full/same/valid.",
    message: "conv2: shape argument must be the string 'full', 'same', or 'valid'",
};
pub const CONV2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.INVALID_INPUT",
    identifier: Some("RunMat:conv2:InvalidInput"),
    when: "An operand is not numeric/logical scalar/vector/matrix compatible.",
    message: "conv2: unsupported input type",
};
pub const CONV2_ERROR_VECTOR_REQUIRED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.VECTOR_REQUIRED",
    identifier: Some("RunMat:conv2:VectorRequired"),
    when: "A separable kernel component is not a vector.",
    message: "conv2: vector input required",
};
pub const CONV2_ERROR_MATRIX_REQUIRED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.MATRIX_REQUIRED",
    identifier: Some("RunMat:conv2:MatrixRequired"),
    when: "A matrix input has non-singleton axes beyond 2-D.",
    message: "conv2: input must be 2-D",
};
pub const CONV2_ERROR_CONVERSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.CONVERSION",
    identifier: Some("RunMat:conv2:Conversion"),
    when: "Input conversion into host matrix domain fails.",
    message: "conv2: input conversion failed",
};
pub const CONV2_ERROR_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.GATHER_FAILED",
    identifier: Some("RunMat:conv2:GatherFailed"),
    when: "GPU input cannot be gathered for host fallback.",
    message: "conv2: failed to gather GPU input",
};
pub const CONV2_ERROR_BUILD_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.BUILD_OUTPUT",
    identifier: Some("RunMat:conv2:BuildOutput"),
    when: "Output tensor allocation fails.",
    message: "conv2: failed to build tensor",
};
pub const CONV2_ERROR_BUILD_COMPLEX_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV2.BUILD_COMPLEX_OUTPUT",
    identifier: Some("RunMat:conv2:BuildComplexOutput"),
    when: "Complex output tensor allocation fails.",
    message: "conv2: failed to build complex tensor",
};
const ERRORS: [BuiltinErrorDescriptor; 9] = [
    CONV2_ERROR_ARG_COUNT,
    CONV2_ERROR_SHAPE_INVALID,
    CONV2_ERROR_INVALID_INPUT,
    CONV2_ERROR_VECTOR_REQUIRED,
    CONV2_ERROR_MATRIX_REQUIRED,
    CONV2_ERROR_CONVERSION,
    CONV2_ERROR_GATHER_FAILED,
    CONV2_ERROR_BUILD_OUTPUT,
    CONV2_ERROR_BUILD_COMPLEX_OUTPUT,
];
pub const CONV2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
