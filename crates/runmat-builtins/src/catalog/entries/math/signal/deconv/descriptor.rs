use crate::*;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "numerator",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficients.",
    },
    BuiltinParamDescriptor {
        name: "denominator",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficients.",
    },
];
const Q: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Quotient polynomial coefficients.",
}];
const QR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Q",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Quotient polynomial coefficients.",
    },
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Remainder polynomial coefficients.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Q = deconv(numerator, denominator)",
        inputs: &INPUTS,
        outputs: &Q,
    },
    BuiltinSignatureDescriptor {
        label: "[Q, R] = deconv(numerator, denominator)",
        inputs: &INPUTS,
        outputs: &QR,
    },
];

pub const DECONV_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.INVALID_INPUT",
    identifier: Some("RunMat:deconv:InvalidInput"),
    when: "An operand cannot be converted to polynomial coefficients.",
    message: "deconv: unsupported input type",
};
pub const DECONV_ERROR_VECTOR_REQUIRED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.VECTOR_REQUIRED",
    identifier: Some("RunMat:deconv:VectorRequired"),
    when: "An input is not a scalar, row, or column vector.",
    message: "deconv: inputs must be scalars, row vectors, or column vectors",
};
pub const DECONV_ERROR_DENOMINATOR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.DENOMINATOR_INVALID",
    identifier: Some("RunMat:deconv:DenominatorInvalid"),
    when: "The denominator is empty or contains only exact zeros.",
    message: "deconv: denominator is invalid",
};
pub const DECONV_ERROR_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.GATHER_FAILED",
    identifier: Some("RunMat:deconv:GatherFailed"),
    when: "A resident input cannot be gathered for division.",
    message: "deconv: failed to gather GPU input",
};
pub const DECONV_ERROR_BUILD_COMPLEX_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.BUILD_COMPLEX_OUTPUT",
    identifier: Some("RunMat:deconv:BuildComplexOutput"),
    when: "Complex output allocation fails.",
    message: "deconv: failed to build complex tensor",
};
pub const DECONV_ERROR_BUILD_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.BUILD_OUTPUT",
    identifier: Some("RunMat:deconv:BuildOutput"),
    when: "Real output allocation fails.",
    message: "deconv: failed to build tensor",
};
pub const DECONV_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:deconv:TooManyOutputs"),
    when: "More than two outputs are requested.",
    message: "deconv: too many output arguments",
};
pub const DECONV_ERROR_LATE_UPLOAD_ADMISSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.LATE_UPLOAD_ADMISSION",
    identifier: Some("RunMat:Placement:LateUploadAdmissionFailed"),
    when: "A computed deconv result exceeds its admitted upload bound or the owning provider rejects its exact upload representation.",
    message: "deconv: result cannot be restored by the admitted provider upload",
};
const ERRORS: [BuiltinErrorDescriptor; 8] = [
    DECONV_ERROR_INVALID_INPUT,
    DECONV_ERROR_VECTOR_REQUIRED,
    DECONV_ERROR_DENOMINATOR_INVALID,
    DECONV_ERROR_GATHER_FAILED,
    DECONV_ERROR_BUILD_COMPLEX_OUTPUT,
    DECONV_ERROR_BUILD_OUTPUT,
    DECONV_ERROR_TOO_MANY_OUTPUTS,
    DECONV_ERROR_LATE_UPLOAD_ADMISSION,
];
pub const DECONV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
