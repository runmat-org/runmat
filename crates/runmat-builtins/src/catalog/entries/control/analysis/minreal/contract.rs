use crate::*;

const SYS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar SISO transfer-function model.",
};
const TOL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tol",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("sqrt(eps)"),
    description: "Positive absolute pole-zero matching tolerance.",
};
const RESULT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "msys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Reduced transfer-function model with unchanged sample time and delays.",
};
const INPUTS_ONE: [BuiltinParamDescriptor; 1] = [SYS];
const INPUTS_TWO: [BuiltinParamDescriptor; 2] = [SYS, TOL];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [RESULT];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "msys = minreal(sys)",
        inputs: &INPUTS_ONE,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "msys = minreal(sys,tol)",
        inputs: &INPUTS_TWO,
        outputs: &OUTPUTS,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 10] = [
    BuiltinErrorDescriptor {
        code: "RM.MINREAL.INVALID_ARGUMENT",
        identifier: Some("RunMat:minreal:InvalidArgument"),
        when: "The arity, output count, model argument, or tolerance is invalid.",
        message: "minreal: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.MINREAL.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:minreal:UnsupportedModel"),
        when: "The input is not an ordinary SISO tf model.",
        message: "minreal: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.MINREAL.NUMERICAL_FAILURE",
        identifier: Some("RunMat:minreal:NumericalFailure"),
        when: "Polynomial root finding or pole-zero deflation is unreliable.",
        message: "minreal: numerical failure",
    },
    // Parsing and serialization retain the shared model owner's diagnostics;
    // declaring them here avoids reclassifying provider/control failures.
    TF_ERROR_INVALID_ARGUMENT,
    TF_ERROR_INVALID_COEFFICIENTS,
    TF_ERROR_INVALID_SAMPLE_TIME,
    TF_ERROR_INVALID_VARIABLE,
    TF_ERROR_INVALID_MODEL,
    TF_ERROR_UNSUPPORTED_MODEL,
    TF_ERROR_INTERNAL,
];
pub const MINREAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
