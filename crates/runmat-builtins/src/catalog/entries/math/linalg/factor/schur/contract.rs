use crate::*;

const A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense square single or double matrix.",
}];
const A_MODE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dense square single or double matrix.",
    },
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real or complex Schur form; complex input always yields complex form.",
    },
];
const T: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real quasitriangular or complex triangular Schur form.",
}];
const U_T: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Orthogonal or unitary Schur vectors.",
    },
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real quasitriangular or complex triangular Schur form.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "T = schur(A)",
        inputs: &A,
        outputs: &T,
    },
    BuiltinSignatureDescriptor {
        label: "T = schur(A,mode)",
        inputs: &A_MODE,
        outputs: &T,
    },
    BuiltinSignatureDescriptor {
        label: "[U,T] = schur(A)",
        inputs: &A,
        outputs: &U_T,
    },
    BuiltinSignatureDescriptor {
        label: "[U,T] = schur(A,mode)",
        inputs: &A_MODE,
        outputs: &U_T,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.SCHUR.INVALID_ARGUMENT",
        identifier: Some("RunMat:schur:InvalidArgument"),
        when: "Mode or output count is invalid.",
        message: "schur: invalid mode or output count",
    },
    BuiltinErrorDescriptor {
        code: "RM.SCHUR.INVALID_INPUT",
        identifier: Some("RunMat:schur:InvalidInput"),
        when: "Input is not a finite dense square floating matrix.",
        message: "schur: input must be a finite square single or double matrix",
    },
    BuiltinErrorDescriptor {
        code: "RM.SCHUR.NUMERICAL",
        identifier: Some("RunMat:schur:Numerical"),
        when: "Bounded Schur iteration or result materialization fails.",
        message: "schur: numerical factorization failed",
    },
];
pub const SCHUR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
