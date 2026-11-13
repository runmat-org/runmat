use crate::*;

const fn input(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

const fn output(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    input(name, description)
}

const A_B: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite dense square single or double coefficient matrix.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second coefficient matrix of the same order.",
    },
];
const A_B_MODE: [BuiltinParamDescriptor; 3] = [
    input(
        "A",
        "Finite dense square single or double coefficient matrix.",
    ),
    input("B", "Second coefficient matrix of the same order."),
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "'complex' (default) or 'real'; real mode requires real coefficients.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 6] = [
    output(
        "AA",
        "Complex upper triangular or real upper quasitriangular factor.",
    ),
    output(
        "BB",
        "Upper triangular second factor; singular B is admitted.",
    ),
    output(
        "Q",
        "Left unitary or orthogonal transform satisfying Q*A*Z=AA.",
    ),
    output(
        "Z",
        "Right unitary or orthogonal transform satisfying Q*B*Z=BB.",
    ),
    output("V", "Right generalized eigenvectors in Schur order."),
    output("W", "Left generalized eigenvectors in Schur order."),
];
const OUTPUT_AA: [BuiltinParamDescriptor; 1] = [output(
    "AA",
    "Complex upper triangular or real upper quasitriangular factor.",
)];
const OUTPUT_4: [BuiltinParamDescriptor; 4] = [
    output(
        "AA",
        "Complex upper triangular or real upper quasitriangular factor.",
    ),
    output(
        "BB",
        "Upper triangular second factor; singular B is admitted.",
    ),
    output(
        "Q",
        "Left unitary or orthogonal transform satisfying Q*A*Z=AA.",
    ),
    output(
        "Z",
        "Right unitary or orthogonal transform satisfying Q*B*Z=BB.",
    ),
];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "AA = qz(A,B)",
        inputs: &A_B,
        outputs: &OUTPUT_AA,
    },
    BuiltinSignatureDescriptor {
        label: "[AA,BB,Q,Z] = qz(A,B)",
        inputs: &A_B,
        outputs: &OUTPUT_4,
    },
    BuiltinSignatureDescriptor {
        label: "[AA,BB,Q,Z,V,W] = qz(A,B)",
        inputs: &A_B,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[AA,BB,Q,Z] = qz(A,B,mode)",
        inputs: &A_B_MODE,
        outputs: &OUTPUT_4,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.QZ.INVALID_INPUT",
        identifier: Some("RunMat:qz:InvalidInput"),
        when: "Coefficient class, shape, mode, output count, or placement is unsupported.",
        message: "qz: expected a finite same-order dense floating matrix pair",
    },
    BuiltinErrorDescriptor {
        code: "RM.QZ.NUMERICAL",
        identifier: Some("RunMat:qz:Numerical"),
        when: "QZ or a retained-factor eigenvector solve cannot be certified.",
        message: "qz: generalized Schur factorization failed",
    },
];
pub const QZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
