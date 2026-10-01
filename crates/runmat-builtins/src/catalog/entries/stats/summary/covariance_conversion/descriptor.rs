use crate::*;
const OUTPUT_R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Correlation matrix corresponding to the covariance matrix.",
};

const OUTPUT_SIGMA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sigma",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Column vector of standard deviations computed from diag(C).",
};

const OUTPUT_EXP_SIGMA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ExpSigma",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row vector of standard deviations computed from diag(ExpCovariance).",
};

const OUTPUT_EXP_CORR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ExpCorrC",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Correlation matrix corresponding to ExpCovariance.",
};

const OUTPUTS_R: [BuiltinParamDescriptor; 1] = [OUTPUT_R];
const OUTPUTS_R_SIGMA: [BuiltinParamDescriptor; 2] = [OUTPUT_R, OUTPUT_SIGMA];
const OUTPUTS_EXP_SIGMA_CORR: [BuiltinParamDescriptor; 2] = [OUTPUT_EXP_SIGMA, OUTPUT_EXP_CORR];

const INPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square covariance matrix.",
}];

const CORRCOV_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "R = corrcov(C)",
        inputs: &INPUT_C,
        outputs: &OUTPUTS_R,
    },
    BuiltinSignatureDescriptor {
        label: "[R, sigma] = corrcov(C)",
        inputs: &INPUT_C,
        outputs: &OUTPUTS_R_SIGMA,
    },
];

const INPUT_EXP_COVARIANCE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ExpCovariance",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square covariance matrix.",
}];

const COV2CORR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[ExpSigma, ExpCorrC] = cov2corr(ExpCovariance)",
    inputs: &INPUT_EXP_COVARIANCE,
    outputs: &OUTPUTS_EXP_SIGMA_CORR,
}];

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COVARIANCE_CONVERSION.INTERNAL",
    identifier: None,
    when: "Internal tensor allocation fails.",
    message: "covariance conversion: internal error",
};

macro_rules! conversion_descriptor {
    ($name:literal, $signatures:expr, $invalid_when:literal) => {
        const ERRORS: [BuiltinErrorDescriptor; 2] = [
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INVALID_ARGUMENT"),
                identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
                when: $invalid_when,
                message: "covariance conversion: invalid argument",
            },
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INTERNAL"),
                identifier: Some(concat!("RunMat:", $name, ":Internal")),
                when: ERROR_INTERNAL.when,
                message: ERROR_INTERNAL.message,
            },
        ];

        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::ByRequestedOutputCount,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

pub mod corrcov {
    use super::*;
    conversion_descriptor!("corrcov", CORRCOV_SIGNATURES, "Input is nonnumeric, complex, outside the documented single/double domain or declared RunMat extensions, non-square, non-finite, asymmetric, not positive semidefinite, or too many arguments/outputs are supplied.");
}
pub mod cov2corr {
    use super::*;
    conversion_descriptor!("cov2corr", COV2CORR_SIGNATURES, "Input is nonnumeric, complex, outside the documented double domain or declared RunMat extensions, non-square, non-finite, asymmetric, not positive semidefinite, or too many arguments/outputs are supplied.");
}
pub use corrcov::DESCRIPTOR as CORRCOV_DESCRIPTOR;
pub use cov2corr::DESCRIPTOR as COV2CORR_DESCRIPTOR;
