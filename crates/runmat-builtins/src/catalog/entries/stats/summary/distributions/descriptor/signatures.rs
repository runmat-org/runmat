use super::parameters::*;
use crate::*;

pub(super) const NORMAL_SIGNATURES_X: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "y = normpdf(x)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = normpdf(x, mu)",
        inputs: &INPUTS_X_MU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = normpdf(x, mu, sigma)",
        inputs: &INPUTS_X_MU_SIGMA,
        outputs: &OUTPUT_Y,
    },
];

pub(super) const NORMAL_CDF_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "p = normcdf(x)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "p = normcdf(x, mu)",
        inputs: &INPUTS_X_MU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "p = normcdf(x, mu, sigma)",
        inputs: &INPUTS_X_MU_SIGMA,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "p = normcdf(x, \"upper\")",
        inputs: &INPUTS_X_UPPER,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "p = normcdf(x, mu, \"upper\")",
        inputs: &INPUTS_X_MU_UPPER,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "p = normcdf(x, mu, sigma, \"upper\")",
        inputs: &INPUTS_X_MU_SIGMA_UPPER,
        outputs: &OUTPUT_Y,
    },
];

pub(super) const NORMAL_INV_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "x = norminv(p)",
        inputs: &INPUTS_P,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = norminv(p, mu)",
        inputs: &INPUTS_P_MU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = norminv(p, mu, sigma)",
        inputs: &INPUTS_P_MU_SIGMA,
        outputs: &OUTPUT_Y,
    },
];

pub(super) const T_PDF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "y = tpdf(x, nu)",
    inputs: &INPUTS_X_NU,
    outputs: &OUTPUT_Y,
}];

pub(super) const T_CDF_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "p = tcdf(x, nu)",
        inputs: &INPUTS_X_NU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "p = tcdf(x, nu, \"upper\")",
        inputs: &INPUTS_X_NU_UPPER,
        outputs: &OUTPUT_Y,
    },
];

pub(super) const T_INV_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "x = tinv(p, nu)",
    inputs: &INPUTS_P_NU,
    outputs: &OUTPUT_Y,
}];

pub(super) const BINOCDF_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "y = binocdf(x, n, p)",
        inputs: &INPUTS_X_N_P,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = binocdf(x, n, p, \"upper\")",
        inputs: &INPUTS_X_N_P_UPPER,
        outputs: &OUTPUT_Y,
    },
];

pub(super) const CHI2CDF_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "p = chi2cdf(x, nu)",
        inputs: &INPUTS_X_NU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "p = chi2cdf(x, nu, \"upper\")",
        inputs: &INPUTS_X_NU_UPPER,
        outputs: &OUTPUT_Y,
    },
];

pub(super) const WBLINV_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "x = wblinv(p)",
        inputs: &INPUTS_P,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = wblinv(p, a, b)",
        inputs: &INPUTS_P_A_B,
        outputs: &OUTPUT_Y,
    },
];

pub(super) const INPUT_PD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "pd",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "ProbabilityDistribution object returned by fitdist.",
};
pub(super) const INPUTS_PD_P: [BuiltinParamDescriptor; 2] = [INPUT_PD, INPUT_P];

pub(super) const ICDF_SIGNATURES: [BuiltinSignatureDescriptor; 19] = [
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Normal\", p)",
        inputs: &INPUTS_DIST_P,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Normal\", p, mu)",
        inputs: &INPUTS_DIST_P_MU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Normal\", p, mu, sigma)",
        inputs: &INPUTS_DIST_P_MU_SIGMA,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"t\", p, nu)",
        inputs: &INPUTS_DIST_P_PARAM,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Weibull\", p)",
        inputs: &INPUTS_DIST_P,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Weibull\", p, a, b)",
        inputs: &INPUTS_DIST_P_A_B,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Chi-square\", p, nu)",
        inputs: &INPUTS_DIST_P_PARAM,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Binomial\", p, n, prob)",
        inputs: &INPUTS_DIST_P_N_PROB,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Gamma\", p, shape, scale)",
        inputs: &INPUTS_DIST_P_SHAPE_SCALE,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Exponential\", p, mu)",
        inputs: &INPUTS_DIST_P_MU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Poisson\", p, lambda)",
        inputs: &INPUTS_DIST_P_LAMBDA,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Uniform\", p)",
        inputs: &INPUTS_DIST_P,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Uniform\", p, a, b)",
        inputs: &INPUTS_DIST_P_BOUNDS,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Lognormal\", p)",
        inputs: &INPUTS_DIST_P,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Lognormal\", p, mu)",
        inputs: &INPUTS_DIST_P_MU,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Lognormal\", p, mu, sigma)",
        inputs: &INPUTS_DIST_P_MU_SIGMA,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"Beta\", p, a, b)",
        inputs: &INPUTS_DIST_P_ALPHA_BETA,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(\"F\", p, v1, v2)",
        inputs: &INPUTS_DIST_P_DF,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "x = icdf(pd, p)",
        inputs: &INPUTS_PD_P,
        outputs: &OUTPUT_Y,
    },
];
