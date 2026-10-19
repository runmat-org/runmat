use super::signatures::*;
use crate::*;

const NORMPDF_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.normpdf.INVALID_ARGUMENT",
        identifier: Some("RunMat:normpdf:InvalidArgument"),
        when: "Inputs are nonnumeric, sizes are incompatible, or too many arguments are supplied.",
        message: "normal distribution: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.normpdf.INTERNAL",
        identifier: Some("RunMat:normpdf:Internal"),
        when: "Internal tensor conversion or allocation fails.",
        message: "normal distribution: internal error",
    },
];
pub const NORMPDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NORMAL_SIGNATURES_X,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NORMPDF_ERRORS,
};

const NORMCDF_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.normcdf.INVALID_ARGUMENT",
        identifier: Some("RunMat:normcdf:InvalidArgument"),
        when: "Inputs are nonnumeric, sizes are incompatible, or too many arguments are supplied.",
        message: "normal distribution: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.normcdf.INTERNAL",
        identifier: Some("RunMat:normcdf:Internal"),
        when: "Internal tensor conversion or allocation fails.",
        message: "normal distribution: internal error",
    },
];
pub const NORMCDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NORMAL_CDF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NORMCDF_ERRORS,
};

const NORMINV_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.norminv.INVALID_ARGUMENT",
        identifier: Some("RunMat:norminv:InvalidArgument"),
        when: "Inputs are nonnumeric, sizes are incompatible, or too many arguments are supplied.",
        message: "normal distribution: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.norminv.INTERNAL",
        identifier: Some("RunMat:norminv:Internal"),
        when: "Internal tensor conversion or allocation fails.",
        message: "normal distribution: internal error",
    },
];
pub const NORMINV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NORMAL_INV_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NORMINV_ERRORS,
};

const TPDF_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.tpdf.INVALID_ARGUMENT",
        identifier: Some("RunMat:tpdf:InvalidArgument"),
        when: "Inputs are nonnumeric, sizes are incompatible, or too many arguments are supplied.",
        message: "normal distribution: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.tpdf.INTERNAL",
        identifier: Some("RunMat:tpdf:Internal"),
        when: "Internal tensor conversion or allocation fails.",
        message: "normal distribution: internal error",
    },
];
pub const TPDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &T_PDF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TPDF_ERRORS,
};

const TCDF_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.tcdf.INVALID_ARGUMENT",
        identifier: Some("RunMat:tcdf:InvalidArgument"),
        when: "Inputs are nonnumeric, sizes are incompatible, or too many arguments are supplied.",
        message: "normal distribution: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.tcdf.INTERNAL",
        identifier: Some("RunMat:tcdf:Internal"),
        when: "Internal tensor conversion or allocation fails.",
        message: "normal distribution: internal error",
    },
];
pub const TCDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &T_CDF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TCDF_ERRORS,
};

const TINV_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.tinv.INVALID_ARGUMENT",
        identifier: Some("RunMat:tinv:InvalidArgument"),
        when: "Inputs are nonnumeric, sizes are incompatible, or too many arguments are supplied.",
        message: "normal distribution: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.tinv.INTERNAL",
        identifier: Some("RunMat:tinv:Internal"),
        when: "Internal tensor conversion or allocation fails.",
        message: "normal distribution: internal error",
    },
];
pub const TINV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &T_INV_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TINV_ERRORS,
};

const BINOCDF_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.binocdf.INVALID_ARGUMENT",
        identifier: Some("RunMat:binocdf:InvalidArgument"),
        when: "Inputs are nonnumeric, sizes are incompatible, or too many arguments are supplied.",
        message: "normal distribution: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.binocdf.INTERNAL",
        identifier: Some("RunMat:binocdf:Internal"),
        when: "Internal tensor conversion or allocation fails.",
        message: "normal distribution: internal error",
    },
];
pub const BINOCDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BINOCDF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BINOCDF_ERRORS,
};

const CHI2CDF_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.chi2cdf.INVALID_ARGUMENT",
        identifier: Some("RunMat:chi2cdf:InvalidArgument"),
        when: "Inputs are nonnumeric, sizes are incompatible, or too many arguments are supplied.",
        message: "normal distribution: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.chi2cdf.INTERNAL",
        identifier: Some("RunMat:chi2cdf:Internal"),
        when: "Internal tensor conversion or allocation fails.",
        message: "normal distribution: internal error",
    },
];
pub const CHI2CDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CHI2CDF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CHI2CDF_ERRORS,
};

const WBLINV_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.wblinv.INVALID_ARGUMENT",
        identifier: Some("RunMat:wblinv:InvalidArgument"),
        when: "Inputs are nonnumeric, sizes are incompatible, or too many arguments are supplied.",
        message: "normal distribution: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.wblinv.INTERNAL",
        identifier: Some("RunMat:wblinv:Internal"),
        when: "Internal tensor conversion or allocation fails.",
        message: "normal distribution: internal error",
    },
];
pub const WBLINV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WBLINV_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WBLINV_ERRORS,
};

const ICDF_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.icdf.INVALID_ARGUMENT",
        identifier: Some("RunMat:icdf:InvalidArgument"),
        when: "Inputs are nonnumeric, sizes are incompatible, or too many arguments are supplied.",
        message: "icdf: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.icdf.INTERNAL",
        identifier: Some("RunMat:icdf:Internal"),
        when: "Internal tensor conversion or allocation fails.",
        message: "icdf: internal error",
    },
];
pub const ICDF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ICDF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ICDF_ERRORS,
};
