use crate::*;

pub const RIDGE_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ridge-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ridge accepts typed-integer response and predictor data as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RidgeIntegerDataExtension"),
};
pub const RIDGE_INTEGER_K_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ridge-integer-k",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ridge accepts typed-integer regularization parameters as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RidgeIntegerKExtension"),
};
pub const RIDGE_INTEGER_SCALED_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ridge-integer-scaled",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ridge accepts a typed-integer scaled flag as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RidgeIntegerScaledExtension"),
};
pub const RIDGE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    RIDGE_INTEGER_DATA_EXTENSION,
    RIDGE_INTEGER_K_EXTENSION,
    RIDGE_INTEGER_SCALED_EXTENSION,
];
const RIDGE_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double response data; typed integers require exact binary64 conversion.",
    },
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double predictors; typed integers require the same checked conversion.",
    },
];
const RIDGE_INTEGER_K_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "k",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed regularization parameters are a RunMat extension and enter the floating solver only after exactness and range checks.",
    }];
const RIDGE_INTEGER_SCALED_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "scaled",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents scaled as numeric 0 or 1 but not a typed integer class; RunMat gates that convenience form independently.",
    }];
pub const RIDGE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "ridge(integer_y, integer_X, k [, scaled])",
        inputs: &RIDGE_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The authoritative integer samples are checked before transparent gather and intentional binary64 standardization/SVD.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ridge(y, X, integer_k [, scaled])",
        inputs: &RIDGE_INTEGER_K_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Regularization parameters are checked exactly before entering the binary64 solver.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ridge(y, X, k, integer_scaled)",
        inputs: &RIDGE_INTEGER_SCALED_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Only exact typed values zero and one are accepted after the compatibility gate.",
    },
];
