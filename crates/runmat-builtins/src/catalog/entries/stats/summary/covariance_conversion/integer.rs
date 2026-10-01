use crate::*;
pub const CORRCOV_INTEGER_C_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "C",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The documented covariance-matrix domain is floating; RunMat mode additionally accepts all eight real integer classes.",
}];

pub const COV2CORR_INTEGER_C_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ExpCovariance",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented Finance Toolbox covariance-matrix domain is double; RunMat mode additionally accepts all eight real integer classes.",
    }];

pub const CORRCOV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[R, sigma] = corrcov(integer_C)",
        inputs: &CORRCOV_INTEGER_C_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat validates exact integer symmetry before converting covariance values into double square-root and normalization arithmetic.",
    }];

pub const COV2CORR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[ExpSigma, ExpCorrC] = cov2corr(integer_ExpCovariance)",
        inputs: &COV2CORR_INTEGER_C_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat validates exact integer symmetry before returning double standard deviations and correlation coefficients in the documented Finance Toolbox order and shapes.",
    }];
