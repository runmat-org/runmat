use crate::*;
const CORR_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X_or_Y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented observation domain is single or double; RunMat mode additionally accepts all eight real integer classes.",
    }];

const CORR_INTEGER_WEIGHTS_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Weights",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented observation-weight domain is single or double; RunMat mode additionally accepts exact nonnegative typed-integer column vectors.",
    }];

pub const CORR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[rho, pval] = corr(integer_X_or_Y, Type=type, Rows=rows, Weights=weights)",
        inputs: &CORR_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat's integer-data extension preserves exact same-class ordering for Spearman/Kendall and exact integer differences for Pearson centering before producing double rho/p-values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[rho, pval] = corr(X, Y, Weights=integer_weights)",
        inputs: &CORR_INTEGER_WEIGHTS_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Typed integer weights are validated exactly, then enter the documented nonnegative weighted-correlation domain; weighted p-values are NaN.",
    },
];
