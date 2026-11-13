use crate::*;

const PREDICTOR: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X", classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer predictors must be exactly representable in binary64 before host fitting; values that would round are rejected.",
}];
const LABEL: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Y or ClassNames", classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Grouping labels retain their exact integer class and identity through model storage and prediction, including above flintmax.",
}];
const NUMERIC_CONTROL: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Weights or numeric option", classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed-integer statistical inputs are independently gated and must cross the binary64 boundary exactly.",
}];
pub const FITCLINEAR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Mdl = fitclinear(integer_X, Y, ___)", inputs: &PREDICTOR,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Matrix and table predictor roles share the same exact conversion boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Mdl = fitclinear(X, integer_Y, ClassNames=integer_names)", inputs: &LABEL,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Class grouping never passes through floating point.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Mdl = fitclinear(X, Y, Weights=integer_W, numericOption=integer_value)",
        inputs: &NUMERIC_CONTROL,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer weights and numeric controls are RunMat-only extensions and reject rather than round.",
    },
];
