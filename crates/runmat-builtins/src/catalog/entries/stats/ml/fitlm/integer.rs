use crate::*;

const PREDICTORS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X or active table predictors",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The compatibility target documents single or double predictors. RunMat mode admits all eight integer classes at an exact binary64 fitting boundary.",
}];
const RESPONSES: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "y or active table response",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The compatibility target documents single or double matrix responses and numeric/logical table responses, but not typed integers. RunMat mode requires exact binary64 representation.",
}];
const WEIGHTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Weights",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The compatibility target documents single or double weights. RunMat mode admits typed integers only at an exact binary64 weighting boundary.",
}];
const SELECTORS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Exclude and numeric selectors",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed-integer structural selectors are parsed exactly as one-based indices or logical-style masks after independent extension admission.",
}];
const INTERCEPT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Intercept",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "A typed-integer Intercept scalar is a RunMat-only exact zero/nonzero control and does not cross through binary64.",
}];

pub const FITLM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "mdl = fitlm(integer_X,y,...) or fitlm(table_with_integer_predictors,...)",
        inputs: &PREDICTORS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Predictors are gated before provider access; admitted resident data gathers exactly and the returned model is currently host-resident with binary64 properties.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "mdl = fitlm(X,integer_y,...) or fitlm(table_with_integer_response,...)",
        inputs: &RESPONSES,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Responses are independently gated and checked before binary64 least-squares computation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "mdl = fitlm(...,Weights=integer_weights)",
        inputs: &WEIGHTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer weights remain authoritative through admission and must be nonnegative, finite, shape-matched, and exactly representable as double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "mdl = fitlm(...,Exclude=integer_indices,...)",
        inputs: &SELECTORS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "One-based integer selectors are decoded exactly in every supported integer class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "mdl = fitlm(...,Intercept=integer_scalar)",
        inputs: &INTERCEPT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Zero disables and every nonzero exact integer enables the intercept.",
    },
];
