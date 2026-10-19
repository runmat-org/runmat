use crate::*;

const STATS_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Xnew for implemented statistical models",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The implemented linear-regression, classification-tree, and linear-classifier references document single, double, or table predictors. RunMat admits typed integer matrices only at a checked model-specific floating boundary.",
}];
const CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Alpha, Subtrees, or other numeric statistical controls",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed integer numeric controls are independently gated; unsupported general options remain ordinary surface gaps rather than silently admitted integer forms.",
}];
const DEEP_LEARNING_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X for supported deep-learning network objects",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The public deep-learning predict surface explicitly includes all eight integer input classes; the selected network determines output class and conversion behavior.",
}];

pub(super) const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "ypred = predict(statistical_model,integer_Xnew,___)",
        inputs: &STATS_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Statistical integer predictors are gated before gather and model dispatch and must be exactly representable at the binary64 prediction boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ypred = predict(statistical_model,Xnew,integer_control,___)",
        inputs: &CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Numeric statistical controls are separately admitted and checked before the selected model parser runs.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = predict(deep_learning_network,integer_X,___)",
        inputs: &DEEP_LEARNING_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Deep-learning integer admission is documented and is intentionally not covered by the statistical-model compatibility gates.",
    },
];
