use crate::*;

const DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "RunMat accepts all eight real integer classes as data; sorting uses exact native integer comparisons before floating interpolation.",
}];
const REQUEST: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "p",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed integer probabilities are independently gated; values must be in [0,1].",
}];
pub const CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = quantile(integer_X, p, ...)", inputs: &DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Ordering and ties retain exact same-class integer values, including uint64 above flintmax; the output is double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = quantile(X, integer_p, ...)", inputs: &REQUEST,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The requested probabilities are checked before entering the double computation domain.",
    },
];
