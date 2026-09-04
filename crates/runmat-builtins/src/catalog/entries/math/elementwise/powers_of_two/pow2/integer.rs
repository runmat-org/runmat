use crate::*;

const UNARY_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "E",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The exponent must cross the binary64 calculation boundary exactly.",
}];
const SIGNIFICAND_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "F",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The significand must cross the binary64 calculation boundary exactly.",
}];
const BINARY_EXPONENT_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "E",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The binary exponent must cross the floating calculation boundary exactly.",
}];

pub const POW2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = pow2(integer_E)",
        inputs: &UNARY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Integer exponents bypass floating provider hooks, convert only after exactness validation, and produce double output.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = pow2(integer_F,E)",
        inputs: &SIGNIFICAND_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Integer significands are independently compatibility-gated and exactly validated.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = pow2(F,integer_E)",
        inputs: &BINARY_EXPONENT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Integer exponents are independently compatibility-gated and never sent to a floating provider hook as integer storage.",
    },
];
