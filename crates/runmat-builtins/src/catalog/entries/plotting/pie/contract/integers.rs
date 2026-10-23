use crate::*;

pub(super) const PIE_INTEGER_VALUES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target limits numeric pie values to double. RunMat admits typed integers only when exact binary64 chart geometry is possible.",
    }];

pub(super) const PIE_INTEGER_EXPLODE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "explode",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public numeric explode mask uses exact zero/nonzero semantics; every native integer class can be evaluated without value conversion.",
    }];

pub const PIE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = pie(integer_X,___)",
        inputs: &PIE_INTEGER_VALUES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer values are gated before gather and normalized only after exact conversion to renderer geometry.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = pie(X,integer_explode,___)",
        inputs: &PIE_INTEGER_EXPLODE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed explode masks gather exactly and use zero/nonzero classification; their magnitude never enters renderer arithmetic.",
    },
];
