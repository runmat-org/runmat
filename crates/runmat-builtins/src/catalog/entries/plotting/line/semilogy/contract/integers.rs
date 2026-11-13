use crate::*;

pub(super) const SEMILOGY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X/Y",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Coordinate data accept all eight integer classes and retain exact native source properties before logarithmic Y-axis display scaling.",
    }];

pub const SEMILOGY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "h = semilogy([X,] integer_Y, ...)",
        inputs: &SEMILOGY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The plot source and persisted graphics properties remain native; logarithmic Y-axis transformation and renderer geometry are explicit floating boundaries.",
    }];
