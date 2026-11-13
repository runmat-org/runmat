use crate::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    ALL_INTEGER_CLASSES,
};

const INTEGER_ARRAYS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A1",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Element extraction preserves the stored integer class.",
    },
    BuiltinIntegerInputCapability {
        name: "An",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The callback decides whether each class combination is valid.",
    },
];
const INTEGER_RESULT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "callback scalar result",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
    notes: "Uniform same-class results retain exact native storage.",
}];
const INTEGER_CONTROL: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "UniformOutput",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed integer controls are rejected rather than coerced.",
}];

pub const ARRAYFUN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "B = arrayfun(func, integer_A1, integer_An...)", inputs: &INTEGER_ARRAYS, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::FunctionSpecific, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::Multiple, notes: "arrayfun passes exact scalar values; the callback owns arithmetic and result semantics." },
    BuiltinIntegerCapabilityDescriptor { form: "integer_B = arrayfun(func_returning_integer, A1, An...)", inputs: &INTEGER_RESULT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::FunctionSpecific, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "Uniform integer results preserve their class and may be re-uploaded after GPU fallback." },
    BuiltinIntegerCapabilityDescriptor { form: "B = arrayfun(func, A1, An..., \"UniformOutput\", typed_integer)", inputs: &INTEGER_CONTROL, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "All typed-integer control classes reject." },
];
