use crate::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};

const CONTENTS: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "integer values stored in C",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes:
        "Integer cell contents reach the callback in their native class after any required gather.",
}];

const CALLBACK_RESULT: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "func or ErrorHandler scalar result",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
    notes: "Uniform output retains one common integer class without conversion through double.",
}];

const UNIFORM_CONTROL: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "UniformOutput",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The control accepts logical true or false and legacy double one or zero; typed integers reject.",
}];

pub const CELLFUN_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = cellfun(func, C...) with integer cell contents",
        inputs: CONTENTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "cellfun performs structural extraction; the callback determines arithmetic and result class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "integer_Y = cellfun(func_returning_integer, C...) with UniformOutput=true",
        inputs: CALLBACK_RESULT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Same-class scalar integer results are collected in native integer storage.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = cellfun(func, C..., \"UniformOutput\", typed_integer)",
        inputs: UNIFORM_CONTROL,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "All fixed-width integer control classes reject rather than being coerced to logical.",
    },
];
