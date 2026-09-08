use crate::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};

const VALUES: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability { name: "integer field values and callback results", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Callbacks receive native integer field values, and uniform output retains one common result class exactly." }];

pub const STRUCTFUN_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor { form: "A = structfun(func, S, options...) with integer fields or results", inputs: VALUES, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "structfun performs field traversal; the callback determines arithmetic and result class. Uniform collection rejects mixed integer classes." }];
