mod documentation;

use crate::{
    BuiltinDescriptor, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinSignatureDescriptor,
    IntegerLimitKind, NumericLimitRule, ALL_INTEGER_CLASSES,
};

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "value = intmax()",
        inputs: &[],
        outputs: &super::contract::VALUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "value = intmax(typename)",
        inputs: &super::contract::CLASS_INPUT,
        outputs: &super::contract::VALUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "value = intmax(\"like\", prototype)",
        inputs: &super::contract::LIKE_INPUTS,
        outputs: &super::contract::VALUE_OUTPUT,
    },
];
const DESCRIPTOR: BuiltinDescriptor = super::contract::descriptor(&SIGNATURES);
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "prototype",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The prototype selects one of the eight integer classes and may be real or complex.",
}];
const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "value = intmax(\"like\", integer_prototype)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Returns the exact maximum as a scalar with the prototype class, complexity, and applicable residency.",
    }];

pub const INTMAX_CATALOG_ENTRY: crate::BuiltinCatalogEntry = super::contract::entry(
    "intmax",
    documentation::DOCUMENTATION,
    &DESCRIPTOR,
    NumericLimitRule::Integer(IntegerLimitKind::Maximum),
    &INTEGER_CAPABILITIES,
);
