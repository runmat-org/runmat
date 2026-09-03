mod documentation;

use super::support::define_metadata_predicate_entry;
use documentation::ISREAL_DOCUMENTATION;

const INTEGER_INPUTS: [crate::BuiltinIntegerInputCapability; 1] =
    [crate::BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: crate::BuiltinIntegerInputAvailability::Documented,
        scalar_double: crate::BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every fixed-width integer class may use real or complex storage; `isreal` inspects that storage kind without converting values.",
    }];
const INTEGER_CAPABILITIES: [crate::BuiltinIntegerCapabilityDescriptor; 1] =
    [crate::BuiltinIntegerCapabilityDescriptor {
        form: "tf = isreal(integer_A)",
        inputs: &INTEGER_INPUTS,
        computation_domain: crate::BuiltinIntegerComputationDomain::Predicate,
        output_class: crate::BuiltinIntegerOutputClassRule::Logical,
        overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
        backend: crate::BuiltinIntegerBackendRule::HostAndGpu,
        overload: crate::BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Returns one logical scalar from storage complexity: real integer storage is true and complex integer storage is false, even when every imaginary component is zero.",
    }];

define_metadata_predicate_entry!(
    entry: ISREAL_CATALOG_ENTRY,
    descriptor: ISREAL_DESCRIPTOR,
    internal_error: ISREAL_ERROR_INTERNAL,
    output_error: ISREAL_ERROR_TOO_MANY_OUTPUTS,
    name: "isreal",
    upper: "ISREAL",
    predicate: crate::MetadataPredicate::Real,
    documentation: ISREAL_DOCUMENTATION,
    input_description: "Value whose real or complex storage kind is queried.",
    output_description: "Logical scalar that is true when the input does not use complex storage.",
    distributed: crate::BuiltinDistributedPolicy::InspectHandles,
    integer_capabilities: &INTEGER_CAPABILITIES,
    integer_audit: None
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISREAL_CATALOG_ENTRY];
