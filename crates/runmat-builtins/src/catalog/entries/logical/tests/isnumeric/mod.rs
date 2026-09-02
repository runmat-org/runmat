mod documentation;

use super::support::define_metadata_predicate_entry;
use documentation::ISNUMERIC_DOCUMENTATION;

const INTEGER_INPUTS: [crate::BuiltinIntegerInputCapability; 1] =
    [crate::BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: crate::BuiltinIntegerInputAvailability::Documented,
        scalar_double: crate::BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every signed and unsigned fixed-width integer class is numeric.",
    }];
const INTEGER_CAPABILITIES: [crate::BuiltinIntegerCapabilityDescriptor; 1] =
    [crate::BuiltinIntegerCapabilityDescriptor {
        form: "tf = isnumeric(integer_A)",
        inputs: &INTEGER_INPUTS,
        computation_domain: crate::BuiltinIntegerComputationDomain::Structural,
        output_class: crate::BuiltinIntegerOutputClassRule::Logical,
        overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
        backend: crate::BuiltinIntegerBackendRule::HostAndGpu,
        overload: crate::BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Returns one logical scalar from authoritative host storage or coherent resident class metadata without reading payload values.",
    }];

define_metadata_predicate_entry!(
    entry: ISNUMERIC_CATALOG_ENTRY,
    descriptor: ISNUMERIC_DESCRIPTOR,
    internal_error: ISNUMERIC_ERROR_INTERNAL,
    output_error: ISNUMERIC_ERROR_TOO_MANY_OUTPUTS,
    name: "isnumeric",
    upper: "ISNUMERIC",
    predicate: crate::MetadataPredicate::Numeric,
    documentation: ISNUMERIC_DOCUMENTATION,
    input_description: "Value whose numeric storage class is queried.",
    output_description: "Logical scalar that is true exactly when the input uses a numeric class.",
    integer_capabilities: &INTEGER_CAPABILITIES,
    integer_audit: None
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISNUMERIC_CATALOG_ENTRY];
