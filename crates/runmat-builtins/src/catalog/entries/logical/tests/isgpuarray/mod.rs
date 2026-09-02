mod documentation;

use super::support::define_metadata_predicate_entry;
use documentation::ISGPUARRAY_DOCUMENTATION;

const INTEGER_INPUTS: [crate::BuiltinIntegerInputCapability; 1] =
    [crate::BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: crate::BuiltinIntegerInputAvailability::Documented,
        scalar_double: crate::BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "An explicitly constructed gpuArray may contain any fixed-width integer class; the predicate inspects residency intent without downloading its payload.",
    }];
const INTEGER_CAPABILITIES: [crate::BuiltinIntegerCapabilityDescriptor; 1] =
    [crate::BuiltinIntegerCapabilityDescriptor {
        form: "tf = isgpuarray(integer_gpuArray)",
        inputs: &INTEGER_INPUTS,
        computation_domain: crate::BuiltinIntegerComputationDomain::Predicate,
        output_class: crate::BuiltinIntegerOutputClassRule::Logical,
        overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
        backend: crate::BuiltinIntegerBackendRule::HostAndGpu,
        overload: crate::BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Returns true for explicit integer gpuArray values and false for host integers or automatically resident internal values, with no gather or conversion.",
    }];

define_metadata_predicate_entry!(
    entry: ISGPUARRAY_CATALOG_ENTRY,
    descriptor: ISGPUARRAY_DESCRIPTOR,
    internal_error: ISGPUARRAY_ERROR_INTERNAL,
    output_error: ISGPUARRAY_ERROR_TOO_MANY_OUTPUTS,
    name: "isgpuarray",
    upper: "ISGPUARRAY",
    predicate: crate::MetadataPredicate::GpuArray,
    documentation: ISGPUARRAY_DOCUMENTATION,
    input_description: "Value whose explicit gpuArray identity is queried.",
    output_description: "Logical scalar that is true exactly for an explicit gpuArray value.",
    integer_capabilities: &INTEGER_CAPABILITIES,
    integer_audit: None
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISGPUARRAY_CATALOG_ENTRY];
