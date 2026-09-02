mod documentation;

use super::support::define_metadata_predicate_entry;
use documentation::ISSPARSE_DOCUMENTATION;

const DENSE_INTEGER_INPUTS: [crate::BuiltinIntegerInputCapability; 1] =
    [crate::BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: crate::BuiltinIntegerInputAvailability::Documented,
        scalar_double: crate::BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Dense arrays of every fixed-width integer class are not sparse.",
    }];
const SPARSE_INTEGER_INPUTS: [crate::BuiltinIntegerInputCapability; 1] =
    [crate::BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: crate::BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: crate::BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed integer CSC storage is a RunMat extension; creation and propagation are compatibility-gated by their owning operations.",
    }];
const INTEGER_CAPABILITIES: [crate::BuiltinIntegerCapabilityDescriptor; 2] = [
    crate::BuiltinIntegerCapabilityDescriptor {
        form: "tf = issparse(dense_integer_A)",
        inputs: &DENSE_INTEGER_INPUTS,
        computation_domain: crate::BuiltinIntegerComputationDomain::Predicate,
        output_class: crate::BuiltinIntegerOutputClassRule::Logical,
        overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
        backend: crate::BuiltinIntegerBackendRule::HostAndGpu,
        overload: crate::BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Returns false from storage metadata; resident integer handles are validated against their exact owner without gathering.",
    },
    crate::BuiltinIntegerCapabilityDescriptor {
        form: "tf = issparse(RunMat_sparse_integer_A)",
        inputs: &SPARSE_INTEGER_INPUTS,
        computation_domain: crate::BuiltinIntegerComputationDomain::Predicate,
        output_class: crate::BuiltinIntegerOutputClassRule::Logical,
        overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
        backend: crate::BuiltinIntegerBackendRule::HostOnly,
        overload: crate::BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Returns true for an existing RunMat sparse integer value without reinterpreting or hiding the value.",
    },
];

define_metadata_predicate_entry!(
    entry: ISSPARSE_CATALOG_ENTRY,
    descriptor: ISSPARSE_DESCRIPTOR,
    internal_error: ISSPARSE_ERROR_INTERNAL,
    output_error: ISSPARSE_ERROR_TOO_MANY_OUTPUTS,
    name: "issparse",
    upper: "ISSPARSE",
    predicate: crate::MetadataPredicate::Sparse,
    documentation: ISSPARSE_DOCUMENTATION,
    input_description: "Value whose dense or sparse storage kind is queried.",
    output_description: "Logical scalar that is true exactly when the input uses sparse storage.",
    integer_capabilities: &INTEGER_CAPABILITIES,
    integer_audit: None
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISSPARSE_CATALOG_ENTRY];
