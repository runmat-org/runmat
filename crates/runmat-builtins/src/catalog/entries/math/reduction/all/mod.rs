mod documentation;

use super::support::define_logical_reduction_entry;
use documentation::ALL_DOCUMENTATION;

pub const ALL_ERROR_INVALID_ARGUMENT: crate::BuiltinErrorDescriptor =
    crate::BuiltinErrorDescriptor {
        code: "RM.ALL.INVALID_ARGUMENT",
        identifier: Some("RunMat:all:InvalidArgument"),
        when: "The dimension, all-selector, or NaN-policy grammar is invalid.",
        message: "all: invalid reduction argument specification",
    };
pub const ALL_ERROR_INVALID_INPUT: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
    code: "RM.ALL.INVALID_INPUT",
    identifier: Some("RunMat:all:InvalidInput"),
    when: "A is not numeric, logical, complex, or character data.",
    message: "all: unsupported input type",
};
pub const ALL_ERROR_INTERNAL: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
    code: "RM.ALL.INTERNAL",
    identifier: Some("RunMat:all:Internal"),
    when: "Internal conversion, provider, allocation, or shape validation fails.",
    message: "all: internal reduction failure",
};
pub const ALL_ERROR_TOO_MANY_OUTPUTS: crate::BuiltinErrorDescriptor =
    crate::BuiltinErrorDescriptor {
        code: "RM.ALL.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:all:TooManyOutputs"),
        when: "More than one output is requested.",
        message: "all: too many output arguments",
    };
const ERRORS: &[crate::BuiltinErrorDescriptor] = &[
    ALL_ERROR_INVALID_ARGUMENT,
    ALL_ERROR_INVALID_INPUT,
    ALL_ERROR_INTERNAL,
    ALL_ERROR_TOO_MANY_OUTPUTS,
];

pub const ALL_NANFLAG_EXTENSION: crate::BuiltinExtensionDescriptor =
    crate::BuiltinExtensionDescriptor {
        id: "all-nanflag",
        mode: crate::BuiltinExtensionMode::RunMatOnly,
        description: "all accepts an explicit omitnan or includenan policy",
        error_identifier: Some("RunMat:compatibility:AllNanflagExtension"),
    };
pub const ALL_EXTENSIONS: [crate::BuiltinExtensionDescriptor; 1] = [ALL_NANFLAG_EXTENSION];

const INTEGER_INPUTS: [crate::BuiltinIntegerInputCapability; 2] = [
    crate::BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: crate::BuiltinIntegerInputAvailability::Documented,
        scalar_double: crate::BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every fixed-width integer element is tested directly for zero without floating conversion.",
    },
    crate::BuiltinIntegerInputCapability {
        name: "dim_or_vecdim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: crate::BuiltinIntegerInputAvailability::Documented,
        scalar_double: crate::BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Scalar and vector dimensions accept every fixed-width integer class and integer-valued floating controls.",
    },
];
pub const ALL_INTEGER_CAPABILITIES: [crate::BuiltinIntegerCapabilityDescriptor; 1] =
    [crate::BuiltinIntegerCapabilityDescriptor {
        form: "B = all(integer_A, dim_or_vecdim|\"all\")",
        inputs: &INTEGER_INPUTS,
        computation_domain: crate::BuiltinIntegerComputationDomain::Predicate,
        output_class: crate::BuiltinIntegerOutputClassRule::Logical,
        overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
        backend: crate::BuiltinIntegerBackendRule::HostAndGpu,
        overload: crate::BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer data is reduced exactly and the result is logical with reduced dimensions retained at size one.",
    }];

define_logical_reduction_entry!(
    entry: ALL_CATALOG_ENTRY,
    descriptor: ALL_DESCRIPTOR,
    name: "all",
    kind: crate::LogicalReductionKind::All,
    documentation: ALL_DOCUMENTATION,
    errors: ERRORS,
    extensions: &ALL_EXTENSIONS,
    integer_capabilities: &ALL_INTEGER_CAPABILITIES
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ALL_CATALOG_ENTRY];
