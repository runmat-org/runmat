mod documentation;

use super::support::define_logical_reduction_entry;
use documentation::ANY_DOCUMENTATION;

pub const ANY_ERROR_INVALID_ARGUMENT: crate::BuiltinErrorDescriptor =
    crate::BuiltinErrorDescriptor {
        code: "RM.ANY.INVALID_ARGUMENT",
        identifier: Some("RunMat:any:InvalidArgument"),
        when: "The dimension, all-selector, or NaN-policy grammar is invalid.",
        message: "any: invalid reduction argument specification",
    };
pub const ANY_ERROR_INVALID_INPUT: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
    code: "RM.ANY.INVALID_INPUT",
    identifier: Some("RunMat:any:InvalidInput"),
    when: "A is not numeric, logical, complex, or character data.",
    message: "any: unsupported input type",
};
pub const ANY_ERROR_INTERNAL: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
    code: "RM.ANY.INTERNAL",
    identifier: Some("RunMat:any:Internal"),
    when: "Internal conversion, provider, allocation, or shape validation fails.",
    message: "any: internal reduction failure",
};
pub const ANY_ERROR_TOO_MANY_OUTPUTS: crate::BuiltinErrorDescriptor =
    crate::BuiltinErrorDescriptor {
        code: "RM.ANY.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:any:TooManyOutputs"),
        when: "More than one output is requested.",
        message: "any: too many output arguments",
    };
const ERRORS: &[crate::BuiltinErrorDescriptor] = &[
    ANY_ERROR_INVALID_ARGUMENT,
    ANY_ERROR_INVALID_INPUT,
    ANY_ERROR_INTERNAL,
    ANY_ERROR_TOO_MANY_OUTPUTS,
];

pub const ANY_NANFLAG_EXTENSION: crate::BuiltinExtensionDescriptor =
    crate::BuiltinExtensionDescriptor {
        id: "any-nanflag",
        mode: crate::BuiltinExtensionMode::RunMatOnly,
        description: "any accepts an explicit omitnan or includenan policy",
        error_identifier: Some("RunMat:compatibility:AnyNanflagExtension"),
    };
pub const ANY_EXTENSIONS: [crate::BuiltinExtensionDescriptor; 1] = [ANY_NANFLAG_EXTENSION];

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
pub const ANY_INTEGER_CAPABILITIES: [crate::BuiltinIntegerCapabilityDescriptor; 1] =
    [crate::BuiltinIntegerCapabilityDescriptor {
        form: "B = any(integer_A, dim_or_vecdim|\"all\")",
        inputs: &INTEGER_INPUTS,
        computation_domain: crate::BuiltinIntegerComputationDomain::Predicate,
        output_class: crate::BuiltinIntegerOutputClassRule::Logical,
        overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
        backend: crate::BuiltinIntegerBackendRule::HostAndGpu,
        overload: crate::BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer data is reduced exactly and the result is logical with reduced dimensions retained at size one.",
    }];

define_logical_reduction_entry!(
    entry: ANY_CATALOG_ENTRY,
    descriptor: ANY_DESCRIPTOR,
    name: "any",
    kind: crate::LogicalReductionKind::Any,
    documentation: ANY_DOCUMENTATION,
    errors: ERRORS,
    extensions: &ANY_EXTENSIONS,
    integer_capabilities: &ANY_INTEGER_CAPABILITIES
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ANY_CATALOG_ENTRY];
