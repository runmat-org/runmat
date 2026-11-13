mod documentation;

use super::support::define_shape_scalar_query_entry;
use crate::{
    BuiltinErrorDescriptor, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use documentation::LENGTH_DOCUMENTATION;

pub const LENGTH_ERROR_UNSUPPORTED_TABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LENGTH.UNSUPPORTED_TABLE",
    identifier: Some("RunMat:length:UnsupportedTable"),
    when: "The input is a table or timetable.",
    message: "length: tables and timetables are not supported; use height, width, or size",
};
pub const LENGTH_ERROR_RESULT_NOT_EXACT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LENGTH.RESULT_NOT_EXACT_DOUBLE",
    identifier: Some("RunMat:length:ResultNotExactDouble"),
    when: "The largest dimension cannot be represented exactly by the documented double output.",
    message: "length: result exceeds exact double range",
};
pub const LENGTH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LENGTH.INTERNAL",
    identifier: Some("RunMat:length:InternalError"),
    when: "Value or distributed shape metadata is invalid.",
    message: "length: invalid shape metadata",
};
pub const LENGTH_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LENGTH.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:length:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "length: too many output arguments",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[
    LENGTH_ERROR_UNSUPPORTED_TABLE,
    LENGTH_ERROR_RESULT_NOT_EXACT,
    LENGTH_ERROR_INTERNAL,
    LENGTH_ERROR_TOO_MANY_OUTPUTS,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer values are not inspected; only array shape participates.",
}];
pub const LENGTH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "n = length(integer_A)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "All integer classes use shape metadata only. Resident input requires no provider access, and the exact result is a host double scalar.",
    }];

define_shape_scalar_query_entry!(
    entry: LENGTH_CATALOG_ENTRY,
    descriptor: LENGTH_DESCRIPTOR,
    name: "length",
    query: crate::ShapeScalarQuery::Length,
    documentation: LENGTH_DOCUMENTATION,
    errors: ERRORS,
    integer_capabilities: &LENGTH_INTEGER_CAPABILITIES,
    output_description: "Exact largest visible dimension extent of the input."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&LENGTH_CATALOG_ENTRY];
