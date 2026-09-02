mod documentation;

use super::support::define_shape_scalar_query_entry;
use crate::{
    BuiltinErrorDescriptor, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use documentation::HEIGHT_DOCUMENTATION;

pub const HEIGHT_ERROR_RESULT_NOT_EXACT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HEIGHT.RESULT_NOT_EXACT_DOUBLE",
    identifier: Some("RunMat:height:ResultNotExactDouble"),
    when: "The row count cannot be represented exactly by the documented double output.",
    message: "height: result exceeds exact double range",
};
pub const HEIGHT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HEIGHT.INTERNAL",
    identifier: Some("RunMat:height:InternalError"),
    when: "Value, tabular, resident, or distributed shape metadata is invalid.",
    message: "height: invalid shape metadata",
};
pub const HEIGHT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HEIGHT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:height:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "height: too many output arguments",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[
    HEIGHT_ERROR_RESULT_NOT_EXACT,
    HEIGHT_ERROR_INTERNAL,
    HEIGHT_ERROR_TOO_MANY_OUTPUTS,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Only the first dimension extent is inspected; integer elements remain untouched.",
}];
pub const HEIGHT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "n = height(integer_A)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Every integer class shares the row-count contract. Resident input is answered from handle shape without payload transfer, and the exact result is returned as a host double scalar.",
    }];

define_shape_scalar_query_entry!(
    entry: HEIGHT_CATALOG_ENTRY,
    descriptor: HEIGHT_DESCRIPTOR,
    name: "height",
    query: crate::ShapeScalarQuery::Height,
    documentation: HEIGHT_DOCUMENTATION,
    errors: ERRORS,
    integer_capabilities: &HEIGHT_INTEGER_CAPABILITIES,
    output_description: "Exact extent of the first MATLAB-visible dimension."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&HEIGHT_CATALOG_ENTRY];
