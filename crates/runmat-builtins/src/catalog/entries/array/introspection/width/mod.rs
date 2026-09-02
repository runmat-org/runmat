mod documentation;

use super::support::define_shape_scalar_query_entry;
use crate::{
    BuiltinErrorDescriptor, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use documentation::WIDTH_DOCUMENTATION;

pub const WIDTH_ERROR_RESULT_NOT_EXACT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WIDTH.RESULT_NOT_EXACT_DOUBLE",
    identifier: Some("RunMat:width:ResultNotExactDouble"),
    when: "The variable or column count cannot be represented exactly by the documented double output.",
    message: "width: result exceeds exact double range",
};
pub const WIDTH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WIDTH.INTERNAL",
    identifier: Some("RunMat:width:InternalError"),
    when: "Value, tabular, resident, or distributed shape metadata is invalid.",
    message: "width: invalid shape metadata",
};
pub const WIDTH_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WIDTH.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:width:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "width: too many output arguments",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[
    WIDTH_ERROR_RESULT_NOT_EXACT,
    WIDTH_ERROR_INTERNAL,
    WIDTH_ERROR_TOO_MANY_OUTPUTS,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Only the second dimension extent is inspected; integer elements remain untouched.",
}];
pub const WIDTH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "n = width(integer_A)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Every integer class shares the column-count contract. Resident input is answered from handle shape without payload transfer, and the exact result is returned as a host double scalar.",
    }];

define_shape_scalar_query_entry!(
    entry: WIDTH_CATALOG_ENTRY,
    descriptor: WIDTH_DESCRIPTOR,
    name: "width",
    query: crate::ShapeScalarQuery::Width,
    documentation: WIDTH_DOCUMENTATION,
    errors: ERRORS,
    integer_capabilities: &WIDTH_INTEGER_CAPABILITIES,
    output_description: "Exact table-variable count or second MATLAB-visible dimension extent."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&WIDTH_CATALOG_ENTRY];
