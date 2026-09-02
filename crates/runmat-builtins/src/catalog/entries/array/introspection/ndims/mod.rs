mod documentation;

use super::support::define_shape_scalar_query_entry;
use crate::{
    BuiltinErrorDescriptor, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use documentation::NDIMS_DOCUMENTATION;

pub const NDIMS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NDIMS.INTERNAL",
    identifier: Some("RunMat:ndims:InternalError"),
    when: "Value or distributed shape metadata is invalid.",
    message: "ndims: invalid shape metadata",
};
pub const NDIMS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NDIMS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:ndims:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "ndims: too many output arguments",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[NDIMS_ERROR_INTERNAL, NDIMS_ERROR_TOO_MANY_OUTPUTS];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every integer class is documented; only normalized shape metadata participates.",
}];
pub const NDIMS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "n = ndims(integer_A)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The result is a host double scalar of at least two, trailing singleton dimensions are ignored, and resident integer payloads are never downloaded.",
    }];

define_shape_scalar_query_entry!(
    entry: NDIMS_CATALOG_ENTRY,
    descriptor: NDIMS_DESCRIPTOR,
    name: "ndims",
    query: crate::ShapeScalarQuery::Rank,
    documentation: NDIMS_DOCUMENTATION,
    errors: ERRORS,
    integer_capabilities: &NDIMS_INTEGER_CAPABILITIES,
    output_description: "MATLAB-visible rank of the input, with a minimum of two."
);

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&NDIMS_CATALOG_ENTRY];
