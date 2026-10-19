//! Integer admission facts for typed physical-signal constructors.

use crate::fea_contract::policy_types::integer::{floating, physical};
use crate::{
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerInputCapability,
};

pub const SCALAR_CONSTANT_INPUTS: [BuiltinIntegerInputCapability; 1] = [floating("value")];
pub const SCALAR_CONSTANT_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed constant scalar signal",
    &SCALAR_CONSTANT_INPUTS,
)];

pub const VECTOR_CONSTANT_INPUTS: [BuiltinIntegerInputCapability; 1] = [floating("components")];
pub const VECTOR_CONSTANT_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed constant vector signal",
    &VECTOR_CONSTANT_INPUTS,
)];

pub const TIME_SAMPLE_INPUTS: [BuiltinIntegerInputCapability; 2] =
    [floating("times"), floating("values")];
pub const TIME_SAMPLE_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [physical(
    "typed time-sampled physical signal",
    &TIME_SAMPLE_INPUTS,
)];

/// `fromFunction` has no numeric positional input; numeric generator arguments
/// travel inside the `Arguments` cell and keep their exact class.
pub const FUNCTION_SIGNAL_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Numeric generator arguments are immutable values inside an explicit Arguments cell array and retain their exact integer class; domain and sampling are typed objects.",
};
