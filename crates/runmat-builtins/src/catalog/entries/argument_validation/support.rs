//! Shared declaration building blocks; forms and inference remain identity-owned.
use crate::*;
pub(super) const VALIDATION_ERROR_TEXT_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGUMENT_VALIDATION.TEXT_INPUT",
    identifier: Some("RunMat:argumentValidation:InvalidArgument"),
    when: "The shared validator text-control parser receives a non-text payload.",
    message: "argumentValidation: expected text",
};
pub(super) const VALIDATOR_INTEGER_VALUE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
 name: "A", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented,
 scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
 notes: "All eight integer classes are validated directly from authoritative class, shape, or exact element storage.",
}];
pub(super) const VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value to validate or inspect.",
};
pub(super) const CLASS_NAMES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "classes",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text class names or supported class categories.",
};
pub(super) const LOGICAL_OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar logical result.",
};
pub(super) const CELL_OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Alternating field-name and unchanged field-value cell row.",
};
pub const VALIDATION_ERROR_PROVIDER_OWNERSHIP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGUMENT_VALIDATION.PROVIDER_OWNERSHIP_MISMATCH",
    identifier: Some("RunMat:validators:ProviderOwnershipMismatch"),
    when: "A resident value has no exact owning provider.",
    message: "argument validation: no acceleration provider owns the input",
};
pub const VALIDATION_ERROR_PROVIDER_PAYLOAD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGUMENT_VALIDATION.PROVIDER_PAYLOAD_MISMATCH",
    identifier: Some("RunMat:validators:ProviderPayloadMismatch"),
    when: "A resident value carries contradictory physical class metadata.",
    message: "argument validation: resident input has contradictory physical class metadata",
};
