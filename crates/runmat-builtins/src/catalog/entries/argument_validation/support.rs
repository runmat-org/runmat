//! Shared declaration building blocks; forms and inference remain identity-owned.
use crate::*;
pub(super) const VALIDATION_ERROR_DOWNLOAD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGUMENT_VALIDATION.DOWNLOAD_FAILED",
    identifier: Some("RunMat:gpu:DownloadFailed"),
    when: "The exact owning provider cannot download admitted numeric content.",
    message: "gpu download failed",
};
pub(super) const VALIDATION_ERROR_DOWNLOAD_PAYLOAD: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.ARGUMENT_VALIDATION.DOWNLOAD_PAYLOAD",
        identifier: Some("RunMat:gpu:ProviderPayloadMismatch"),
        when: "Downloaded class, shape, or storage contradicts the borrowed source metadata.",
        message: "gpu download payload mismatch",
    };
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

pub(super) const VALIDATOR_INTEGER_VALUE_AND_BOUND: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes participate in exact mixed numeric comparisons without conversion through binary64.",
    },
    BuiltinIntegerInputCapability {
        name: "bound",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer, logical, single, and double compatible-size bounds retain their native values during comparison.",
    },
];

pub(super) const VALIDATOR_INTEGER_RANGE_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "All eight integer classes are accepted, and the documented form requires both range bounds to use the same class as A.",
    },
    BuiltinIntegerInputCapability {
        name: "lower",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "The lower bound must share A's integer class and is compared exactly.",
    },
    BuiltinIntegerInputCapability {
        name: "upper",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "The upper bound must share A's integer class and is compared exactly.",
    },
];

pub(super) const VALIDATOR_INTEGER_MEMBER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Membership reads authoritative integer elements exactly.",
    },
    BuiltinIntegerInputCapability {
        name: "S",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Unlike nondouble numeric inputs normally share a class; double retains the documented cross-class exception.",
    },
];
