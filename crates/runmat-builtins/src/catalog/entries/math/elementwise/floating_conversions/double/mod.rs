mod documentation;

use super::contract::*;
use documentation::DOUBLE_DOCUMENTATION;

define_floating_conversion_contract!(
    "double",
    "DOUBLE",
    NumericClass::Double,
    crate::BuiltinPreparationProviderRequirement::F64,
    &[DEFAULT_PREPARATION_FORM, crate::BuiltinPreparationForm {
        id: crate::BuiltinPreparationFormId(1),
        binding_variant: "default",
        signatures: &[0],
        data_arguments: &[0],
        operation: crate::BuiltinPreparationOperation::FloatingCast(NumericClass::Double),
        provider_requirement: crate::BuiltinPreparationProviderRequirement::F64,
    }, LIKE_PREPARATION_FORM],
    DOUBLE_DOCUMENTATION,
    "Double-precision output value.",
    "binary64 storage",
    "double-like-prototype",
    "double(X, 'like', prototype) is a RunMat extension",
    "RunMat:compatibility:DoubleLikePrototypeExtension",
    BuiltinIntegerBackendRule::GatherFallback,
    BuiltinIntegerOutputClassRule::Double,
    "Every fixed-width integer class converts directly to IEEE binary64; wide int64 and uint64 values may round.",
    "Host conversion reads authoritative integer storage. Resident conversion uses the owning provider only when it can produce true F64 and otherwise gathers."
);
