use runmat_test::lifecycle::QualificationKind;
use runmat_value::Value;

use super::shared::{self, BinaryPredicate};
use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.verifyEqual",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_equal(
    receiver: Value,
    actual: Value,
    expected: Value,
    diagnostics: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    shared::qualify_binary(
        "verifyEqual",
        QualificationKind::VerificationFailed,
        BinaryPredicate::Equal,
        receiver,
        actual,
        expected,
        diagnostics,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.verifyNotEqual",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_not_equal(
    receiver: Value,
    actual: Value,
    expected: Value,
    diagnostics: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    shared::qualify_binary(
        "verifyNotEqual",
        QualificationKind::VerificationFailed,
        BinaryPredicate::NotEqual,
        receiver,
        actual,
        expected,
        diagnostics,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.verifyTrue",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_true(
    receiver: Value,
    actual: Value,
    diagnostics: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    shared::qualify_logical(
        "verifyTrue",
        QualificationKind::VerificationFailed,
        true,
        receiver,
        actual,
        diagnostics,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.verifyFalse",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_false(
    receiver: Value,
    actual: Value,
    diagnostics: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    shared::qualify_logical(
        "verifyFalse",
        QualificationKind::VerificationFailed,
        false,
        receiver,
        actual,
        diagnostics,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.verifyGreaterThan",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_greater_than(
    receiver: Value,
    actual: Value,
    expected: Value,
    diagnostics: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    shared::qualify_binary(
        "verifyGreaterThan",
        QualificationKind::VerificationFailed,
        BinaryPredicate::GreaterThan,
        receiver,
        actual,
        expected,
        diagnostics,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.verifyLessThan",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_less_than(
    receiver: Value,
    actual: Value,
    expected: Value,
    diagnostics: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    shared::qualify_binary(
        "verifyLessThan",
        QualificationKind::VerificationFailed,
        BinaryPredicate::LessThan,
        receiver,
        actual,
        expected,
        diagnostics,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.verifyEmpty",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_empty(
    receiver: Value,
    actual: Value,
    diagnostics: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    shared::qualify_empty(
        "verifyEmpty",
        QualificationKind::VerificationFailed,
        true,
        receiver,
        actual,
        diagnostics,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.verifyNotEmpty",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_not_empty(
    receiver: Value,
    actual: Value,
    diagnostics: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    shared::qualify_empty(
        "verifyNotEmpty",
        QualificationKind::VerificationFailed,
        false,
        receiver,
        actual,
        diagnostics,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.verifyThat",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
fn verify_that(
    receiver: Value,
    actual: Value,
    constraint: Value,
    diagnostics: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    shared::qualify_that(
        "verifyThat",
        QualificationKind::VerificationFailed,
        receiver,
        actual,
        constraint,
        diagnostics,
    )
}
