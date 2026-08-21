use runmat_test::lifecycle::QualificationKind;
use runmat_value::Value;

use super::shared::{self, BinaryPredicate};
use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.verifyEqual",
    category = "testing/qualifications",
    summary = "Verify that two values are equal without aborting the test.",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_equal(
    receiver: Value,
    actual: Value,
    expected: Value,
    diagnostics: Vec<Value>,
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
    category = "testing/qualifications",
    summary = "Verify that two values are not equal without aborting the test.",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_not_equal(
    receiver: Value,
    actual: Value,
    expected: Value,
    diagnostics: Vec<Value>,
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
    category = "testing/qualifications",
    summary = "Verify that a value is logical true without aborting the test.",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_true(
    receiver: Value,
    actual: Value,
    diagnostics: Vec<Value>,
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
    category = "testing/qualifications",
    summary = "Verify that a value is logical false without aborting the test.",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_false(
    receiver: Value,
    actual: Value,
    diagnostics: Vec<Value>,
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
    category = "testing/qualifications",
    summary = "Verify that the actual scalar is greater than the expected scalar.",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_greater_than(
    receiver: Value,
    actual: Value,
    expected: Value,
    diagnostics: Vec<Value>,
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
    category = "testing/qualifications",
    summary = "Verify that the actual scalar is less than the expected scalar.",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_less_than(
    receiver: Value,
    actual: Value,
    expected: Value,
    diagnostics: Vec<Value>,
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
    category = "testing/qualifications",
    summary = "Verify that a value has no elements.",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_empty(
    receiver: Value,
    actual: Value,
    diagnostics: Vec<Value>,
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
    category = "testing/qualifications",
    summary = "Verify that a value has at least one element.",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
async fn verify_not_empty(
    receiver: Value,
    actual: Value,
    diagnostics: Vec<Value>,
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
    category = "testing/qualifications",
    summary = "Verify that a value satisfies a compatible constraint.",
    builtin_path = "crate::builtins::testing::qualifications::verify"
)]
fn verify_that(
    receiver: Value,
    actual: Value,
    constraint: Value,
    diagnostics: Vec<Value>,
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
