use runmat_builtins::Value;
use runmat_test::lifecycle::QualificationKind;

use super::shared::{self, BinaryPredicate};
use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.assertEqual",
    category = "testing/qualifications",
    summary = "Assert equality and abort the current test on failure.",
    builtin_path = "crate::builtins::testing::qualifications::assert"
)]
async fn assert_equal(
    receiver: Value,
    actual: Value,
    expected: Value,
    diagnostics: Vec<Value>,
) -> BuiltinResult<Value> {
    shared::qualify_binary(
        "assertEqual",
        QualificationKind::AssertionFailed,
        BinaryPredicate::Equal,
        receiver,
        actual,
        expected,
        diagnostics,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.assertTrue",
    category = "testing/qualifications",
    summary = "Assert logical truth and abort the current test on failure.",
    builtin_path = "crate::builtins::testing::qualifications::assert"
)]
async fn assert_true(
    receiver: Value,
    actual: Value,
    diagnostics: Vec<Value>,
) -> BuiltinResult<Value> {
    shared::qualify_logical(
        "assertTrue",
        QualificationKind::AssertionFailed,
        true,
        receiver,
        actual,
        diagnostics,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.assertThat",
    category = "testing/qualifications",
    summary = "Assert that a value satisfies a compatible constraint.",
    builtin_path = "crate::builtins::testing::qualifications::assert"
)]
fn assert_that(
    receiver: Value,
    actual: Value,
    constraint: Value,
    diagnostics: Vec<Value>,
) -> BuiltinResult<Value> {
    shared::qualify_that(
        "assertThat",
        QualificationKind::AssertionFailed,
        receiver,
        actual,
        constraint,
        diagnostics,
    )
}
