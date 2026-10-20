use runmat_test::lifecycle::QualificationKind;
use runmat_value::Value;

use super::shared::{self, BinaryPredicate};
use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.assertEqual",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::assert"
)]
async fn assert_equal(
    receiver: Value,
    actual: Value,
    expected: Value,
    diagnostics: crate::call::BuiltinArguments,
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
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::assert"
)]
async fn assert_true(
    receiver: Value,
    actual: Value,
    diagnostics: crate::call::BuiltinArguments,
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
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::assert"
)]
fn assert_that(
    receiver: Value,
    actual: Value,
    constraint: Value,
    diagnostics: crate::call::BuiltinArguments,
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
