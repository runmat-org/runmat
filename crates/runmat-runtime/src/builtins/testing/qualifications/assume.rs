use runmat_test::lifecycle::QualificationKind;
use runmat_value::Value;

use super::shared::{self, BinaryPredicate};
use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.assumeEqual",
    category = "testing/qualifications",
    summary = "Require equality or filter the current test.",
    builtin_path = "crate::builtins::testing::qualifications::assume"
)]
async fn assume_equal(
    receiver: Value,
    actual: Value,
    expected: Value,
    diagnostics: Vec<Value>,
) -> BuiltinResult<Value> {
    shared::qualify_binary(
        "assumeEqual",
        QualificationKind::AssumptionFailed,
        BinaryPredicate::Equal,
        receiver,
        actual,
        expected,
        diagnostics,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.assumeTrue",
    category = "testing/qualifications",
    summary = "Require logical truth or filter the current test.",
    builtin_path = "crate::builtins::testing::qualifications::assume"
)]
async fn assume_true(
    receiver: Value,
    actual: Value,
    diagnostics: Vec<Value>,
) -> BuiltinResult<Value> {
    shared::qualify_logical(
        "assumeTrue",
        QualificationKind::AssumptionFailed,
        true,
        receiver,
        actual,
        diagnostics,
    )
    .await
}
