use runmat_test::lifecycle::QualificationKind;
use runmat_value::Value;

use super::shared;
use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.fatalAssertTrue",
    category = "testing/qualifications",
    summary = "Require logical truth or abort the run after safe teardown.",
    builtin_path = "crate::builtins::testing::qualifications::fatal"
)]
async fn fatal_assert_true(
    receiver: Value,
    actual: Value,
    diagnostics: Vec<Value>,
) -> BuiltinResult<Value> {
    shared::qualify_logical(
        "fatalAssertTrue",
        QualificationKind::FatalAssertionFailed,
        true,
        receiver,
        actual,
        diagnostics,
    )
    .await
}
