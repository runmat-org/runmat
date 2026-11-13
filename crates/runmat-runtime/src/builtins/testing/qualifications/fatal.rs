use runmat_test::lifecycle::QualificationKind;
use runmat_value::Value;

use super::shared;
use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.fatalAssertTrue",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::fatal"
)]
async fn fatal_assert_true(
    receiver: Value,
    actual: Value,
    diagnostics: crate::call::BuiltinArguments,
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
