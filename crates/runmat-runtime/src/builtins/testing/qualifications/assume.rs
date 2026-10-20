use runmat_test::lifecycle::QualificationKind;
use runmat_value::Value;

use super::shared::{self, BinaryPredicate};
use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.assumeEqual",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::assume"
)]
async fn assume_equal(
    receiver: Value,
    actual: Value,
    expected: Value,
    diagnostics: crate::call::BuiltinArguments,
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
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::assume"
)]
async fn assume_true(
    receiver: Value,
    actual: Value,
    diagnostics: crate::call::BuiltinArguments,
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
