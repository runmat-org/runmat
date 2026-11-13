use runmat_value::Value;

use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "testsuite",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::testsuite"
)]
async fn testsuite(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::TESTSUITE_DESCRIPTOR,
        &runmat_builtins::TESTSUITE_ERROR_TOO_MANY_OUTPUTS,
        "testsuite",
    )?;
    crate::testing::discover_tests_in_context(runtime, args).await
}
