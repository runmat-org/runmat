use runmat_builtins::SETFIELD_ERROR_NOT_ENOUGH_INPUTS;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "setfield",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::structs::core::setfield"
)]
pub(super) async fn setfield_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    base: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::SETFIELD_DESCRIPTOR,
        &runmat_builtins::SETFIELD_ERROR_TOO_MANY_OUTPUTS,
        "setfield",
    )?;

    if rest.len() < 2 {
        return Err(super::errors::with_message(
            SETFIELD_ERROR_NOT_ENOUGH_INPUTS.message,
            &SETFIELD_ERROR_NOT_ENOUGH_INPUTS,
        ));
    }
    let mut arguments = rest;
    let value = arguments.pop().expect("input count is validated above");
    let parsed = super::arguments::parse(arguments)?;
    super::assignment::assign_value(
        runtime.as_ref(),
        base,
        parsed.leading_index,
        parsed.fields,
        value,
    )
    .await
}
