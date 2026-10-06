use super::*;

pub(super) async fn dispatch(
    ctx: CallDispatchContext<'_>,
) -> Result<Option<DispatchHandled>, RuntimeError> {
    let CallDispatchContext { instr, stack, .. } = ctx;
    match instr {
        Instr::CallSuperConstructorMulti {
            current_class,
            super_class,
            arg_count,
            out_count,
        } => {
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let _roots = crate::call::root_values(&args)?;
            let result = runmat_runtime::call_super_constructor(
                current_class.clone(),
                super_class.clone(),
                args,
            )
            .await?;
            calls::publish_fixed_outputs(
                stack,
                runmat_runtime::call::arguments::adapt_legacy_builtin_result(result)?,
                *out_count,
            )?;
        }
        Instr::CallSuperMethodMulti {
            current_class,
            super_class,
            method,
            arg_count,
            out_count,
        } => {
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let _roots = crate::call::root_values(&args)?;
            let result = runmat_runtime::call_super_method(
                current_class.clone(),
                super_class.clone(),
                method.clone(),
                args,
            )
            .await?;
            calls::publish_fixed_outputs(
                stack,
                runmat_runtime::call::arguments::adapt_legacy_builtin_result(result)?,
                *out_count,
            )?;
        }
        _ => unreachable!("superclass route received another instruction"),
    }
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}
