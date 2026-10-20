use runmat_builtins::ASSIGNIN_ERROR_TOO_MANY_OUTPUTS;
use runmat_types::ASSIGNIN_BUILTIN_NAME;

pub fn assignin_too_many_outputs_error() -> crate::RuntimeError {
    crate::runtime_descriptor_error(ASSIGNIN_BUILTIN_NAME, &ASSIGNIN_ERROR_TOO_MANY_OUTPUTS)
}

#[runmat_macros::runtime_builtin(
    name = "eval",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    output_sequence = true,
    builtin_path = "crate::builtins::introspection::dynamic_workspace"
)]
pub async fn eval_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    crate::context::execute_selected_dynamic_workspace_retained(
        runtime,
        runmat_builtins::DynamicWorkspaceBuiltin::Eval,
        args,
        requested_outputs,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "evalc",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    output_sequence = true,
    builtin_path = "crate::builtins::introspection::dynamic_workspace"
)]
pub async fn evalc_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    crate::context::execute_selected_dynamic_workspace_retained(
        runtime,
        runmat_builtins::DynamicWorkspaceBuiltin::Evalc,
        args,
        requested_outputs,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "evalin",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    output_sequence = true,
    builtin_path = "crate::builtins::introspection::dynamic_workspace"
)]
pub async fn evalin_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    crate::context::execute_selected_dynamic_workspace_retained(
        runtime,
        runmat_builtins::DynamicWorkspaceBuiltin::Evalin,
        args,
        requested_outputs,
    )
    .await
}

#[runmat_macros::runtime_builtin(
    name = "assignin",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    output_sequence = true,
    builtin_path = "crate::builtins::introspection::dynamic_workspace"
)]
pub async fn assignin_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    crate::context::execute_selected_dynamic_workspace_retained(
        runtime,
        runmat_builtins::DynamicWorkspaceBuiltin::Assignin,
        args,
        requested_outputs,
    )
    .await
}

#[cfg(test)]
mod tests {
    #[test]
    fn dynamic_workspace_requires_explicit_execution_context() {
        use runmat_builtins::DynamicWorkspaceBuiltin;

        for builtin in [
            DynamicWorkspaceBuiltin::Eval,
            DynamicWorkspaceBuiltin::Evalc,
            DynamicWorkspaceBuiltin::Evalin,
            DynamicWorkspaceBuiltin::Assignin,
        ] {
            let err = futures::executor::block_on(
                crate::context::execute_selected_dynamic_workspace_retained(
                    None,
                    builtin,
                    crate::builtins::common::test_support::builtin_arguments(Vec::new()),
                    None,
                ),
            )
            .expect_err("dynamic workspace execution requires an explicit context");
            assert_eq!(err.identifier(), Some("RunMat:DynamicWorkspaceRequiresVm"));
        }
    }
}
