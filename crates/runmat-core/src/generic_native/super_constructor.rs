//! Issue superclass calls from the native invocation's selected VM registry.

pub(super) fn bind_selected(
    runtime: runmat_runtime::context::RuntimeContext,
    registry: &runmat_vm::FunctionRegistry,
    caller: runmat_hir::FunctionId,
    execution_sources: &runmat_runtime::user_functions::ExecutionSourceFacts,
) -> Result<runmat_runtime::context::RuntimeContext, runmat_runtime::RuntimeError> {
    let targets = registry
        .retained_super_constructor_targets_for(caller)
        .map_err(|error| super::error::stage("NativeSuperConstructorOwner", error))?;
    let implicit = registry
        .retained_implicit_super_constructor_targets_for(caller)
        .map_err(|error| super::error::stage("NativeSuperConstructorOwner", error))?;
    let Some(caller_origin) = registry.get(caller).and_then(|body| body.binding_origin) else {
        if targets.is_empty() && implicit.is_empty() {
            return Ok(runtime);
        }
        return Err(super::error::stage(
            "NativeSuperConstructorOwner",
            "superclass caller lacks immutable source identity",
        ));
    };
    if targets.is_empty() && implicit.is_empty() {
        return Ok(runtime.with_retained_super_caller_origin(caller_origin));
    }
    let mut bindings = targets
        .into_iter()
        .map(|target| {
            let binding =
                runmat_runtime::user_functions::resolve_binding_origin_binding_in_context(
                    execution_sources,
                    target.origin,
                    &runtime,
                )
                .ok_or_else(|| {
                    super::error::stage(
                        "NativeSuperConstructorOwner",
                        "selected superclass lacks its exact executable definition",
                    )
                })?;
            runmat_runtime::context::RetainedSuperConstructorBinding::from_admitted_definition(
                &target.caller_declaration,
                &target.target_declaration,
                target.caller_origin,
                target.origin,
                target.owner,
                binding,
            )
        })
        .collect::<Result<Vec<_>, runmat_runtime::RuntimeError>>()?;
    bindings.extend(
        implicit
            .into_iter()
            .map(|target| target.into_binding())
            .collect::<Result<Vec<_>, runmat_runtime::RuntimeError>>()?,
    );
    runtime.with_retained_super_constructors(bindings, caller_origin)
}
