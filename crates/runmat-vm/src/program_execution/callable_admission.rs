//! Canonical selected binding shared by VM tasks and worker callback batches.
use crate::FunctionRegistry;
use runmat_runtime::RuntimeError;

pub fn callable_capabilities(
    descriptor: &runmat_runtime::call::descriptor::CallableDescriptor,
    registry: &FunctionRegistry,
) -> Result<runmat_types::CapabilitySet, RuntimeError> {
    use runmat_hir::CallableIdentity;

    use runmat_runtime::call::descriptor::CallableTarget;

    if let Some(name) = descriptor.target.selected_builtin_name() {
        return if runmat_builtins::builtin_name_is_known(name) {
            Ok(runmat_builtins::builtin_required_capabilities(name))
        } else {
            Err(crate::interpreter::errors::mex(
                "ExecutionProgram",
                "async builtin is absent from the exact builtin registry",
            ))
        };
    }

    if let CallableTarget::Bound(binding) | CallableTarget::BoundLexical { binding, .. } =
        &descriptor.target
    {
        let function = runmat_hir::FunctionId(binding.function);
        let selected = registry
            .owned_view(function, &binding.authority)
            .and_then(|view| view.get(function).map(|body| body.capabilities.clone()))
            .ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ExecutionProgram",
                    "async bound callable has no exact selected definition",
                )
            })?;
        return Ok(selected);
    }
    let identity = match &descriptor.target {
        CallableTarget::Resolved { identity, .. }
        | CallableTarget::CapturedResolved { identity, .. } => identity.as_identity(),
        _ => {
            return Err(crate::interpreter::errors::mex(
                "ExecutionProgram",
                "async execution requires an exact selected callable before capability binding",
            ));
        }
    };
    match identity {
        CallableIdentity::BoundFunction(function)
        | CallableIdentity::AnonymousFunction(function)
        | CallableIdentity::ExternalFunction { function, .. } => registry
            .get(*function)
            .map(|function| function.capabilities.clone())
            .ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "ExecutionProgram",
                    "async callable is absent from the exact function registry",
                )
            }),
        CallableIdentity::Builtin(name) if runmat_builtins::builtin_name_is_known(&name.0) => {
            Ok(runmat_builtins::builtin_required_capabilities(&name.0))
        }
        CallableIdentity::Builtin(_) => Err(crate::interpreter::errors::mex(
            "ExecutionProgram",
            "async builtin is absent from the exact builtin registry",
        )),
        _ => Err(crate::interpreter::errors::mex(
            "ExecutionProgram",
            "async callable kind has no portable execution identity",
        )),
    }
}

/// A semantic name is resolved only at Future admission, while its selected
/// registry and source owner are still present. The resulting local authority
/// retains the exact body/dependency leases; the worker snapshot never
/// resolves the name or a reused numeric function ID.
pub fn bind_selected_semantic_target(
    descriptor: &mut runmat_runtime::call::descriptor::CallableDescriptor,
    registry: &FunctionRegistry,
) -> Result<(), RuntimeError> {
    use runmat_hir::CallableIdentity;
    use runmat_runtime::call::descriptor::CallableTarget;

    let identity = match &descriptor.target {
        CallableTarget::Resolved { identity, .. }
        | CallableTarget::CapturedResolved { identity, .. } => identity.as_identity(),
        _ => return Ok(()),
    };
    let function = match identity {
        CallableIdentity::BoundFunction(function)
        | CallableIdentity::AnonymousFunction(function)
        | CallableIdentity::ExternalFunction { function, .. } => *function,
        _ => return Ok(()),
    };
    let binding = registry.exact_bound_binding(function).ok_or_else(|| {
        crate::interpreter::errors::mex(
            "ExecutionProgram",
            "async semantic callable has no exact selected definition and source graph",
        )
    })?;
    descriptor.target = match &descriptor.target {
        CallableTarget::CapturedResolved { closure, .. } => CallableTarget::BoundLexical {
            binding,
            closure: closure.clone(),
        },
        _ => CallableTarget::Bound(binding),
    };
    Ok(())
}

pub fn select_worker_callable(
    runtime: &runmat_runtime::context::RuntimeContext,
    target: runmat_runtime::execution::RootedValue,
    registry: &FunctionRegistry,
) -> Result<runmat_runtime::execution::SelectedWorkerCallable, RuntimeError> {
    let target = runmat_runtime::call::descriptor::RepeatableCallable::select_rooted(
        Some(runtime),
        target,
        registry,
    )?;
    select_retained_worker_callable(runtime, target, registry)
}

/// Admit the existing immutable selected target. No name is resolved again;
/// capabilities, exact binding proof and builtin program capture remain the
/// same owners used by ordinary worker callback selection.
pub fn select_retained_worker_callable(
    runtime: &runmat_runtime::context::RuntimeContext,
    target: runmat_runtime::call::descriptor::RepeatableCallable,
    registry: &FunctionRegistry,
) -> Result<runmat_runtime::execution::SelectedWorkerCallable, RuntimeError> {
    let mut descriptor = target.selected_descriptor().clone();
    let capabilities = callable_capabilities(&descriptor, registry)?;
    bind_selected_semantic_target(&mut descriptor, registry)?;
    let direct_builtin = if descriptor.target.selected_builtin_name().is_some() {
        Some(capture_builtin_program(runtime, registry)?)
    } else {
        None
    };
    Ok(runmat_runtime::execution::SelectedWorkerCallable::new(
        descriptor,
        capabilities,
        target,
        direct_builtin,
    ))
}

fn capture_builtin_program(
    runtime: &runmat_runtime::context::RuntimeContext,
    registry: &FunctionRegistry,
) -> Result<std::sync::Arc<[u8]>, RuntimeError> {
    runmat_runtime::context::host_allocation::require_unbounded_workspace(
        Some(runtime),
        "parallel builtin callback",
        "selected builtin program and task metadata have no complete finite workspace bound",
    )?;
    crate::encode_interpreter_program_v2(registry)
        .map(std::sync::Arc::from)
        .map_err(|error| {
            crate::interpreter::errors::mex(
                "ExecutionProgram",
                &format!("failed to capture selected builtin program: {error}"),
            )
        })
}

#[cfg(test)]
#[path = "callable_admission/tests.rs"]
mod tests;
