//! Canonical selected binding shared by VM tasks and worker callback batches.
use crate::FunctionRegistry;
use runmat_runtime::RuntimeError;

/// Semantic requirements travel together from the exact selected body.
pub struct CallableExecutionContract {
    pub capabilities: runmat_types::CapabilitySet,
    pub spawn_safety: runmat_types::SpawnSafetyFact,
}

/// Imported bodies renew their nonwire spawn evidence from validated MIR.
/// Names, numeric IDs and generic Future facts do not supply that evidence.
pub fn callable_execution_contract(
    descriptor: &runmat_runtime::call::descriptor::CallableDescriptor,
    registry: &FunctionRegistry,
) -> Result<CallableExecutionContract, RuntimeError> {
    use runmat_hir::CallableIdentity;
    use runmat_runtime::call::descriptor::CallableTarget;
    if let Some(name) = descriptor.target.selected_builtin_name() {
        return if runmat_builtins::builtin_name_is_known(name) {
            Ok(CallableExecutionContract {
                capabilities: runmat_builtins::builtin_required_capabilities(name),
                spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
            })
        } else {
            Err(crate::interpreter::errors::mex(
                "ExecutionProgram",
                "async builtin is absent from the exact builtin registry",
            ))
        };
    }
    let read = |body: &crate::bytecode::FunctionBytecode| {
        body.spawn_contract
            .as_ref()
            .map(|contract| CallableExecutionContract {
                capabilities: body.capabilities.clone(),
                spawn_safety: contract.safety().clone(),
            })
    };
    let selected = match &descriptor.target {
        CallableTarget::Bound(binding) | CallableTarget::BoundLexical { binding, .. } => {
            let function = runmat_hir::FunctionId(binding.function);
            registry
                .owned_view(function, &binding.authority)
                .and_then(|view| view.get(function).and_then(read))
        }
        CallableTarget::Resolved { identity, .. }
        | CallableTarget::CapturedResolved { identity, .. } => match identity.as_identity() {
            CallableIdentity::BoundFunction(function)
            | CallableIdentity::AnonymousFunction(function)
            | CallableIdentity::ExternalFunction { function, .. } => {
                registry.get(*function).and_then(read)
            }
            _ => None,
        },
        _ => None,
    };
    selected.ok_or_else(|| {
        crate::interpreter::errors::mex(
            "SpawnContractUnavailable",
            "async callable has no exact compiler-owned execution contract",
        )
    })
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
    let CallableExecutionContract {
        capabilities,
        spawn_safety,
    } = callable_execution_contract(&descriptor, registry)?;
    bind_selected_semantic_target(&mut descriptor, registry)?;
    let direct_builtin = if descriptor.target.selected_builtin_name().is_some() {
        Some(capture_builtin_program(runtime, registry)?)
    } else {
        None
    };
    Ok(runmat_runtime::execution::SelectedWorkerCallable::new(
        descriptor,
        capabilities,
        spawn_safety,
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
