use crate::bytecode::FunctionRegistry;
use crate::call::execute_callable_descriptor;
use runmat_runtime::call::descriptor::CallableDescriptor;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

/// Private names belong to the actual calling definition, not the registry's
/// global name index. Retained callers reuse that definition's existing lease.
struct CallerNames<'a> {
    registry: &'a FunctionRegistry,
    caller: Option<runmat_hir::FunctionId>,
}

impl<'a> CallerNames<'a> {
    fn capture(
        runtime: &runmat_runtime::context::RuntimeContext,
        registry: &'a FunctionRegistry,
    ) -> Self {
        let _scope = runtime.enter();
        Self {
            registry,
            caller: runmat_runtime::user_functions::current_active_semantic_function()
                .map(runmat_hir::FunctionId),
        }
    }
}

impl runmat_runtime::call::descriptor::FunctionNameResolver for CallerNames<'_> {
    fn resolve_function(&self, name: &str) -> Option<runmat_types::FunctionId> {
        self.caller
            .and_then(|caller| self.registry.resolve_private_name_for_caller(caller, name))
            .or_else(|| self.registry.resolve_name(name))
    }

    fn resolve_bound_function(&self, name: &str) -> Option<runmat_value::BoundFunctionBinding> {
        self.caller.and_then(|caller| {
            self.registry
                .scoped_private_binding_for_caller(caller, name)
        })
    }

    fn name_resolution_completeness(
        &self,
    ) -> runmat_runtime::call::descriptor::NameResolutionCompleteness {
        runmat_runtime::call::descriptor::NameResolutionCompleteness::Complete
    }
}

pub enum FevalDispatch {
    Completed(runmat_runtime::execution::RootedValueSequence),
}

/// The same VM selector and registry with canonical retained argument slots.
/// A target-only descriptor carries no discarded ordinary argument backing.
pub async fn execute_feval_retained(
    runtime: &runmat_runtime::context::RuntimeContext,
    func_val: Value,
    args: runmat_runtime::call::BuiltinArguments,
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
) -> Result<FevalDispatch, RuntimeError> {
    let _target_roots =
        runmat_runtime::execution::RootedValueLease::for_values_iterator_in_context(
            std::iter::once(&func_val),
            runtime,
        )?;
    let descriptor = CallableDescriptor::from_feval_value(
        func_val,
        Vec::new(),
        requested_outputs,
        &CallerNames::capture(runtime, function_registry),
    )
    .with_retained_arguments(args)?;
    Ok(FevalDispatch::Completed(
        runmat_runtime::call::descriptor::execute_retained_callable_descriptor(
            Some(runtime),
            descriptor,
        )
        .await?,
    ))
}

pub async fn execute_feval(
    runtime: &runmat_runtime::context::RuntimeContext,
    func_val: Value,
    args: Vec<Value>,
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
) -> Result<FevalDispatch, RuntimeError> {
    let _roots = crate::call::root_values(args.iter().chain(std::iter::once(&func_val)))?;
    let descriptor = CallableDescriptor::from_feval_value(
        func_val,
        args,
        requested_outputs,
        &CallerNames::capture(runtime, function_registry),
    );
    Ok(FevalDispatch::Completed(
        execute_callable_descriptor(runtime, descriptor).await?,
    ))
}
