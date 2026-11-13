//! Canonical invocation-local callback selection over one exact registry.
//! The resolver borrows the registry only during selection; the returned
//! descriptor retains actual definition leases and the original callable roots.
use runmat_runtime::call::descriptor::{FunctionNameResolver, RepeatableCallable};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::RootedValue;
use runmat_runtime::RuntimeError;

struct UnitResolver<'a>(&'a runmat_vm::FunctionRegistry);
impl FunctionNameResolver for UnitResolver<'_> {
    fn name_resolution_completeness(
        &self,
    ) -> runmat_runtime::call::descriptor::NameResolutionCompleteness {
        runmat_runtime::call::descriptor::NameResolutionCompleteness::Complete
    }

    fn resolve_function(&self, name: &str) -> Option<runmat_types::FunctionId> {
        self.0
            .resolve_name(name)
            .map(|id| runmat_types::FunctionId(id.0))
    }
    fn resolve_bound_function(&self, name: &str) -> Option<runmat_value::BoundFunctionBinding> {
        self.0.exact_bound_binding(self.0.resolve_name(name)?)
    }
}

pub(crate) fn select(
    runtime: &RuntimeContext,
    registry: &runmat_vm::FunctionRegistry,
    callback: RootedValue,
) -> Result<RepeatableCallable, RuntimeError> {
    // resolve_name is the canonical published-name visibility rule: it
    // excludes nested functions. Actual bound/lexical handles retain their
    // immutable binding and activation through RepeatableCallable directly.
    RepeatableCallable::select_rooted(Some(runtime), callback, &UnitResolver(registry))
}
