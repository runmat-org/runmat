mod error_map;

use crate::builtins::common::mapped_callable::MappedCallable;
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_builtins::BuiltinCatalogIdentity;
use runmat_value::Value;

#[derive(Clone)]
pub(super) struct Callable(MappedCallable);

impl Callable {
    pub(in crate::builtins::acceleration::gpu::arrayfun) fn from_function(
        value: Value,
    ) -> BuiltinResult<Self> {
        MappedCallable::parse(value)
            .map(Self)
            .map_err(error_map::parse)
    }

    pub(super) fn builtin_identity(&self) -> Option<BuiltinCatalogIdentity> {
        self.0.builtin_identity()
    }

    pub(in crate::builtins::acceleration::gpu::arrayfun) async fn call(
        &self,
        context: Option<&crate::context::RuntimeContext>,
        arguments: &[Value],
    ) -> BuiltinResult<RootedValue> {
        self.0
            .invoke_with_outputs_in_context(context, arguments, 1)
            .await
            .map_err(error_map::call)
    }
}
