//! Value-only exact callable fixtures. Tests install their own typed invoker
//! in the same runtime library copy and session that dispatches the call.

use crate::context::RuntimeContext;
use runmat_value::{BoundFunctionBinding, FunctionDefinitionOwner, LocalExport};
use std::sync::Arc;

pub fn exact_local_binding(runtime: &RuntimeContext, function: usize) -> BoundFunctionBinding {
    let owner = Arc::new(FunctionDefinitionOwner::default());
    BoundFunctionBinding::local(
        function,
        runtime.session_lease(),
        FunctionDefinitionOwner::issue(&owner, function),
        Vec::new(),
        LocalExport::Ephemeral,
    )
}
