//! Allocation-free canonical name classification. Owning dispatch identities
//! are created only by the already-owned selector; retained selection keeps
//! its original admitted text backing instead.
use super::{CallableDescriptor, FunctionNameResolver};
use runmat_types::{BuiltinId, CallableFallbackPolicy, CallableIdentity, SymbolName};

pub(super) enum Selected {
    Bound(runmat_value::BoundFunctionBinding),
    Resolved(Resolved),
}
pub(super) enum Resolved {
    Function(runmat_types::FunctionId),
    Builtin,
    Qualified,
    Dynamic,
}
pub(super) fn select(name: &str, resolver: &impl FunctionNameResolver) -> Selected {
    if let Some(binding) = resolver.resolve_bound_function(name) {
        Selected::Bound(binding)
    } else {
        Selected::Resolved(resolve(name, resolver))
    }
}
pub(super) fn resolve(name: &str, resolver: &impl FunctionNameResolver) -> Resolved {
    if let Some(function) = resolver.resolve_function(name) {
        Resolved::Function(function)
    } else if runmat_builtins::builtin_name_is_known(name) {
        Resolved::Builtin
    } else if CallableDescriptor::is_well_formed_qualified_name(name) {
        Resolved::Qualified
    } else {
        Resolved::Dynamic
    }
}
impl Resolved {
    pub(super) fn owned(self, name: &str) -> (CallableIdentity, CallableFallbackPolicy) {
        match self {
            Self::Function(function) => (
                CallableIdentity::BoundFunction(function),
                CallableFallbackPolicy::None,
            ),
            Self::Builtin => (
                CallableIdentity::Builtin(BuiltinId(name.to_owned())),
                CallableFallbackPolicy::None,
            ),
            Self::Qualified => (
                CallableDescriptor::qualified_identity_from_name(name),
                CallableFallbackPolicy::ExternalBoundary,
            ),
            Self::Dynamic => (
                CallableIdentity::DynamicName(SymbolName(name.to_owned())),
                CallableFallbackPolicy::RuntimeNameResolution,
            ),
        }
    }
}

/// Text targets use the same admitted text grammar as feval; handle names
/// remain exact, matching the owned handle selector rather than trimming anew.
pub(super) fn retained_name(value: &runmat_value::Value) -> Option<&str> {
    match value {
        runmat_value::Value::String(text) => text
            .as_str()
            .and_then(crate::call::admission::text_target_name),
        runmat_value::Value::StringArray(array) if array.data().len() == 1 => array.data()[0]
            .as_str()
            .and_then(crate::call::admission::text_target_name),
        runmat_value::Value::FunctionHandle(name)
        | runmat_value::Value::ExternalFunctionHandle(name) => Some(name.name()),
        _ => None,
    }
}
