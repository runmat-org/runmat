use runmat_value::Value;

mod base;
pub mod bindings;
mod checked;
mod completion;
mod detach;
mod dynamic_target;
mod logical_generation;
#[cfg(test)]
#[path = "workspace/resolver_tests.rs"]
mod resolver_tests;
mod retained;
pub mod session;
mod slot_read_lease;

pub use base::base_workspace_error_to_runtime;
pub use base::{
    assign_logical_base_value, assign_logical_base_value_ref_with_context,
    assign_logical_base_value_with_context, lookup_logical_base_value, BaseWorkspaceBindingNames,
    BaseWorkspaceCommit, BaseWorkspaceError, BaseWorkspaceOwner, BaseWorkspaceProjection,
    BaseWorkspaceProjectionService, BaseWorkspaceReconciliation, BaseWorkspaceService,
    BaseWorkspaceSnapshot, PreparedWorkspaceCompletion, WorkspaceChange, WorkspaceChangeKind,
    WorkspaceDelta, WorkspaceProjectionPublication, WorkspacePublicationBatch,
    WorkspacePublicationMutation, WorkspacePublicationObserver, WorkspacePublicationOrigin,
    WorkspacePublicationProtocol, WorkspacePublicationStamp, WorkspacePublicationToken,
    WorkspaceRevision,
};
pub use bindings::GlobalBindingTarget;
pub use checked::{
    base_workspace_owner, begin_base_workspace_projection, try_assign, try_clear,
    try_detach_selected, try_global_names, try_lookup, try_remove, try_snapshot,
};
pub use completion::{
    WorkspaceCompletionAction, WorkspaceCompletionIdentity, WorkspaceCompletionParticipant,
    WorkspaceCompletionSource, WorkspaceCompletionValue,
};
pub use detach::{DetachedWorkspaceValues, WorkspaceDetachSelection};
pub use dynamic_target::{
    CapturedWorkspaceReadView, DynamicWorkspaceOverlayEntry, DynamicWorkspaceTargetCapture,
    DynamicWorkspaceTargetEntry, DynamicWorkspaceTargetOverlay,
};
pub use logical_generation::LogicalWorkspaceGeneration;
pub use retained::{WorkspaceName, WorkspaceValueReadLease};
pub use slot_read_lease::WorkspaceSlotReadLease;

#[cfg(test)]
use once_cell::sync::Lazy;
#[cfg(test)]
use std::sync::Mutex;

/// Resolver used by the runtime to access the caller workspace when builtins
/// (such as `save`) need to look up variables by name.
type AssignFn = fn(&str, Value) -> Result<(), crate::RuntimeError>;
type ClearFn = fn() -> Result<(), crate::RuntimeError>;
type RemoveFn = fn(&str) -> Result<(), crate::RuntimeError>;

#[derive(Clone, Copy, Debug)]
pub struct WorkspaceResolver {
    pub lookup: fn(&str) -> Option<Value>,
    pub snapshot: fn() -> Vec<(String, Value)>,
    pub globals: fn() -> Vec<String>,
    pub assign: Option<AssignFn>,
    pub clear: Option<ClearFn>,
    pub remove: Option<RemoveFn>,
}

mod resolver_storage {
    use super::WorkspaceResolver;

    pub(super) fn set(resolver: WorkspaceResolver) {
        imp::set(resolver)
    }

    pub(super) fn with<R>(f: impl FnOnce(Option<&WorkspaceResolver>) -> R) -> R {
        imp::with(f)
    }

    #[cfg(test)]
    mod imp {
        use super::WorkspaceResolver;
        use std::cell::RefCell;

        // In tests, the resolver is frequently swapped by many modules. Using a global resolver
        // makes tests flaky under the default parallel test runner.
        // Thread-local storage matches the "resolver is tied to an executing context" model and
        // avoids cross-test interference.
        thread_local! {
            static RESOLVER: RefCell<Option<WorkspaceResolver>> = const { RefCell::new(None) };
        }

        pub(super) fn set(resolver: WorkspaceResolver) {
            RESOLVER.with(|slot| {
                *slot.borrow_mut() = Some(resolver);
            });
        }

        pub(super) fn with<R>(f: impl FnOnce(Option<&WorkspaceResolver>) -> R) -> R {
            RESOLVER.with(|slot| {
                let guard = slot.borrow();
                f(guard.as_ref())
            })
        }
    }

    #[cfg(not(test))]
    mod imp {
        use super::WorkspaceResolver;
        use once_cell::sync::Lazy;
        use std::sync::RwLock;

        static RESOLVER: Lazy<RwLock<Option<WorkspaceResolver>>> = Lazy::new(|| RwLock::new(None));

        pub(super) fn set(resolver: WorkspaceResolver) {
            let mut guard = RESOLVER
                .write()
                .unwrap_or_else(|poison| poison.into_inner());
            *guard = Some(resolver);
        }

        pub(super) fn with<R>(f: impl FnOnce(Option<&WorkspaceResolver>) -> R) -> R {
            let guard = RESOLVER.read().unwrap_or_else(|poison| poison.into_inner());
            f(guard.as_ref())
        }
    }
}

#[cfg(test)]
static TEST_WORKSPACE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

/// Register the workspace resolver. The VM installs this once during
/// initialization so that language builtins can query variables lazily.
pub fn register_workspace_resolver(resolver: WorkspaceResolver) {
    if let Some(context) = crate::context::legacy::active() {
        context.state().workspace.replace(Some(resolver));
    }
    // The VM resolver consists only of context-independent function pointers. Retain it as the
    // standalone fallback as well: one-time registration may occur while a RuntimeContext is
    // active, and later interpreter sessions must not depend on that first context's lifetime.
    resolver_storage::set(resolver);
}

/// Lookup a variable by name in the active workspace.
pub fn lookup(name: &str) -> Option<Value> {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service.lookup(name);
        }
        if let Some(resolver) = context.state().workspace.borrow().as_ref() {
            return (resolver.lookup)(name);
        }
    }
    resolver_storage::with(|resolver| resolver.and_then(|r| (r.lookup)(name)))
}

/// Snapshot the active workspace into a vector of `(name, value)` pairs.
/// Returns `None` when no resolver/workspace is active.
pub fn snapshot() -> Option<Vec<(String, Value)>> {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return Some(service.snapshot());
        }
        if let Some(resolver) = context.state().workspace.borrow().as_ref() {
            return Some((resolver.snapshot)());
        }
    }
    resolver_storage::with(|resolver| resolver.map(|r| (r.snapshot)()))
}

/// Return the list of global variable names visible to the active workspace.
pub fn global_names() -> Vec<String> {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service.global_names();
        }
        if let Some(resolver) = context.state().workspace.borrow().as_ref() {
            return (resolver.globals)();
        }
    }
    resolver_storage::with(|resolver| resolver.map(|r| (r.globals)()).unwrap_or_default())
}

pub fn assign(name: &str, value: Value) -> Result<(), crate::RuntimeError> {
    try_assign(name, value)
}

pub fn clear() -> Result<(), crate::RuntimeError> {
    try_clear()
}

pub fn remove(name: &str) -> Result<(), crate::RuntimeError> {
    try_remove(name)
}

/// Returns true when a resolver has been registered.
pub fn is_available() -> bool {
    if let Some(context) = crate::context::legacy::active() {
        if context.service_ports().workspace().is_some()
            || context.state().workspace.borrow().is_some()
        {
            return true;
        }
    }
    resolver_storage::with(|resolver| resolver.is_some())
}

fn with_context_resolver<R>(
    context: &crate::context::RuntimeContext,
    callback: impl FnOnce(&WorkspaceResolver) -> Result<R, crate::RuntimeError>,
) -> Result<R, crate::RuntimeError> {
    let resolver = context.state().workspace.borrow();
    callback(resolver.as_ref().ok_or_else(|| {
        crate::runtime_error::semantic_error(
            "RunMat:Workspace:OperationUnavailable",
            "workspace state unavailable",
        )
    })?)
}

#[cfg(test)]
pub(crate) fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    TEST_WORKSPACE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}
