use once_cell::sync::Lazy;
use runmat_builtins::Value;
use std::cell::RefCell;
use std::sync::RwLock;

/// Resolver used by the runtime to access the caller workspace when builtins
/// (such as `save`) need to look up variables by name.
#[derive(Clone, Copy)]
pub struct WorkspaceResolver {
    pub lookup: fn(&str) -> Option<Value>,
    pub snapshot: fn() -> Vec<(String, Value)>,
    pub globals: fn() -> Vec<String>,
}

static RESOLVER: Lazy<RwLock<Option<WorkspaceResolver>>> = Lazy::new(|| RwLock::new(None));
thread_local! {
    static TLS_RESOLVER: RefCell<Option<WorkspaceResolver>> = const { RefCell::new(None) };
}

/// Register the workspace resolver. Ignition installs this once during
/// initialization so that language builtins can query variables lazily.
pub fn register_workspace_resolver(resolver: WorkspaceResolver) {
    if let Ok(mut guard) = RESOLVER.write() {
        *guard = Some(resolver);
    }
}

/// Unregister the workspace resolver, leaving the global hook empty.
pub fn unregister_workspace_resolver() {
    if let Ok(mut guard) = RESOLVER.write() {
        *guard = None;
    }
}

fn current_workspace_resolver() -> Option<WorkspaceResolver> {
    TLS_RESOLVER.with(|slot| *slot.borrow()).or_else(|| {
        RESOLVER
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().cloned())
    })
}

pub struct ThreadWorkspaceGuard {
    previous: Option<WorkspaceResolver>,
}

impl ThreadWorkspaceGuard {
    fn install(resolver: Option<WorkspaceResolver>) -> Self {
        let previous = TLS_RESOLVER.with(|slot| {
            let mut guard = slot.borrow_mut();
            let previous = guard.take();
            *guard = resolver;
            previous
        });
        Self { previous }
    }
}

impl Drop for ThreadWorkspaceGuard {
    fn drop(&mut self) {
        TLS_RESOLVER.with(|slot| {
            *slot.borrow_mut() = self.previous.take();
        });
    }
}

pub fn replace_thread_local_workspace_resolver(
    resolver: Option<WorkspaceResolver>,
) -> ThreadWorkspaceGuard {
    ThreadWorkspaceGuard::install(resolver)
}

/// Lookup a variable by name in the active workspace.
pub fn lookup(name: &str) -> Option<Value> {
    current_workspace_resolver().and_then(|resolver| (resolver.lookup)(name))
}

/// Snapshot the active workspace into a vector of `(name, value)` pairs.
/// Returns `None` when no resolver/workspace is active.
pub fn snapshot() -> Option<Vec<(String, Value)>> {
    current_workspace_resolver().map(|resolver| (resolver.snapshot)())
}

/// Return the list of global variable names visible to the active workspace.
pub fn global_names() -> Vec<String> {
    current_workspace_resolver()
        .map(|resolver| (resolver.globals)())
        .unwrap_or_default()
}

/// Returns true when a resolver has been registered.
pub fn is_available() -> bool {
    current_workspace_resolver().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dummy_lookup(_name: &str) -> Option<Value> {
        None
    }
    fn dummy_snapshot() -> Vec<(String, Value)> {
        Vec::new()
    }
    fn dummy_globals() -> Vec<String> {
        Vec::new()
    }

    #[test]
    fn register_unregister_clear_resolver() {
        register_workspace_resolver(WorkspaceResolver {
            lookup: dummy_lookup,
            snapshot: dummy_snapshot,
            globals: dummy_globals,
        });
        assert!(is_available());
        unregister_workspace_resolver();
        assert!(!is_available());
    }
}
