use once_cell::sync::OnceCell;
use runmat_builtins::Value;

/// Resolver used by the runtime to access the caller workspace when builtins
/// (such as `save`) need to look up variables by name.
pub struct WorkspaceResolver {
    pub lookup: fn(&str) -> Option<Value>,
    pub snapshot: fn() -> Vec<(String, Value)>,
}

static RESOLVER: OnceCell<WorkspaceResolver> = OnceCell::new();

/// Register the workspace resolver. Ignition installs this once during
/// initialization so that language builtins can query variables lazily.
pub fn register_workspace_resolver(resolver: WorkspaceResolver) {
    let _ = RESOLVER.set(resolver);
}

/// Lookup a variable by name in the active workspace.
pub fn lookup(name: &str) -> Option<Value> {
    RESOLVER.get().and_then(|resolver| (resolver.lookup)(name))
}

/// Snapshot the active workspace into a vector of `(name, value)` pairs.
/// Returns `None` when no resolver/workspace is active.
pub fn snapshot() -> Option<Vec<(String, Value)>> {
    RESOLVER.get().map(|resolver| (resolver.snapshot)())
}

/// Returns true when a resolver has been registered.
pub fn is_available() -> bool {
    RESOLVER.get().is_some()
}
