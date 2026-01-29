use once_cell::sync::Lazy;
use runmat_builtins::Value;
use std::sync::RwLock;

#[cfg(test)]
use std::sync::Mutex;

/// Resolver used by the runtime to access the caller workspace when builtins
/// (such as `save`) need to look up variables by name.
pub struct WorkspaceResolver {
    pub lookup: fn(&str) -> Option<Value>,
    pub snapshot: fn() -> Vec<(String, Value)>,
    pub globals: fn() -> Vec<String>,
}

static RESOLVER: Lazy<RwLock<Option<WorkspaceResolver>>> = Lazy::new(|| RwLock::new(None));

#[cfg(test)]
static TEST_WORKSPACE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

/// Register the workspace resolver. Ignition installs this once during
/// initialization so that language builtins can query variables lazily.
pub fn register_workspace_resolver(resolver: WorkspaceResolver) {
    let mut guard = RESOLVER
        .write()
        .unwrap_or_else(|poison| poison.into_inner());
    *guard = Some(resolver);
}

/// Lookup a variable by name in the active workspace.
pub fn lookup(name: &str) -> Option<Value> {
    let guard = RESOLVER.read().unwrap_or_else(|poison| poison.into_inner());
    guard.as_ref().and_then(|resolver| (resolver.lookup)(name))
}

/// Snapshot the active workspace into a vector of `(name, value)` pairs.
/// Returns `None` when no resolver/workspace is active.
pub fn snapshot() -> Option<Vec<(String, Value)>> {
    let guard = RESOLVER.read().unwrap_or_else(|poison| poison.into_inner());
    guard.as_ref().map(|resolver| (resolver.snapshot)())
}

/// Return the list of global variable names visible to the active workspace.
pub fn global_names() -> Vec<String> {
    let guard = RESOLVER.read().unwrap_or_else(|poison| poison.into_inner());
    guard
        .as_ref()
        .map(|resolver| (resolver.globals)())
        .unwrap_or_default()
}

/// Returns true when a resolver has been registered.
pub fn is_available() -> bool {
    RESOLVER
        .read()
        .unwrap_or_else(|poison| poison.into_inner())
        .is_some()
}

#[cfg(test)]
pub(crate) fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    TEST_WORKSPACE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}
