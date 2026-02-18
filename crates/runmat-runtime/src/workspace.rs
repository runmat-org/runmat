use runmat_builtins::Value;

#[cfg(test)]
use once_cell::sync::Lazy;
#[cfg(test)]
use std::sync::Mutex;

/// Resolver used by the runtime to access the caller workspace when builtins
/// (such as `save`) need to look up variables by name.
type AssignFn = fn(&str, Value) -> Result<(), String>;

pub struct WorkspaceResolver {
    pub lookup: fn(&str) -> Option<Value>,
    pub snapshot: fn() -> Vec<(String, Value)>,
    pub globals: fn() -> Vec<String>,
    pub assign: Option<AssignFn>,
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

/// Register the workspace resolver. Ignition installs this once during
/// initialization so that language builtins can query variables lazily.
pub fn register_workspace_resolver(resolver: WorkspaceResolver) {
    resolver_storage::set(resolver);
}

/// Lookup a variable by name in the active workspace.
pub fn lookup(name: &str) -> Option<Value> {
    resolver_storage::with(|resolver| resolver.and_then(|r| (r.lookup)(name)))
}

/// Snapshot the active workspace into a vector of `(name, value)` pairs.
/// Returns `None` when no resolver/workspace is active.
pub fn snapshot() -> Option<Vec<(String, Value)>> {
    resolver_storage::with(|resolver| resolver.map(|r| (r.snapshot)()))
}

/// Return the list of global variable names visible to the active workspace.
pub fn global_names() -> Vec<String> {
    resolver_storage::with(|resolver| resolver.map(|r| (r.globals)()).unwrap_or_default())
}

pub fn assign(name: &str, value: Value) -> Result<(), String> {
    resolver_storage::with(|resolver| {
        let resolver = resolver.ok_or_else(|| "workspace state unavailable".to_string())?;
        let assign = resolver
            .assign
            .ok_or_else(|| "workspace assignment unavailable".to_string())?;
        (assign)(name, value)
    })
}

/// Returns true when a resolver has been registered.
pub fn is_available() -> bool {
    resolver_storage::with(|resolver| resolver.is_some())
}

#[cfg(test)]
pub(crate) fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    TEST_WORKSPACE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}
