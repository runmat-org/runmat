use runmat_value::Value;

pub mod session;

#[cfg(test)]
use once_cell::sync::Lazy;
#[cfg(test)]
use std::sync::Mutex;

/// Resolver used by the runtime to access the caller workspace when builtins
/// (such as `save`) need to look up variables by name.
type AssignFn = fn(&str, Value) -> Result<(), String>;
type ClearFn = fn() -> Result<(), String>;
type RemoveFn = fn(&str) -> Result<(), String>;

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
        return;
    }
    resolver_storage::set(resolver);
}

/// Lookup a variable by name in the active workspace.
pub fn lookup(name: &str) -> Option<Value> {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service.lookup(name);
        }
        return context
            .state()
            .workspace
            .borrow()
            .as_ref()
            .and_then(|resolver| (resolver.lookup)(name));
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
        return context
            .state()
            .workspace
            .borrow()
            .as_ref()
            .map(|resolver| (resolver.snapshot)());
    }
    resolver_storage::with(|resolver| resolver.map(|r| (r.snapshot)()))
}

/// Return the list of global variable names visible to the active workspace.
pub fn global_names() -> Vec<String> {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service.global_names();
        }
        return context
            .state()
            .workspace
            .borrow()
            .as_ref()
            .map(|resolver| (resolver.globals)())
            .unwrap_or_default();
    }
    resolver_storage::with(|resolver| resolver.map(|r| (r.globals)()).unwrap_or_default())
}

pub fn assign(name: &str, value: Value) -> Result<(), String> {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service
                .assign(name, value)
                .map_err(|error| error.to_string());
        }
        return with_context_resolver(&context, |resolver| {
            let assign = resolver
                .assign
                .ok_or_else(|| "workspace assignment unavailable".to_string())?;
            assign(name, value)
        });
    }
    resolver_storage::with(|resolver| {
        let resolver = resolver.ok_or_else(|| "workspace state unavailable".to_string())?;
        let assign = resolver
            .assign
            .ok_or_else(|| "workspace assignment unavailable".to_string())?;
        (assign)(name, value)
    })
}

pub fn clear() -> Result<(), String> {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service.clear().map_err(|error| error.to_string());
        }
        return with_context_resolver(&context, |resolver| {
            let clear = resolver
                .clear
                .ok_or_else(|| "workspace clearing unavailable".to_string())?;
            clear()
        });
    }
    resolver_storage::with(|resolver| {
        let resolver = resolver.ok_or_else(|| "workspace state unavailable".to_string())?;
        let clear = resolver
            .clear
            .ok_or_else(|| "workspace clearing unavailable".to_string())?;
        (clear)()
    })
}

pub fn remove(name: &str) -> Result<(), String> {
    if let Some(context) = crate::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service.remove(name).map_err(|error| error.to_string());
        }
        return with_context_resolver(&context, |resolver| {
            let remove = resolver
                .remove
                .ok_or_else(|| "workspace removal unavailable".to_string())?;
            remove(name)
        });
    }
    resolver_storage::with(|resolver| {
        let resolver = resolver.ok_or_else(|| "workspace state unavailable".to_string())?;
        let remove = resolver
            .remove
            .ok_or_else(|| "workspace removal unavailable".to_string())?;
        (remove)(name)
    })
}

/// Returns true when a resolver has been registered.
pub fn is_available() -> bool {
    if let Some(context) = crate::context::legacy::active() {
        return context.service_ports().workspace().is_some()
            || context.state().workspace.borrow().is_some();
    }
    resolver_storage::with(|resolver| resolver.is_some())
}

fn with_context_resolver<R>(
    context: &crate::context::RuntimeContext,
    callback: impl FnOnce(&WorkspaceResolver) -> Result<R, String>,
) -> Result<R, String> {
    let resolver = context.state().workspace.borrow();
    callback(
        resolver
            .as_ref()
            .ok_or_else(|| "workspace state unavailable".to_string())?,
    )
}

#[cfg(test)]
pub(crate) fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    TEST_WORKSPACE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}
