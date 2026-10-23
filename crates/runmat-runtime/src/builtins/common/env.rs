#[cfg(target_arch = "wasm32")]
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
#[cfg(target_arch = "wasm32")]
use std::sync::RwLock;

#[cfg(target_arch = "wasm32")]
use once_cell::sync::OnceCell;

fn require_selected(effect: runmat_types::EffectKind) -> crate::BuiltinResult<()> {
    if let Some(context) = crate::context::legacy::active() {
        context
            .service_ports()
            .require_selected_local_effect(effect)?;
    }
    Ok(())
}

pub(crate) fn require_read() -> crate::BuiltinResult<()> {
    require_selected(runmat_types::EffectKind::EnvironmentRead)
}

pub(crate) fn require_write() -> crate::BuiltinResult<()> {
    require_selected(runmat_types::EffectKind::EnvironmentWrite)
}

fn require_effect_io(effect: runmat_types::EffectKind) -> io::Result<()> {
    if let Some(context) = crate::context::legacy::active() {
        context
            .service_ports()
            .check_selected_local_effect(effect)
            .map_err(|source| io::Error::new(io::ErrorKind::PermissionDenied, source))?;
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn default_env_map() -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    // Match the documented baseline process environment in wasm sessions.
    env.insert("HOME".to_string(), "/".to_string());
    env.insert("PATH".to_string(), "/".to_string());
    env.insert("USER".to_string(), "user".to_string());
    env
}

#[cfg(target_arch = "wasm32")]
fn env_lock() -> &'static RwLock<BTreeMap<String, String>> {
    static ENV: OnceCell<RwLock<BTreeMap<String, String>>> = OnceCell::new();
    ENV.get_or_init(|| RwLock::new(default_env_map()))
}

pub fn var(key: &str) -> io::Result<String> {
    require_effect_io(runmat_types::EffectKind::EnvironmentRead)?;
    var_unchecked(key)
}

pub(crate) fn optional_var(key: &str) -> crate::BuiltinResult<Option<String>> {
    require_read()?;
    match var_unchecked(key) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(crate::runtime_error::build_runtime_error(format!(
            "failed to read environment variable '{key}': {error}"
        ))
        .with_source(error)
        .build()),
    }
}

fn var_unchecked(key: &str) -> io::Result<String> {
    if let Some(context) = crate::context::legacy::active() {
        return context.environment_state().borrow().var(key);
    }
    #[cfg(target_arch = "wasm32")]
    {
        let guard = env_lock().read().expect("env lock poisoned");
        if let Some(value) = guard.get(key) {
            return Ok(value.clone());
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Environment variable not found: {key}"),
        ))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var(key).map_err(|err| io::Error::new(io::ErrorKind::NotFound, err))
    }
}

pub fn try_vars() -> io::Result<Vec<(String, String)>> {
    require_effect_io(runmat_types::EffectKind::EnvironmentRead)?;
    if let Some(context) = crate::context::legacy::active() {
        return Ok(context.environment_state().borrow().vars());
    }
    #[cfg(target_arch = "wasm32")]
    {
        let guard = env_lock().read().expect("env lock poisoned");
        Ok(guard.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Ok(std::env::vars().collect())
    }
}

#[cfg(test)]
pub fn vars() -> Vec<(String, String)> {
    try_vars().expect("test environment read must be admitted")
}

pub fn try_set_var(key: &str, value: &str) -> io::Result<()> {
    require_effect_io(runmat_types::EffectKind::EnvironmentWrite)?;
    assert!(
        !key.is_empty() && !key.contains(['=', '\0']) && !value.contains('\0'),
        "invalid environment variable"
    );
    if let Some(context) = crate::context::legacy::active() {
        context.environment_state().borrow_mut().set_var(key, value);
        return Ok(());
    }
    #[cfg(target_arch = "wasm32")]
    {
        let mut guard = env_lock().write().expect("env lock poisoned");
        guard.insert(key.to_string(), value.to_string());
        Ok(())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::set_var(key, value);
        Ok(())
    }
}

pub fn try_remove_var(key: &str) -> io::Result<()> {
    require_effect_io(runmat_types::EffectKind::EnvironmentWrite)?;
    assert!(
        !key.is_empty() && !key.contains(['=', '\0']),
        "invalid environment variable"
    );
    if let Some(context) = crate::context::legacy::active() {
        context.environment_state().borrow_mut().remove_var(key);
        return Ok(());
    }
    #[cfg(target_arch = "wasm32")]
    {
        let mut guard = env_lock().write().expect("env lock poisoned");
        guard.remove(key);
        Ok(())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::remove_var(key);
        Ok(())
    }
}

#[cfg(test)]
pub fn set_var(key: &str, value: &str) {
    try_set_var(key, value).expect("test environment write must be admitted");
}

#[cfg(test)]
pub fn remove_var(key: &str) {
    try_remove_var(key).expect("test environment write must be admitted");
}

pub fn temp_dir() -> crate::BuiltinResult<PathBuf> {
    if let Some(context) = crate::context::legacy::active() {
        if context.service_ports().has_selected_local_services() {
            require_read()?;
            let fallback = context
                .service_ports()
                .selected_temporary_directory()
                .ok_or_else(|| {
                    crate::runtime_error::build_runtime_error(
                        "selected environment has no admitted temporary directory",
                    )
                    .with_identifier("RunMat:Native:SelectedLocalServiceMismatch")
                    .build()
                })?;
            return Ok(selected_temp_dir(&context, &fallback));
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        Ok(PathBuf::from("/tmp"))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Ok(std::env::temp_dir())
    }
}

fn selected_temp_dir(context: &crate::context::RuntimeContext, fallback: &Path) -> PathBuf {
    let environment = context.environment_state().borrow();
    if let Some(path) = ["TMPDIR", "TEMP", "TMP"]
        .into_iter()
        .filter_map(|key| environment.var(key).ok())
        .find(|path| !path.is_empty())
    {
        return PathBuf::from(path);
    }
    fallback.to_path_buf()
}

/// Native subprocesses inherit the active session snapshot, including local changes.
/// Standalone compatibility calls retain their existing host-process behavior.
#[cfg(not(target_arch = "wasm32"))]
pub fn apply_to_command(command: &mut std::process::Command) {
    if let Some(context) = crate::context::legacy::active() {
        command.env_clear().envs(context.environment_snapshot());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context::RuntimeContext, execution::RuntimeExecutionService};
    use std::rc::Rc;

    #[test]
    fn deterministic_temp_directory_uses_only_the_selected_snapshot() {
        let explicit = RuntimeContext::with_frozen_environment(
            Rc::new(RuntimeExecutionService::new()),
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            vec![("TMPDIR".into(), "/declared-temp".into())],
        )
        .unwrap();
        assert_eq!(
            selected_temp_dir(&explicit, Path::new("/fallback")),
            PathBuf::from("/declared-temp")
        );
        let empty = RuntimeContext::with_frozen_environment(
            Rc::new(RuntimeExecutionService::new()),
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            selected_temp_dir(&empty, Path::new("/fallback")),
            PathBuf::from("/fallback")
        );
        let skips_empty = RuntimeContext::with_frozen_environment(
            Rc::new(RuntimeExecutionService::new()),
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            vec![
                ("TEMP".into(), "/second".into()),
                ("TMPDIR".into(), String::new()),
            ],
        )
        .unwrap();
        assert_eq!(
            selected_temp_dir(&skips_empty, Path::new("/fallback")),
            PathBuf::from("/second")
        );
    }
}
