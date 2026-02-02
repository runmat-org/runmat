use std::io;
use std::path::PathBuf;
#[cfg(target_arch = "wasm32")]
use std::collections::BTreeMap;
#[cfg(target_arch = "wasm32")]
use std::sync::RwLock;

#[cfg(target_arch = "wasm32")]
use once_cell::sync::OnceCell;

#[cfg(target_arch = "wasm32")]
fn env_lock() -> &'static RwLock<BTreeMap<String, String>> {
    static ENV: OnceCell<RwLock<BTreeMap<String, String>>> = OnceCell::new();
    ENV.get_or_init(|| {
        let mut vars = BTreeMap::new();
        vars.insert("HOME".to_string(), "/home/user".to_string());
        vars.insert(
            "RUNMAT_PATH".to_string(),
            "/home/user/runmat/toolbox:/home/user/runmat/user".to_string(),
        );
        vars.insert(
            "PATH".to_string(),
            "/usr/local/bin:/usr/bin:/bin".to_string(),
        );
        vars.insert("USER".to_string(), "user".to_string());
        vars.insert("SHELL".to_string(), "/bin/sh".to_string());
        vars.insert("TMPDIR".to_string(), "/tmp".to_string());
        RwLock::new(vars)
    })
}

pub fn var(key: &str) -> io::Result<String> {
    #[cfg(target_arch = "wasm32")]
    {
        let guard = env_lock().read().expect("env lock poisoned");
        if let Some(value) = guard.get(key) {
            return Ok(value.clone());
        }
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Environment variable not found: {key}"),
        ));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var(key).map_err(|err| io::Error::new(io::ErrorKind::NotFound, err))
    }
}

pub fn vars() -> Vec<(String, String)> {
    #[cfg(target_arch = "wasm32")]
    {
        let guard = env_lock().read().expect("env lock poisoned");
        return guard.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::vars().collect()
    }
}

pub fn set_var(key: &str, value: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        let mut guard = env_lock().write().expect("env lock poisoned");
        guard.insert(key.to_string(), value.to_string());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::set_var(key, value);
    }
}

pub fn remove_var(key: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        let mut guard = env_lock().write().expect("env lock poisoned");
        guard.remove(key);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::remove_var(key);
    }
}

pub fn temp_dir() -> PathBuf {
    #[cfg(target_arch = "wasm32")]
    {
        PathBuf::from("/tmp")
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::temp_dir()
    }
}
