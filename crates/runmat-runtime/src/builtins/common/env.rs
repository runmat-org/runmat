#[cfg(target_arch = "wasm32")]
use std::collections::BTreeMap;
use std::io;
use std::path::PathBuf;
#[cfg(target_arch = "wasm32")]
use std::sync::RwLock;

#[cfg(target_arch = "wasm32")]
use once_cell::sync::OnceCell;

#[cfg(target_arch = "wasm32")]
fn env_lock() -> &'static RwLock<BTreeMap<String, String>> {
    static ENV: OnceCell<RwLock<BTreeMap<String, String>>> = OnceCell::new();
    ENV.get_or_init(|| RwLock::new(BTreeMap::new()))
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
