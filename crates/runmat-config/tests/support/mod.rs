use once_cell::sync::Lazy;
use std::sync::{Mutex, MutexGuard};

static ENV_GUARD: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

pub fn env_lock() -> MutexGuard<'static, ()> {
    ENV_GUARD.lock().unwrap()
}

pub fn clear_env(vars: &[&str]) {
    for var in vars {
        std::env::remove_var(var);
    }
}
