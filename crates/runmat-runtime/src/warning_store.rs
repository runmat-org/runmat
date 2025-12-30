use once_cell::sync::Lazy;
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct RuntimeWarning {
    pub identifier: String,
    pub message: String,
}

static WARNINGS: Lazy<Mutex<Vec<RuntimeWarning>>> = Lazy::new(|| Mutex::new(Vec::new()));

pub fn push(identifier: &str, message: &str) {
    if let Ok(mut guard) = WARNINGS.lock() {
        guard.push(RuntimeWarning {
            identifier: identifier.to_string(),
            message: message.to_string(),
        });
    }
}

pub fn take_all() -> Vec<RuntimeWarning> {
    WARNINGS
        .lock()
        .map(|mut guard| guard.drain(..).collect())
        .unwrap_or_default()
}

pub fn reset() {
    if let Ok(mut guard) = WARNINGS.lock() {
        guard.clear();
    }
}
