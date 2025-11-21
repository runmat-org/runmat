use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::hash::Hash;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct AutotuneController<K, V> {
    enabled: bool,
    cache: Mutex<HashMap<K, V>>,
    json_path: Option<PathBuf>,
}

impl<K, V> AutotuneController<K, V>
where
    K: Eq + Hash + Clone + Serialize + DeserializeOwned,
    V: Copy + Serialize + DeserializeOwned + PartialEq,
{
    pub fn new_from_env(
        var: &str,
        kernel: &str,
        base_dir: Option<PathBuf>,
        device_tag: &str,
    ) -> Self {
        let enabled = std::env::var(var)
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        let json_path = if enabled {
            base_dir.map(|mut dir| {
                dir.push("autotune");
                dir.push(kernel);
                dir.push(format!("{device_tag}.json"));
                dir
            })
        } else {
            None
        };
        let controller = Self {
            enabled,
            cache: Mutex::new(HashMap::new()),
            json_path,
        };
        if controller.enabled {
            controller.load_from_disk().ok();
        }
        controller
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.cache.lock().ok()?.get(key).copied()
    }

    pub fn insert(&self, key: K, value: V) {
        if let Ok(mut guard) = self.cache.lock() {
            let needs_flush = !matches!(guard.get(&key), Some(existing) if *existing == value);
            if needs_flush {
                guard.insert(key, value);
                self.save_to_disk(&guard).ok();
            }
        }
    }

    fn load_from_disk(&self) -> std::io::Result<()> {
        let path = match (&self.json_path, self.enabled) {
            (Some(p), true) => p,
            _ => return Ok(()),
        };
        if !path.exists() {
            return Ok(());
        }
        let data = std::fs::read_to_string(path)?;
        let entries: Vec<(K, V)> = serde_json::from_str(&data).unwrap_or_default();
        if let Ok(mut guard) = self.cache.lock() {
            guard.clear();
            for (k, v) in entries {
                guard.insert(k, v);
            }
        }
        Ok(())
    }

    fn save_to_disk(&self, guard: &HashMap<K, V>) -> std::io::Result<()> {
        let path = match (&self.json_path, self.enabled) {
            (Some(p), true) => p,
            _ => return Ok(()),
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let entries: Vec<(&K, &V)> = guard.iter().collect();
        let payload = serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string());
        std::fs::write(path, payload)?;
        log::info!("autotune cache saved to {:?}", path);
        Ok(())
    }
}
