//! Registry for loaded native libraries.

use crate::library::NativeLibrary;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

/// Global registry of loaded libraries.
static LIBRARY_REGISTRY: Lazy<Mutex<LibraryRegistry>> =
    Lazy::new(|| Mutex::new(LibraryRegistry::new()));

/// Registry for managing loaded native libraries.
pub struct LibraryRegistry {
    libraries: HashMap<String, NativeLibrary>,
}

impl LibraryRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            libraries: HashMap::new(),
        }
    }

    /// Load a library and register it by name.
    pub fn load(&mut self, name: &str) -> Result<(), String> {
        if self.libraries.contains_key(name) {
            return Ok(()); // Already loaded
        }

        let library = NativeLibrary::load_by_name(name)?;
        self.libraries.insert(name.to_string(), library);
        Ok(())
    }

    /// Get a reference to a loaded library.
    pub fn get(&self, name: &str) -> Option<&NativeLibrary> {
        self.libraries.get(name)
    }

    /// Check if a library is loaded.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.libraries.contains_key(name)
    }

    /// Unload a library.
    pub fn unload(&mut self, name: &str) -> bool {
        self.libraries.remove(name).is_some()
    }

    /// List all loaded libraries.
    pub fn list(&self) -> Vec<&str> {
        self.libraries.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for LibraryRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the global library registry.
pub fn global_registry() -> &'static Mutex<LibraryRegistry> {
    &LIBRARY_REGISTRY
}

/// Load a library into the global registry.
pub fn load_library(name: &str) -> Result<(), String> {
    let mut registry = LIBRARY_REGISTRY
        .lock()
        .map_err(|_| "Failed to acquire library registry lock")?;
    registry.load(name)
}

/// Check if a library is loaded in the global registry.
#[allow(dead_code)]
pub fn is_library_loaded(name: &str) -> Result<bool, String> {
    let registry = LIBRARY_REGISTRY
        .lock()
        .map_err(|_| "Failed to acquire library registry lock")?;
    Ok(registry.is_loaded(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_basic() {
        let registry = LibraryRegistry::new();
        assert!(!registry.is_loaded("nonexistent"));
        assert!(registry.list().is_empty());
    }
}
