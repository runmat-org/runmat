//! Registry for loaded native libraries.

use crate::library::NativeLibrary;
use crate::parser::SignatureFile;
use crate::types::FfiSignature;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// Global registry of loaded libraries.
static LIBRARY_REGISTRY: Lazy<Mutex<LibraryRegistry>> =
    Lazy::new(|| Mutex::new(LibraryRegistry::new()));

/// Entry for a loaded library with optional signatures.
pub struct LibraryEntry {
    /// The native library handle
    pub library: NativeLibrary,
    /// Function signatures (if loaded from .ffi file)
    pub signatures: Option<SignatureFile>,
}

/// Registry for managing loaded native libraries.
pub struct LibraryRegistry {
    libraries: HashMap<String, LibraryEntry>,
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
        self.libraries.insert(
            name.to_string(),
            LibraryEntry {
                library,
                signatures: None,
            },
        );
        Ok(())
    }

    /// Load a library with a signature file.
    pub fn load_with_signatures(
        &mut self,
        name: &str,
        sig_path: impl AsRef<Path>,
    ) -> Result<(), String> {
        // Parse signature file first
        let signatures = SignatureFile::parse_file(sig_path.as_ref())
            .map_err(|e| format!("Failed to parse signature file: {}", e))?;

        // Load library if not already loaded
        let library = if self.libraries.contains_key(name) {
            // Library already loaded, just update signatures
            self.libraries.get_mut(name).unwrap().signatures = Some(signatures);
            return Ok(());
        } else {
            NativeLibrary::load_by_name(name)?
        };

        self.libraries.insert(
            name.to_string(),
            LibraryEntry {
                library,
                signatures: Some(signatures),
            },
        );
        Ok(())
    }

    /// Get a reference to a loaded library.
    pub fn get(&self, name: &str) -> Option<&NativeLibrary> {
        self.libraries.get(name).map(|e| &e.library)
    }

    /// Get a reference to a library entry (includes signatures).
    pub fn get_entry(&self, name: &str) -> Option<&LibraryEntry> {
        self.libraries.get(name)
    }

    /// Get a function signature if available.
    pub fn get_signature(&self, lib_name: &str, func_name: &str) -> Option<&FfiSignature> {
        self.libraries
            .get(lib_name)
            .and_then(|e| e.signatures.as_ref())
            .and_then(|s| s.get(func_name))
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

/// Load a library with signatures into the global registry.
pub fn load_library_with_signatures(name: &str, sig_path: impl AsRef<Path>) -> Result<(), String> {
    let mut registry = LIBRARY_REGISTRY
        .lock()
        .map_err(|_| "Failed to acquire library registry lock")?;
    registry.load_with_signatures(name, sig_path)
}

/// Get a function signature from the global registry.
pub fn get_function_signature(lib_name: &str, func_name: &str) -> Result<Option<FfiSignature>, String> {
    let registry = LIBRARY_REGISTRY
        .lock()
        .map_err(|_| "Failed to acquire library registry lock")?;
    Ok(registry.get_signature(lib_name, func_name).cloned())
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
