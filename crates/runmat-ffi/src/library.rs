//! Native library loading and function lookup.

use libloading::{Library, Symbol};
use std::ffi::CString;
use std::path::Path;

/// A loaded native library.
pub struct NativeLibrary {
    /// The underlying library handle
    library: Library,
    /// Path to the library (for debugging)
    path: String,
}

impl NativeLibrary {
    /// Load a native library from a path.
    ///
    /// On Windows, looks for `.dll` files.
    /// On macOS, looks for `.dylib` files.
    /// On Linux, looks for `.so` files.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();

        // Try the exact path first
        let library = unsafe { Library::new(path) }.map_err(|e| {
            format!(
                "Failed to load library '{}': {}",
                path.display(),
                e
            )
        })?;

        Ok(Self {
            library,
            path: path.display().to_string(),
        })
    }

    /// Load a library by name, searching standard paths.
    ///
    /// The name should be the base name without extension (e.g., "mylib").
    pub fn load_by_name(name: &str) -> Result<Self, String> {
        // Construct platform-specific library name
        let lib_name = Self::platform_lib_name(name);

        // Try loading from current directory first
        if let Ok(lib) = Self::load(&lib_name) {
            return Ok(lib);
        }

        // Try loading from system paths
        let library = unsafe { Library::new(&lib_name) }.map_err(|e| {
            format!(
                "Failed to load library '{}' (tried '{}'): {}",
                name, lib_name, e
            )
        })?;

        Ok(Self {
            library,
            path: lib_name,
        })
    }

    /// Get the platform-specific library filename.
    fn platform_lib_name(name: &str) -> String {
        #[cfg(target_os = "windows")]
        {
            format!("{}.dll", name)
        }
        #[cfg(target_os = "macos")]
        {
            format!("lib{}.dylib", name)
        }
        #[cfg(target_os = "linux")]
        {
            format!("lib{}.so", name)
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            format!("lib{}.so", name)
        }
    }

    /// Get a function pointer from the library.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The function exists in the library
    /// - The type `F` matches the actual function signature
    pub unsafe fn get_function<F>(&self, name: &str) -> Result<Symbol<'_, F>, String> {
        let c_name =
            CString::new(name).map_err(|_| format!("Invalid function name: {}", name))?;

        self.library
            .get(c_name.as_bytes_with_nul())
            .map_err(|e| format!("Function '{}' not found in '{}': {}", name, self.path, e))
    }

    /// Get the path of this library.
    pub fn path(&self) -> &str {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_lib_name() {
        let name = NativeLibrary::platform_lib_name("test");
        #[cfg(target_os = "windows")]
        assert_eq!(name, "test.dll");
        #[cfg(target_os = "macos")]
        assert_eq!(name, "libtest.dylib");
        #[cfg(target_os = "linux")]
        assert_eq!(name, "libtest.so");
    }
}
