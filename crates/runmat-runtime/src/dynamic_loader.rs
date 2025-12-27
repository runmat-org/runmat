//! Dynamic loading of `.m` files from the MATLAB search path.
//!
//! This module provides the `load_function_from_path` function which searches
//! for a `.m` file matching a function name, parses it, and returns the
//! compiled function definition for use by the VM.

use crate::builtins::common::path_search::find_file_with_extensions;
use std::fs;
use std::path::PathBuf;

/// Result of loading a function from disk.
#[derive(Debug, Clone)]
pub struct LoadedFunction {
    /// The path to the `.m` file that was loaded.
    pub path: PathBuf,
    /// The source code of the file.
    pub source: String,
    /// The name of the function (may differ from filename for scripts).
    pub name: String,
}

/// Search for a `.m` file matching the function name on the search path.
///
/// Returns the path to the file and its contents if found.
pub fn find_function_file(name: &str) -> Result<Option<LoadedFunction>, String> {
    // First check current working directory
    let cwd = std::env::current_dir().map_err(|e| format!("dynamic_loader: {e}"))?;
    let cwd_candidate = cwd.join(format!("{name}.m"));
    if cwd_candidate.is_file() {
        let source =
            fs::read_to_string(&cwd_candidate).map_err(|e| format!("dynamic_loader: {e}"))?;
        return Ok(Some(LoadedFunction {
            path: cwd_candidate,
            source,
            name: name.to_string(),
        }));
    }

    // Then search the MATLAB path
    let extensions = &[".m"];
    match find_file_with_extensions(name, extensions, "dynamic_loader")? {
        Some(path) => {
            let source = fs::read_to_string(&path).map_err(|e| format!("dynamic_loader: {e}"))?;
            Ok(Some(LoadedFunction {
                path,
                source,
                name: name.to_string(),
            }))
        }
        None => Ok(None),
    }
}

/// Load a function from the search path, parse it, and return parsed HIR.
///
/// This is the main entry point for dynamic function loading.
/// Returns `None` if the function is not found on the path.
/// Returns `Err` if the file is found but cannot be parsed.
pub fn load_and_parse_function(
    name: &str,
) -> Result<Option<(LoadedFunction, runmat_hir::HirProgram)>, String> {
    let loaded = match find_function_file(name)? {
        Some(l) => l,
        None => return Ok(None),
    };

    // Parse the source
    let ast = runmat_parser::parse(&loaded.source)
        .map_err(|e| format!("dynamic_loader: failed to parse {}: {e}", loaded.path.display()))?;

    // Lower to HIR
    let hir = runmat_hir::lower(&ast)
        .map_err(|e| format!("dynamic_loader: failed to lower {}: {e}", loaded.path.display()))?;

    Ok(Some((loaded, hir)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use tempfile::tempdir;

    #[test]
    fn test_find_function_in_cwd() {
        let dir = tempdir().expect("tempdir");
        let prev_cwd = env::current_dir().expect("cwd");

        // Create a test function file
        let func_path = dir.path().join("test_func.m");
        fs::write(&func_path, "function y = test_func(x)\n    y = x * 2;\nend").expect("write");

        // Change to temp directory
        env::set_current_dir(dir.path()).expect("chdir");

        let result = find_function_file("test_func");
        env::set_current_dir(&prev_cwd).expect("restore cwd");

        assert!(result.is_ok());
        let loaded = result.unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.name, "test_func");
        assert!(loaded.source.contains("function y = test_func"));
    }
}
