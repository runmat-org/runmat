//! The full design check of an imported project, shared by the import tests: the circuit's
//! check (elaboration and ERC, which a compile alone doesn't report) and its layout's check.

use runmat_circuit::runtime::{check, CompileOptions, Files};

/// Error messages from checking `circuit` and, when given, `layout`.
#[allow(dead_code)]
pub fn design_errors(files: &Files, circuit: &str, layout: Option<&str>) -> Vec<String> {
    let mut errors = Vec::new();
    for path in std::iter::once(circuit).chain(layout) {
        let out = check(path, &files[path], files, &CompileOptions::default());
        errors.extend(
            out.diagnostics
                .iter()
                .filter(|d| d.severity == "error")
                .map(|d| format!("{path}: {}", d.message)),
        );
    }
    errors
}
