use super::super::schema::InventoryValidationError;

const RUNTIME_CRATE_SOURCE: &str = "crates/runmat-runtime/";

pub(super) fn canonical_compiler_source_path(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    if let Some(offset) = normalized.find(RUNTIME_CRATE_SOURCE) {
        return normalized[offset..].to_owned();
    }
    if normalized.starts_with("src/") {
        return format!("{RUNTIME_CRATE_SOURCE}{normalized}");
    }
    normalized
}

pub(super) fn validate_registration_provenance(
    errors: &mut Vec<InventoryValidationError>,
    source: &'static str,
    identity: &str,
    source_file: &str,
    module_path: &str,
    builtin_path: &str,
    function: Option<&str>,
) {
    if !valid_repository_source_path(source_file) {
        push(
            errors,
            source,
            identity,
            "source file is not a canonical runtime repository path",
        );
    }
    if !valid_rust_path(module_path) {
        push(
            errors,
            source,
            identity,
            "compiler module path is not a valid Rust module path",
        );
    }
    if !valid_rust_path(builtin_path) {
        push(
            errors,
            source,
            identity,
            "declared builtin path is not a valid Rust module path",
        );
    }
    if valid_rust_path(module_path)
        && valid_rust_path(builtin_path)
        && !module_is_within_scope(module_path, builtin_path)
    {
        push(
            errors,
            source,
            identity,
            "declared builtin path differs from the compiler module path",
        );
    }
    if function.is_some_and(|name| !valid_rust_identifier(name)) {
        push(
            errors,
            source,
            identity,
            "implementation function is not a valid Rust identifier",
        );
    }
}

pub(super) fn validate_spec_path(
    errors: &mut Vec<InventoryValidationError>,
    source: &'static str,
    identity: &str,
    builtin_path: &str,
) {
    if !valid_rust_path(builtin_path) {
        push(
            errors,
            source,
            identity,
            "spec builtin path is not a valid Rust module path",
        );
    }
}

fn valid_repository_source_path(path: &str) -> bool {
    path.starts_with(&format!("{RUNTIME_CRATE_SOURCE}src/"))
        && path.ends_with(".rs")
        && !path.contains("//")
        && !path
            .split('/')
            .any(|component| component == "." || component == ".." || component.is_empty())
}

fn canonical_module_path(path: &str) -> &str {
    path.strip_prefix("crate::")
        .or_else(|| path.strip_prefix("runmat_runtime::"))
        .unwrap_or(path)
}

pub(super) fn module_scopes_overlap(left: &str, right: &str) -> bool {
    module_is_within_scope(left, right) || module_is_within_scope(right, left)
}

fn module_is_within_scope(module_path: &str, scope: &str) -> bool {
    let module_path = canonical_module_path(module_path);
    let scope = canonical_module_path(scope);
    module_path == scope
        || module_path
            .strip_prefix(scope)
            .is_some_and(|suffix| suffix.starts_with("::"))
}

fn valid_rust_path(path: &str) -> bool {
    let path = canonical_module_path(path);
    !path.is_empty() && path.split("::").all(valid_rust_identifier)
}

fn valid_rust_identifier(value: &str) -> bool {
    let value = value.strip_prefix("r#").unwrap_or(value);
    let mut characters = value.chars();
    matches!(characters.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

fn push(
    errors: &mut Vec<InventoryValidationError>,
    source: &'static str,
    identity: &str,
    message: &'static str,
) {
    errors.push(InventoryValidationError {
        source,
        identity: Some(identity.to_owned()),
        message: message.to_owned(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiler_source_paths_are_canonical_across_host_separators() {
        assert_eq!(
            canonical_compiler_source_path(
                r"C:\work\runmat\crates\runmat-runtime\src\builtins\foo.rs"
            ),
            "crates/runmat-runtime/src/builtins/foo.rs"
        );
        assert_eq!(
            canonical_compiler_source_path("src/builtins/foo.rs"),
            "crates/runmat-runtime/src/builtins/foo.rs"
        );
    }

    #[test]
    fn declaration_paths_must_agree_after_crate_normalization() {
        let mut errors = Vec::new();
        validate_registration_provenance(
            &mut errors,
            "implementation_provenance",
            "foo",
            "crates/runmat-runtime/src/builtins/foo.rs",
            "runmat_runtime::builtins::foo",
            "crate::builtins::other",
            Some("foo_builtin"),
        );
        assert!(errors.iter().any(|error| error.message.contains("differs")));
    }

    #[test]
    fn declaration_modules_may_live_below_the_builtin_scope() {
        let mut errors = Vec::new();
        validate_registration_provenance(
            &mut errors,
            "implementation_provenance",
            "foo",
            "crates/runmat-runtime/src/builtins/foo/implementation.rs",
            "runmat_runtime::builtins::foo::implementation",
            "crate::builtins::foo",
            Some("foo_builtin"),
        );
        assert!(errors.is_empty());
    }
}
