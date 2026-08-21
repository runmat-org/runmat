use super::ArchiveError;
use runmat_package::NormalizedRelativePath;
use unicode_normalization::UnicodeNormalization;

pub(crate) fn normalize_entry_path(
    value: &str,
    max_path_bytes: usize,
    max_component_bytes: usize,
) -> Result<NormalizedRelativePath, ArchiveError> {
    if value.as_bytes().contains(&0) {
        return Err(ArchiveError::InvalidPath {
            path: value.to_string(),
            reason: "contains NUL".to_string(),
        });
    }
    let unified = value.replace('\\', "/");
    if unified.starts_with('/') || unified.starts_with("//") || has_windows_drive_prefix(&unified) {
        return invalid(value, "is absolute");
    }
    if unified.len() > max_path_bytes {
        return invalid(value, "exceeds the path length limit");
    }
    let mut components = Vec::new();
    for component in unified.trim_end_matches('/').split('/') {
        validate_component(value, component, max_component_bytes)?;
        components.push(component.nfc().collect::<String>());
    }
    if components.is_empty() {
        return invalid(value, "is empty");
    }
    let normalized = components.join("/");
    if normalized.len() > max_path_bytes {
        return invalid(value, "normalized path exceeds the path length limit");
    }
    NormalizedRelativePath::new(normalized).map_err(|error| ArchiveError::InvalidPath {
        path: value.to_string(),
        reason: error.to_string(),
    })
}

pub(crate) fn normalize_link_target(
    entry: &NormalizedRelativePath,
    target: &str,
    max_path_bytes: usize,
    max_component_bytes: usize,
) -> Result<NormalizedRelativePath, ArchiveError> {
    let unified = target.replace('\\', "/");
    if unified.starts_with('/')
        || unified.starts_with("//")
        || has_windows_drive_prefix(&unified)
        || unified.as_bytes().contains(&0)
        || unified.len() > max_path_bytes
    {
        return invalid(target, "link target is absolute, invalid, or too long");
    }
    let mut components: Vec<String> = entry
        .as_str()
        .rsplit_once('/')
        .map_or_else(Vec::new, |(parent, _)| {
            parent.split('/').map(str::to_string).collect()
        });
    for component in unified.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                if components.pop().is_none() {
                    return invalid(target, "link target escapes the archive root");
                }
            }
            _ => {
                validate_component(target, component, max_component_bytes)?;
                components.push(component.nfc().collect());
            }
        }
    }
    if components.is_empty() {
        return invalid(target, "link target resolves to the archive root");
    }
    let normalized = components.join("/");
    if normalized.len() > max_path_bytes {
        return invalid(target, "resolved link target exceeds the path length limit");
    }
    NormalizedRelativePath::new(normalized).map_err(|error| ArchiveError::InvalidPath {
        path: target.to_string(),
        reason: error.to_string(),
    })
}

fn validate_component(
    full_path: &str,
    component: &str,
    max_component_bytes: usize,
) -> Result<(), ArchiveError> {
    if component.is_empty() || component == "." || component == ".." {
        return invalid(full_path, "contains an empty, dot, or parent component");
    }
    if component.len() > max_component_bytes {
        return invalid(full_path, "contains an overlong component");
    }
    if component.ends_with(['.', ' '])
        || component
            .chars()
            .any(|character| character.is_control() || r#"<>:"|?*"#.contains(character))
    {
        return invalid(full_path, "contains a cross-platform-invalid component");
    }
    let stem = component.split('.').next().unwrap_or(component);
    let upper = stem.to_ascii_uppercase();
    if matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || is_numbered_reserved(&upper, "COM")
        || is_numbered_reserved(&upper, "LPT")
    {
        return invalid(full_path, "contains a Windows-reserved component");
    }
    Ok(())
}

fn has_windows_drive_prefix(value: &str) -> bool {
    value.as_bytes().get(1) == Some(&b':')
        && value
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphabetic)
}

fn is_numbered_reserved(value: &str, prefix: &str) -> bool {
    value
        .strip_prefix(prefix)
        .is_some_and(|suffix| matches!(suffix, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"))
}

fn invalid<T>(path: &str, reason: &str) -> Result<T, ArchiveError> {
    Err(ArchiveError::InvalidPath {
        path: path.to_string(),
        reason: reason.to_string(),
    })
}
