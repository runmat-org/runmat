use crate::LogicalSourcePath;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogicalPathError {
    Absolute,
    EscapesRoot,
    Empty,
    PlatformSeparator,
}

impl fmt::Display for LogicalPathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Absolute => formatter.write_str("absolute source paths are not allowed"),
            Self::EscapesRoot => {
                formatter.write_str("source path escapes the declared source root")
            }
            Self::Empty => formatter.write_str("source path resolves to an empty path"),
            Self::PlatformSeparator => {
                formatter.write_str("logical source paths must use `/` separators")
            }
        }
    }
}

impl std::error::Error for LogicalPathError {}

pub fn resolve_logical_path(
    source_root: &LogicalSourcePath,
    referrer: &LogicalSourcePath,
    imported: &str,
) -> Result<LogicalSourcePath, LogicalPathError> {
    if imported.contains('\\') {
        return Err(LogicalPathError::PlatformSeparator);
    }
    if imported.starts_with('/') || imported.starts_with("//") {
        return Err(LogicalPathError::Absolute);
    }

    let root = normalize(source_root.as_str(), &[])?;
    let referrer = normalize(referrer.as_str(), &[])?;
    if !has_prefix(&referrer, &root) {
        return Err(LogicalPathError::EscapesRoot);
    }
    let mut base = referrer;
    base.pop();
    let resolved = normalize(imported, &base)?;
    if !has_prefix(&resolved, &root) {
        return Err(LogicalPathError::EscapesRoot);
    }
    LogicalSourcePath::new(resolved.join("/")).map_err(|_| LogicalPathError::Empty)
}

pub fn logical_path_is_within(source_root: &LogicalSourcePath, path: &LogicalSourcePath) -> bool {
    let Ok(root) = normalize(source_root.as_str(), &[]) else {
        return false;
    };
    let Ok(path) = normalize(path.as_str(), &[]) else {
        return false;
    };
    has_prefix(&path, &root)
}

fn normalize(path: &str, prefix: &[String]) -> Result<Vec<String>, LogicalPathError> {
    let mut parts = prefix.to_vec();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if parts.pop().is_none() {
                    return Err(LogicalPathError::EscapesRoot);
                }
            }
            component => parts.push(component.to_string()),
        }
    }
    if parts.is_empty() && path != "." {
        return Err(LogicalPathError::Empty);
    }
    Ok(parts)
}

fn has_prefix(path: &[String], root: &[String]) -> bool {
    path.len() >= root.len() && path[..root.len()] == *root
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative_paths_inside_the_source_root() {
        let root = LogicalSourcePath::root();
        let source = LogicalSourcePath::new("studies/main.fea").unwrap();
        assert_eq!(
            resolve_logical_path(&root, &source, "../materials/steel.fea")
                .unwrap()
                .as_str(),
            "materials/steel.fea"
        );
    }

    #[test]
    fn rejects_paths_that_escape_the_source_root() {
        let root = LogicalSourcePath::root();
        let source = LogicalSourcePath::new("main.fea").unwrap();
        assert_eq!(
            resolve_logical_path(&root, &source, "../outside.fea"),
            Err(LogicalPathError::EscapesRoot)
        );
    }
}
