use std::path::{Path, PathBuf};

use crate::project::PROJECT_MANIFEST_FILENAMES;

pub(crate) const USER_CONFIG_FILENAMES: &[&str] = &["config.toml", "config.json"];

/// Walk up from the provided directory looking for the first project config file.
pub(crate) fn discover_project_config_path_from(start: &Path) -> Option<PathBuf> {
    let mut current = if start.is_dir() {
        start.to_path_buf()
    } else {
        start.parent().map(Path::to_path_buf)?
    };
    loop {
        for name in PROJECT_MANIFEST_FILENAMES {
            let candidate = current.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        if !current.pop() {
            break;
        }
    }
    None
}

pub(crate) fn user_config_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(home_dir) = dirs::home_dir() {
        let user_dir = home_dir.join(".config/runmat");
        for name in USER_CONFIG_FILENAMES {
            paths.push(user_dir.join(name));
        }
    }
    paths
}
