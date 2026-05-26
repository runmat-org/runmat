use std::env;
use std::path::{Path, PathBuf};

use crate::PROJECT_MANIFEST_FILENAMES;

use super::env::env_value;

pub(crate) const USER_CONFIG_FILENAMES: &[&str] = &["config.toml", "config.json"];

/// Find configuration file path using hard-cutover precedence:
/// 1. RUNMAT_CONFIG explicit override
/// 2. nearest project runmat.toml/runmat.json (walking up from cwd)
/// 3. user config (~/.config/runmat/config.toml|json)
pub(crate) fn find_config_file() -> Option<PathBuf> {
    if let Some(config_path) = env_value("RUNMAT_CONFIG", &[]) {
        return Some(PathBuf::from(config_path));
    }

    if let Ok(current_dir) = env::current_dir() {
        if let Some(project) = discover_project_config_path_from(&current_dir) {
            return Some(project);
        }
    }

    user_config_candidates()
        .into_iter()
        .find(|candidate| candidate.is_file())
}

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
