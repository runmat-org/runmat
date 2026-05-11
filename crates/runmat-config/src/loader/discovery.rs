use std::env;
use std::path::{Path, PathBuf};

use super::env::env_value;

pub(crate) const CONFIG_FILENAMES: &[&str] = &[
    ".runmat",
    ".runmat.toml",
    ".runmat.yaml",
    ".runmat.yml",
    ".runmat.json",
    "runmat.config.toml",
    "runmat.config.yaml",
    "runmat.config.yml",
    "runmat.config.json",
];

/// Find potential configuration file paths.
pub(crate) fn find_config_files() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // 1. Environment variable override
    if let Some(config_path) = env_value("RUNMAT_CONFIG", &[]) {
        paths.push(PathBuf::from(config_path));
    }

    // 2. Current directory
    if let Ok(current_dir) = env::current_dir() {
        for name in CONFIG_FILENAMES {
            paths.push(current_dir.join(name));
        }
    }

    // 3. Home directory
    if let Some(home_dir) = dirs::home_dir() {
        for name in CONFIG_FILENAMES {
            paths.push(home_dir.join(name));
        }
        paths.push(home_dir.join(".config/runmat/config.yaml"));
        paths.push(home_dir.join(".config/runmat/config.yml"));
        paths.push(home_dir.join(".config/runmat/config.json"));
    }

    // 4. System-wide configurations
    #[cfg(unix)]
    {
        paths.push(PathBuf::from("/etc/runmat/config.yaml"));
        paths.push(PathBuf::from("/etc/runmat/config.yml"));
        paths.push(PathBuf::from("/etc/runmat/config.json"));
    }

    paths
}

/// Walk up from the provided directory looking for the first config file.
pub(crate) fn discover_config_path_from(start: &Path) -> Option<PathBuf> {
    let mut current = if start.is_dir() {
        start.to_path_buf()
    } else {
        start.parent().map(Path::to_path_buf)?
    };
    loop {
        for name in CONFIG_FILENAMES {
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
