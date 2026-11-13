use std::path::{Path, PathBuf};

use crate::project::PROJECT_MANIFEST_FILENAMES;

pub(crate) const USER_CONFIG_FILENAMES: &[&str] = &["config.toml", "config.json"];

/// User-application settings location, never a project filesystem location.
/// Discovery does not create a directory or write any setting.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ApplicationSettingsDirectory(PathBuf);

impl ApplicationSettingsDirectory {
    pub fn discover() -> Option<Self> {
        dirs::home_dir().and_then(Self::from_absolute_home)
    }

    /// Explicit host/test override. Relative paths cannot become ambient-CWD
    /// preference authorities.
    pub fn from_absolute_home(home: PathBuf) -> Option<Self> {
        home.is_absolute()
            .then(|| Self(home.join(".config/runmat")))
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    /// One application-owned versioned settings document, never project VFS.
    /// Looking up the path does not create a directory or file.
    pub fn settings_file(&self) -> PathBuf {
        self.0.join("settings-v1.json")
    }
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
    if let Some(user_dir) = ApplicationSettingsDirectory::discover() {
        paths.push(user_dir.settings_file());
        for name in USER_CONFIG_FILENAMES {
            paths.push(user_dir.path().join(name));
        }
    }
    paths
}

#[cfg(test)]
mod settings_tests {
    use super::ApplicationSettingsDirectory;
    use std::path::PathBuf;

    #[test]
    fn explicit_home_is_absolute_and_does_not_touch_disk() {
        assert!(
            ApplicationSettingsDirectory::from_absolute_home(PathBuf::from("relative")).is_none()
        );
        let home = std::env::temp_dir().join("runmat-settings-location-only");
        let location = ApplicationSettingsDirectory::from_absolute_home(home.clone()).unwrap();
        assert_eq!(location.path(), home.join(".config/runmat"));
        assert_eq!(
            location.settings_file(),
            home.join(".config/runmat/settings-v1.json")
        );
    }
}
