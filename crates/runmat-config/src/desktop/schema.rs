use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopConfig {
    pub artifacts: DesktopArtifactsConfig,
    pub run_history: DesktopRunHistoryConfig,
    pub script: DesktopScriptConfig,
    pub notebook: DesktopNotebookConfig,
}

impl DesktopConfig {
    pub fn validate(&self) -> Result<(), DesktopConfigValidationError> {
        validate_artifact_root(&self.artifacts.root)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopArtifactsConfig {
    pub root: PathBuf,
}

impl Default for DesktopArtifactsConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from(".artifacts"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopRunHistoryConfig {
    pub mode: DesktopRunHistoryMode,
    pub trace: bool,
    pub logs: DesktopRunLogMode,
}

impl Default for DesktopRunHistoryConfig {
    fn default() -> Self {
        Self {
            mode: DesktopRunHistoryMode::Budgeted,
            trace: true,
            logs: DesktopRunLogMode::All,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopRunHistoryMode {
    Off,
    Budgeted,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopRunLogMode {
    Off,
    Errors,
    All,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopScriptConfig {
    pub clear_workspace_before_run: bool,
    pub clear_figures_before_run: bool,
}

impl Default for DesktopScriptConfig {
    fn default() -> Self {
        Self {
            clear_workspace_before_run: true,
            clear_figures_before_run: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopNotebookConfig {
    pub on_error: DesktopNotebookOnError,
    pub rerun_after_cancel: DesktopNotebookRerunAfterCancel,
}

impl Default for DesktopNotebookConfig {
    fn default() -> Self {
        Self {
            on_error: DesktopNotebookOnError::Stop,
            rerun_after_cancel: DesktopNotebookRerunAfterCancel::Remaining,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopNotebookOnError {
    Stop,
    Continue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopNotebookRerunAfterCancel {
    Remaining,
    All,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct DesktopConfigValidationError {
    pub field: &'static str,
    pub message: String,
}

fn validate_artifact_root(path: &Path) -> Result<(), DesktopConfigValidationError> {
    let invalid = |message: &str| DesktopConfigValidationError {
        field: "desktop.artifacts.root",
        message: message.to_string(),
    };
    if path.as_os_str().is_empty() || path == Path::new(".") {
        return Err(invalid(
            "[desktop.artifacts].root must name a project-relative subdirectory",
        ));
    }
    let normalized = path.to_string_lossy().replace('\\', "/");
    let windows_absolute = normalized.as_bytes().get(1) == Some(&b':')
        && normalized
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphabetic);
    if path.is_absolute()
        || normalized.starts_with('/')
        || windows_absolute
        || normalized.starts_with('~')
        || normalized.split('/').any(|component| component == "..")
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(invalid(
            "[desktop.artifacts].root must be relative and cannot contain `..`",
        ));
    }
    let first = normalized
        .trim_start_matches("./")
        .split('/')
        .next()
        .unwrap_or("");
    if matches!(
        first.to_ascii_lowercase().as_str(),
        ".runmat" | "runmat.toml" | "runmat.json" | "runmat.lock"
    ) {
        return Err(invalid(
            "[desktop.artifacts].root cannot overlap RunMat configuration or internal state",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_desktop_product_behavior() {
        let config = DesktopConfig::default();
        assert_eq!(config.artifacts.root, Path::new(".artifacts"));
        assert_eq!(config.run_history.mode, DesktopRunHistoryMode::Budgeted);
        assert!(config.run_history.trace);
        assert_eq!(config.run_history.logs, DesktopRunLogMode::All);
        assert!(config.script.clear_workspace_before_run);
        assert!(config.script.clear_figures_before_run);
        assert_eq!(config.notebook.on_error, DesktopNotebookOnError::Stop);
        assert_eq!(
            config.notebook.rerun_after_cancel,
            DesktopNotebookRerunAfterCancel::Remaining
        );
    }

    #[test]
    fn rejects_unsafe_artifact_roots() {
        for root in [
            "",
            ".",
            "..",
            "../artifacts",
            r"..\artifacts",
            "/artifacts",
            r"C:\artifacts",
            r"\\server\artifacts",
            "~/artifacts",
            ".runmat/cache",
        ] {
            let mut config = DesktopConfig::default();
            config.artifacts.root = root.into();
            assert!(config.validate().is_err(), "{root}");
        }
    }
}
