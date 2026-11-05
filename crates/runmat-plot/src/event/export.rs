//! Scene export: the byte budget, export policy and the errors an export reports.

use super::*;

pub const DEFAULT_FIGURE_SCENE_EXPORT_BUDGET_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneExportPolicy {
    pub max_scene_bytes: usize,
}

impl Default for SceneExportPolicy {
    fn default() -> Self {
        Self {
            max_scene_bytes: DEFAULT_FIGURE_SCENE_EXPORT_BUDGET_BYTES,
        }
    }
}

pub fn resolve_scene_export_policy(max_scene_bytes: Option<usize>) -> SceneExportPolicy {
    SceneExportPolicy {
        max_scene_bytes: max_scene_bytes
            .filter(|bytes| *bytes > 0)
            .unwrap_or(DEFAULT_FIGURE_SCENE_EXPORT_BUDGET_BYTES),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneExportErrorKind {
    BudgetExceeded,
    UnexportableGpuData,
    Serialization,
    Readback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneExportError {
    pub kind: SceneExportErrorKind,
    pub message: String,
}

impl SceneExportError {
    pub(super) fn budget_exceeded(used: usize, added: usize, max: usize) -> Self {
        Self {
            kind: SceneExportErrorKind::BudgetExceeded,
            message: format!(
                "figure scene export exceeds budget: {} + {} bytes > {} bytes",
                used, added, max
            ),
        }
    }

    pub(super) fn unexportable(message: impl Into<String>) -> Self {
        Self {
            kind: SceneExportErrorKind::UnexportableGpuData,
            message: message.into(),
        }
    }

    pub(super) fn serialization(message: impl Into<String>) -> Self {
        Self {
            kind: SceneExportErrorKind::Serialization,
            message: message.into(),
        }
    }

    pub(super) fn readback(message: impl Into<String>) -> Self {
        Self {
            kind: SceneExportErrorKind::Readback,
            message: message.into(),
        }
    }
}

impl fmt::Display for SceneExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for SceneExportError {}

/// An export failure with the public category/message and an inspectable plot-data cause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneExportFailure {
    pub error: SceneExportError,
    pub(super) source: Option<crate::plots::PlotDataExportError>,
}

impl SceneExportFailure {
    pub(super) fn readback_typed(source: crate::plots::PlotDataExportError) -> Self {
        Self {
            error: SceneExportError::readback(source.to_string()),
            source: Some(source),
        }
    }
}

impl From<SceneExportError> for SceneExportFailure {
    fn from(error: SceneExportError) -> Self {
        Self {
            error,
            source: None,
        }
    }
}

impl fmt::Display for SceneExportFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}

impl Error for SceneExportFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_ref()
            .map(|source| source as &(dyn Error + 'static))
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SceneExportBudget {
    pub(super) max_bytes: usize,
    pub(super) used_bytes: usize,
}

impl SceneExportBudget {
    pub(super) fn new(policy: SceneExportPolicy) -> Self {
        Self {
            max_bytes: policy.max_scene_bytes,
            used_bytes: 0,
        }
    }

    pub(super) fn reserve_plot(&mut self, plot: &ScenePlot) -> Result<(), SceneExportError> {
        let bytes = serde_json::to_vec(plot)
            .map_err(|err| SceneExportError::serialization(err.to_string()))?;
        self.reserve_bytes(bytes.len())
    }

    pub(super) fn reserve_bytes(&mut self, byte_len: usize) -> Result<(), SceneExportError> {
        let next = self.used_bytes.saturating_add(byte_len);
        if next > self.max_bytes {
            return Err(SceneExportError::budget_exceeded(
                self.used_bytes,
                byte_len,
                self.max_bytes,
            ));
        }
        self.used_bytes = next;
        Ok(())
    }
}
