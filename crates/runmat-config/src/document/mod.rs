mod error;
mod format;
mod legacy;
mod patch;
mod projection;
mod replace;

pub use error::RunmatConfigDocumentError;
pub use format::RunmatConfigFormat;
pub use legacy::{
    migrate_legacy_desktop_config, migrate_legacy_desktop_config_between,
    migrate_legacy_desktop_config_into, LegacyDesktopMigration,
};
pub use patch::{
    DesktopArtifactsPatch, DesktopNotebookPatch, DesktopRunHistoryPatch, DesktopScriptPatch,
    RunmatConfigPatch, RuntimeConfigPatch,
};

use crate::desktop::DesktopConfig;
use crate::runtime::RunMatRuntimeConfig;

#[derive(Clone, Debug)]
pub struct RunmatConfigDocument {
    source: String,
    format: RunmatConfigFormat,
    desktop: DesktopConfig,
    runtime: RunMatRuntimeConfig,
}

impl RunmatConfigDocument {
    pub fn parse(
        source: impl Into<String>,
        format: RunmatConfigFormat,
    ) -> Result<Self, RunmatConfigDocumentError> {
        let source = source.into();
        let projection = projection::parse_projection(&source, format)?;
        projection.desktop.validate()?;
        Ok(Self {
            source,
            format,
            desktop: projection.desktop,
            runtime: projection.runtime,
        })
    }

    pub fn parse_path(
        source: impl Into<String>,
        path: &std::path::Path,
    ) -> Result<Self, RunmatConfigDocumentError> {
        Self::parse(source, RunmatConfigFormat::from_path(path)?)
    }

    pub fn format(&self) -> RunmatConfigFormat {
        self.format
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn into_source(self) -> String {
        self.source
    }

    pub fn desktop(&self) -> &DesktopConfig {
        &self.desktop
    }

    pub fn runtime(&self) -> &RunMatRuntimeConfig {
        &self.runtime
    }

    pub fn patched(&self, patch: &RunmatConfigPatch) -> Result<Self, RunmatConfigDocumentError> {
        let source = patch::apply_patch(&self.source, self.format, patch)?;
        Self::parse(source, self.format)
    }

    /// Replace the complete runtime-owned section while preserving every
    /// unrelated top-level section in the document.
    pub fn with_runtime(
        &self,
        runtime: &RunMatRuntimeConfig,
    ) -> Result<Self, RunmatConfigDocumentError> {
        let source = replace::replace_runtime(&self.source, self.format, runtime)?;
        Self::parse(source, self.format)
    }
}
