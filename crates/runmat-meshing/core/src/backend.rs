use serde::{Deserialize, Serialize};

use crate::VolumeMeshingOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshBackendKind {
    Auto,
    Production,
    StructuredTetFallback,
}

impl Default for MeshBackendKind {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshBackendSelection {
    pub requested: MeshBackendKind,
    pub selected: MeshBackendKind,
    pub production_ready: bool,
    pub reason: &'static str,
}

pub fn select_volume_backend(options: &VolumeMeshingOptions) -> MeshBackendSelection {
    match options.backend {
        MeshBackendKind::Auto => MeshBackendSelection {
            requested: MeshBackendKind::Auto,
            selected: MeshBackendKind::Production,
            production_ready: true,
            reason: "production_backend_default",
        },
        MeshBackendKind::Production => MeshBackendSelection {
            requested: MeshBackendKind::Production,
            selected: MeshBackendKind::Production,
            production_ready: true,
            reason: "explicit_production_backend",
        },
        MeshBackendKind::StructuredTetFallback => MeshBackendSelection {
            requested: MeshBackendKind::StructuredTetFallback,
            selected: MeshBackendKind::StructuredTetFallback,
            production_ready: false,
            reason: "explicit_structured_tet_fallback",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_selects_production_backend() {
        let selection = select_volume_backend(&VolumeMeshingOptions::default());

        assert_eq!(selection.requested, MeshBackendKind::Auto);
        assert_eq!(selection.selected, MeshBackendKind::Production);
        assert!(selection.production_ready);
        assert_eq!(selection.reason, "production_backend_default");
    }

    #[test]
    fn explicit_fallback_remains_available() {
        let selection = select_volume_backend(&VolumeMeshingOptions {
            backend: MeshBackendKind::StructuredTetFallback,
            ..VolumeMeshingOptions::default()
        });

        assert_eq!(selection.requested, MeshBackendKind::StructuredTetFallback);
        assert_eq!(selection.selected, MeshBackendKind::StructuredTetFallback);
        assert!(!selection.production_ready);
        assert_eq!(selection.reason, "explicit_structured_tet_fallback");
    }
}
