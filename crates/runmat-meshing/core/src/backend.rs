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
            selected: MeshBackendKind::StructuredTetFallback,
            production_ready: false,
            reason: "production_backend_pending",
        },
        MeshBackendKind::Production => MeshBackendSelection {
            requested: MeshBackendKind::Production,
            selected: MeshBackendKind::Production,
            production_ready: false,
            reason: "production_backend_pending",
        },
        MeshBackendKind::StructuredTetFallback => MeshBackendSelection {
            requested: MeshBackendKind::StructuredTetFallback,
            selected: MeshBackendKind::StructuredTetFallback,
            production_ready: false,
            reason: "explicit_structured_tet_fallback",
        },
    }
}
