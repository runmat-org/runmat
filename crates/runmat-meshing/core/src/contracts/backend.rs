use serde::{Deserialize, Serialize};

use super::options::VolumeMeshingOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshBackendKind {
    Auto,
    Solid,
    StructuredGridTetrahedron,
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
    pub solid_ready: bool,
    pub reason: &'static str,
}

pub fn select_volume_backend(options: &VolumeMeshingOptions) -> MeshBackendSelection {
    match options.backend {
        MeshBackendKind::Auto => MeshBackendSelection {
            requested: MeshBackendKind::Auto,
            selected: MeshBackendKind::Solid,
            solid_ready: true,
            reason: "solid_backend_default",
        },
        MeshBackendKind::Solid => MeshBackendSelection {
            requested: MeshBackendKind::Solid,
            selected: MeshBackendKind::Solid,
            solid_ready: true,
            reason: "explicit_solid_backend",
        },
        MeshBackendKind::StructuredGridTetrahedron => MeshBackendSelection {
            requested: MeshBackendKind::StructuredGridTetrahedron,
            selected: MeshBackendKind::StructuredGridTetrahedron,
            solid_ready: false,
            reason: "explicit_structured_grid_tetrahedron",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_selects_solid_backend() {
        let selection = select_volume_backend(&VolumeMeshingOptions::default());

        assert_eq!(selection.requested, MeshBackendKind::Auto);
        assert_eq!(selection.selected, MeshBackendKind::Solid);
        assert!(selection.solid_ready);
        assert_eq!(selection.reason, "solid_backend_default");
    }

    #[test]
    fn explicit_structured_grid_tetrahedron_selects_structured_backend() {
        let selection = select_volume_backend(&VolumeMeshingOptions {
            backend: MeshBackendKind::StructuredGridTetrahedron,
            ..VolumeMeshingOptions::default()
        });

        assert_eq!(
            selection.requested,
            MeshBackendKind::StructuredGridTetrahedron
        );
        assert_eq!(
            selection.selected,
            MeshBackendKind::StructuredGridTetrahedron
        );
        assert!(!selection.solid_ready);
        assert_eq!(selection.reason, "explicit_structured_grid_tetrahedron");
    }
}
