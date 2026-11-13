//! Read-only acceptance never needs to duplicate a refined mesh.
use crate::{ExactFaceGeometry, ExactFaceRefinedMesh, ExactFaceRefinedTopology};

#[derive(Clone, Copy)]
pub(crate) struct ExactFaceRefinedMeshView<'a> {
    pub topology: &'a ExactFaceRefinedTopology,
    pub geometry: &'a ExactFaceGeometry,
}

impl<'a> From<&'a ExactFaceRefinedMesh> for ExactFaceRefinedMeshView<'a> {
    fn from(mesh: &'a ExactFaceRefinedMesh) -> Self {
        Self {
            topology: &mesh.topology,
            geometry: &mesh.geometry,
        }
    }
}
