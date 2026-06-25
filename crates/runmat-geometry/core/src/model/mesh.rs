use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshKind {
    Surface,
    Volume,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshDescriptor {
    pub mesh_id: String,
    pub kind: MeshKind,
    pub vertex_count: u64,
    pub element_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceMesh {
    pub mesh_id: String,
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

impl SurfaceMesh {
    pub fn new(
        mesh_id: impl Into<String>,
        vertices: Vec<[f64; 3]>,
        triangles: Vec<[u32; 3]>,
    ) -> Self {
        Self {
            mesh_id: mesh_id.into(),
            vertices,
            triangles,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.mesh_id.is_empty() {
            return Err("surface mesh id must not be empty");
        }
        if self
            .vertices
            .iter()
            .flatten()
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err("surface mesh vertices must be finite");
        }
        let vertex_count = self.vertices.len();
        if self
            .triangles
            .iter()
            .flatten()
            .any(|index| *index as usize >= vertex_count)
        {
            return Err("surface mesh triangle index out of bounds");
        }
        Ok(())
    }
}
