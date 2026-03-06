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
