use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityKind {
    Node,
    Edge,
    Face,
    Element,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityRef {
    pub geometry_id: String,
    pub geometry_revision: u32,
    pub mesh_id: String,
    pub entity_kind: EntityKind,
    pub entity_id: u64,
}
