use serde::{Deserialize, Serialize};

use crate::selection::EntityKind;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region {
    pub region_id: String,
    pub name: String,
    pub tag: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntityIdRange {
    pub start: u64,
    pub count: u64,
}

impl EntityIdRange {
    pub fn new(start: u64, count: u64) -> Self {
        Self { start, count }
    }

    pub fn end_exclusive(&self) -> Option<u64> {
        self.start.checked_add(self.count)
    }

    pub fn contains(&self, entity_id: u64) -> bool {
        self.end_exclusive()
            .is_some_and(|end| entity_id >= self.start && entity_id < end)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RegionEntityMapping {
    pub region_id: String,
    pub mesh_id: String,
    pub entity_kind: EntityKind,
    pub ranges: Vec<EntityIdRange>,
}

impl RegionEntityMapping {
    pub fn new(
        region_id: impl Into<String>,
        mesh_id: impl Into<String>,
        entity_kind: EntityKind,
        ranges: Vec<EntityIdRange>,
    ) -> Self {
        Self {
            region_id: region_id.into(),
            mesh_id: mesh_id.into(),
            entity_kind,
            ranges,
        }
    }

    pub fn all_faces(
        region_id: impl Into<String>,
        mesh_id: impl Into<String>,
        face_count: u64,
    ) -> Self {
        Self::new(
            region_id,
            mesh_id,
            EntityKind::Face,
            if face_count == 0 {
                Vec::new()
            } else {
                vec![EntityIdRange::new(0, face_count)]
            },
        )
    }

    pub fn entity_count(&self) -> u64 {
        self.ranges.iter().map(|range| range.count).sum()
    }

    pub fn contains_entity(&self, entity_id: u64) -> bool {
        self.ranges.iter().any(|range| range.contains(entity_id))
    }
}
