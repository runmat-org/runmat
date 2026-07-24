use serde::{Deserialize, Serialize};

use crate::selection::EntityKind;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region {
    pub region_id: String,
    pub name: String,
    pub tag: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cad_ownership: Option<CadRegionOwnership>,
}

impl Region {
    pub fn has_material_role(&self) -> bool {
        self.tag.as_deref().is_some_and(is_material_role_text)
            || self
                .cad_ownership
                .as_ref()
                .is_some_and(CadRegionOwnership::has_material_role)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadSemanticKind {
    Assembly,
    Component,
    Reference,
    Body,
    Compound,
    Face,
    Subshape,
    Layer,
    Color,
    Material,
    Shape,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CadLabelRef {
    pub label_entry: String,
    pub name: String,
    pub kind: CadSemanticKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CadColorEvidence {
    pub source: String,
    pub color_type: String,
    pub hex_rgba: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CadPhysicalMaterialEvidence {
    pub label_entry: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub density: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub density_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub density_value_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CadRegionOwnership {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub face_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curve_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<CadLabelRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub owner_path: Vec<CadLabelRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<CadColorEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<CadPhysicalMaterialEvidence>,
}

impl CadRegionOwnership {
    pub fn has_material_role(&self) -> bool {
        self.material.is_some()
            || self
                .label
                .as_ref()
                .is_some_and(|label| label.kind == CadSemanticKind::Material)
            || self
                .owner_path
                .iter()
                .any(|label| label.kind == CadSemanticKind::Material)
    }
}

fn is_material_role_text(value: &str) -> bool {
    value
        .split(|character: char| !character.is_ascii_alphanumeric())
        .any(|token| token.eq_ignore_ascii_case("material"))
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
