use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct StructuralModel {
    #[serde(default)]
    pub nodes: Vec<StructuralNode>,
    #[serde(default)]
    pub elements: Vec<StructuralElement>,
    #[serde(default)]
    pub beam_sections: Vec<BeamSectionModel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructuralNode {
    pub node_id: u32,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralElement {
    pub element_id: String,
    pub region_id: String,
    pub kind: StructuralElementKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralElementKind {
    Beam(BeamElementModel),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeamElementModel {
    pub node_ids: [u32; 2],
    pub section_id: String,
    #[serde(default = "default_beam_reference_axis")]
    pub reference_axis: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeamSectionModel {
    pub section_id: String,
    pub area_m2: f64,
    pub iy_m4: f64,
    pub iz_m4: f64,
    pub torsion_j_m4: f64,
}

fn default_beam_reference_axis() -> [f64; 3] {
    [0.0, 0.0, 1.0]
}
