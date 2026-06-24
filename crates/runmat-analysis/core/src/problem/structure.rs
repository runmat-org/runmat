use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct StructuralModel {
    #[serde(default)]
    pub nodes: Vec<StructuralNode>,
    #[serde(default)]
    pub elements: Vec<StructuralElement>,
    #[serde(default)]
    pub beam_sections: Vec<BeamSectionModel>,
    #[serde(default)]
    pub shell_sections: Vec<ShellSectionModel>,
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
    Shell(ShellElementModel),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeamElementModel {
    pub node_ids: [u32; 2],
    pub section_id: String,
    #[serde(default = "default_beam_reference_axis")]
    pub reference_axis: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellElementModel {
    pub node_ids: [u32; 3],
    pub section_id: String,
    #[serde(default = "default_shell_reference_axis")]
    pub reference_axis: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeamSectionModel {
    pub section_id: String,
    pub area_m2: f64,
    pub iy_m4: f64,
    pub iz_m4: f64,
    pub torsion_j_m4: f64,
    #[serde(default)]
    pub outer_fiber_y_m: f64,
    #[serde(default)]
    pub outer_fiber_z_m: f64,
    #[serde(default)]
    pub torsion_outer_radius_m: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellSectionModel {
    pub section_id: String,
    pub thickness_m: f64,
    #[serde(default = "default_shell_shear_correction")]
    pub shear_correction: f64,
    #[serde(default = "default_shell_drilling_stiffness_scale")]
    pub drilling_stiffness_scale: f64,
}

fn default_beam_reference_axis() -> [f64; 3] {
    [0.0, 0.0, 1.0]
}

fn default_shell_reference_axis() -> [f64; 3] {
    [1.0, 0.0, 0.0]
}

fn default_shell_shear_correction() -> f64 {
    5.0 / 6.0
}

fn default_shell_drilling_stiffness_scale() -> f64 {
    1.0e-4
}
