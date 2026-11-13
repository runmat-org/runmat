use serde::{Deserialize, Serialize};

use runmat_geometry_core::PersistentEntityId;

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralNode {
    pub node_id: u32,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
    pub solid_attachment: Option<ConformalSolidAttachment>,
}

/// Declares that an explicit beam/shell node and one exact canonical solid-mesh node are the same
/// structural node. Translation is shared; the explicit formulation contributes any rotational
/// degrees of freedom. Geometry-authored nodes bind by occurrence-qualified CAD identity, while
/// mesh-derived nodes bind directly by canonical solver-node identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ConformalSolidAttachment {
    GeometryVertex { entity: PersistentEntityId },
    SolverNode { stable_identity: [u8; 32] },
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
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub area_m2: runmat_types::physical::Area,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub iy_m4: runmat_types::physical::SecondMomentOfArea,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub iz_m4: runmat_types::physical::SecondMomentOfArea,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub torsion_j_m4: runmat_types::physical::SecondMomentOfArea,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    #[serde(default)]
    pub outer_fiber_y_m: runmat_types::physical::Length,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    #[serde(default)]
    pub outer_fiber_z_m: runmat_types::physical::Length,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    #[serde(default)]
    pub torsion_outer_radius_m: runmat_types::physical::Length,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HomogeneousShellSectionModel {
    pub section_id: String,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub thickness_m: runmat_types::physical::Length,
    pub reference_surface: HomogeneousShellReferenceSurface,
    #[serde(default = "default_shell_shear_correction")]
    pub shear_correction: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "distance_m")]
pub enum HomogeneousShellReferenceSurface {
    MidSurface,
    NegativeNormalSurface,
    PositiveNormalSurface,
    OffsetFromMidSurface(f64),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaminateShellSectionModel {
    pub section_id: String,
    pub reference_orientation_id: String,
    pub ply_order: PlyOrderDirection,
    pub reference_surface: ShellReferenceSurface,
    pub plies: Vec<LaminatePlyModel>,
    pub shear_correction: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlyOrderDirection {
    AlongSurfaceNormal,
    OppositeSurfaceNormal,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "distance_m")]
pub enum ShellReferenceSurface {
    MidSurface,
    FirstPlySurface,
    LastPlySurface,
    OffsetFromMidSurface(f64),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "section")]
pub enum ShellSectionModel {
    Homogeneous(HomogeneousShellSectionModel),
    Laminate(LaminateShellSectionModel),
}

impl ShellSectionModel {
    pub fn section_id(&self) -> &str {
        match self {
            Self::Homogeneous(value) => &value.section_id,
            Self::Laminate(value) => &value.section_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaminatePlyModel {
    pub ply_id: String,
    pub material_id: String,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub thickness_m: runmat_types::physical::Length,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub angle_rad: runmat_types::physical::Quantity<runmat_types::physical::kind::Angle, f64>,
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
