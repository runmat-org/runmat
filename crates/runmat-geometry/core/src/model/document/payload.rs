use serde::{Deserialize, Serialize};

use super::GeometryDigest;

pub const EXACT_BREP_MEDIA_TYPE: &str = "application/vnd.runmat.geometry.exact-brep.v2+cbor";
pub const FACETED_SOLID_MEDIA_TYPE: &str = "application/vnd.runmat.geometry.faceted-solid.v2+cbor";
pub const DISPLAY_TESSELLATION_MEDIA_TYPE: &str =
    "application/vnd.runmat.geometry.display-tessellation.v2+cbor";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryObjectRef {
    pub digest: GeometryDigest,
    pub encoded_length: u64,
    pub media_type: String,
    pub schema_version: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactGeometryCapabilities {
    pub curve_point: bool,
    pub curve_tangent: bool,
    pub curve_curvature: bool,
    pub curve_arc_length: bool,
    pub curve_inverse_projection: bool,
    pub pcurve_point: bool,
    pub pcurve_derivatives: bool,
    pub surface_point: bool,
    pub surface_first_derivatives: bool,
    pub surface_second_derivatives: bool,
    pub surface_normal: bool,
    pub surface_principal_curvature: bool,
    pub surface_uv_bounds: bool,
    pub surface_periodicity: bool,
    pub surface_closest_point: bool,
    pub trim_domain_classification: bool,
    pub mass_properties: bool,
}

impl ExactGeometryCapabilities {
    pub const fn complete_for_meshing(&self) -> bool {
        self.curve_point
            && self.curve_tangent
            && self.curve_curvature
            && self.curve_arc_length
            && self.curve_inverse_projection
            && self.pcurve_point
            && self.pcurve_derivatives
            && self.surface_point
            && self.surface_first_derivatives
            && self.surface_second_derivatives
            && self.surface_normal
            && self.surface_principal_curvature
            && self.surface_uv_bounds
            && self.surface_periodicity
            && self.surface_closest_point
            && self.trim_domain_classification
            && self.mass_properties
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactBRepModel {
    pub artifact: GeometryObjectRef,
    pub kernel_abi: String,
    pub capabilities: ExactGeometryCapabilities,
    pub assembly_count: u64,
    pub instance_count: u64,
    pub body_count: u64,
    pub lump_count: u64,
    pub solid_count: u64,
    pub region_count: u64,
    pub shell_count: u64,
    pub face_count: u64,
    pub wire_count: u64,
    pub coedge_count: u64,
    pub edge_count: u64,
    pub vertex_count: u64,
    pub interface_count: u64,
    pub contact_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FacetedSolidModel {
    pub artifact: GeometryObjectRef,
    pub vertex_count: u64,
    pub triangle_count: u64,
    pub shell_count: u64,
    pub is_watertight: bool,
    pub is_oriented: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GeometryModel {
    ExactBRep { model: ExactBRepModel },
    FacetedSolid { model: FacetedSolidModel },
}

impl GeometryModel {
    pub const fn primary_artifact(&self) -> &GeometryObjectRef {
        match self {
            Self::ExactBRep { model } => &model.artifact,
            Self::FacetedSolid { model } => &model.artifact,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DisplayTessellationRef {
    pub profile_id: String,
    pub geometry_revision: u64,
    pub derived_from_primary_digest: GeometryDigest,
    pub artifact: GeometryObjectRef,
}
