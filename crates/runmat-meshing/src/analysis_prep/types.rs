use runmat_geometry_core::MeshKind;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshingProfile {
    SurfaceOnly,
    AnalysisReady,
    AdaptiveRefine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshingOptions {
    pub profile: MeshingProfile,
    pub target_element_budget: usize,
}

impl Default for MeshingOptions {
    fn default() -> Self {
        Self {
            profile: MeshingProfile::AnalysisReady,
            target_element_budget: 250_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshConnectivityClass {
    SparseBand,
    SurfacePatch,
    VolumeCore,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementFamilyHint {
    Triangle,
    Quad,
    Tetrahedron,
    Hex,
    Mixed,
}

pub(super) fn default_coordinate_span_m() -> [f64; 3] {
    [1.0, 0.0, 0.0]
}

fn default_coordinate_active_dimension_count() -> u8 {
    1
}

pub(super) fn default_coordinate_characteristic_length_m() -> f64 {
    1.0
}

fn default_zero_u64() -> u64 {
    0
}

fn default_zero_f64() -> f64 {
    0.0
}

fn default_reference_element_coordinates_m() -> [[f64; 3]; 3] {
    [[0.0; 3]; 3]
}

fn default_element_topology_sample_edge_nodes() -> [[u32; 2]; 8] {
    [[0; 2]; 8]
}

fn default_element_topology_sample_node_coordinates_m() -> [[f64; 3]; 8] {
    [[0.0; 3]; 8]
}

fn default_element_topology_sample_element_edges() -> [[u32; 3]; 4] {
    [[0; 3]; 4]
}

fn default_element_topology_sample_element_orientations() -> [[i8; 3]; 4] {
    [[0; 3]; 4]
}

fn default_element_topology_sample_element_areas_m2() -> [f64; 4] {
    [0.0; 4]
}

fn default_element_topology_node_coordinates_m() -> Vec<[f64; 3]> {
    Vec::new()
}

fn default_element_topology_edge_nodes() -> Vec<[u32; 2]> {
    Vec::new()
}

fn default_element_topology_element_edges() -> Vec<[u32; 3]> {
    Vec::new()
}

fn default_element_topology_element_orientations() -> Vec<[i8; 3]> {
    Vec::new()
}

fn default_element_topology_element_areas_m2() -> Vec<f64> {
    Vec::new()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedMeshDescriptor {
    pub prepared_mesh_id: String,
    pub source_mesh_id: String,
    pub kind: MeshKind,
    pub node_count: u64,
    pub element_count: u64,
    pub connectivity_class: MeshConnectivityClass,
    pub element_family_hint: ElementFamilyHint,
    pub region_span_hint: u32,
    #[serde(default = "default_coordinate_span_m")]
    pub coordinate_span_m: [f64; 3],
    #[serde(default = "default_coordinate_active_dimension_count")]
    pub coordinate_active_dimension_count: u8,
    #[serde(default = "default_coordinate_characteristic_length_m")]
    pub coordinate_characteristic_length_m: f64,
    #[serde(default = "default_zero_u64")]
    pub element_geometry_node_count: u64,
    #[serde(default = "default_zero_u64")]
    pub element_geometry_edge_count: u64,
    #[serde(default = "default_zero_f64")]
    pub mean_element_edge_length_m: f64,
    #[serde(default = "default_zero_f64")]
    pub mean_element_area_m2: f64,
    #[serde(default = "default_zero_f64")]
    pub element_geometry_coverage_ratio: f64,
    #[serde(default = "default_reference_element_coordinates_m")]
    pub reference_element_coordinates_m: [[f64; 3]; 3],
    #[serde(default = "default_zero_f64")]
    pub reference_element_area_m2: f64,
    #[serde(default = "default_zero_u64")]
    pub control_volume_cell_count: u64,
    #[serde(default = "default_zero_u64")]
    pub control_volume_face_count: u64,
    #[serde(default = "default_zero_u64")]
    pub control_volume_internal_face_count: u64,
    #[serde(default = "default_zero_u64")]
    pub control_volume_boundary_face_count: u64,
    #[serde(default = "default_zero_f64")]
    pub control_volume_connectivity_coverage_ratio: f64,
    #[serde(default = "default_zero_u64")]
    pub element_topology_sample_element_count: u64,
    #[serde(default = "default_zero_u64")]
    pub element_topology_sample_edge_count: u64,
    #[serde(default = "default_element_topology_sample_edge_nodes")]
    pub element_topology_sample_edge_nodes: [[u32; 2]; 8],
    #[serde(default = "default_element_topology_sample_node_coordinates_m")]
    pub element_topology_sample_node_coordinates_m: [[f64; 3]; 8],
    #[serde(default = "default_element_topology_sample_element_edges")]
    pub element_topology_sample_element_edges: [[u32; 3]; 4],
    #[serde(default = "default_element_topology_sample_element_orientations")]
    pub element_topology_sample_element_orientations: [[i8; 3]; 4],
    #[serde(default = "default_element_topology_sample_element_areas_m2")]
    pub element_topology_sample_element_areas_m2: [f64; 4],
    #[serde(default = "default_element_topology_node_coordinates_m")]
    pub element_topology_node_coordinates_m: Vec<[f64; 3]>,
    #[serde(default = "default_element_topology_edge_nodes")]
    pub element_topology_edge_nodes: Vec<[u32; 2]>,
    #[serde(default = "default_element_topology_element_edges")]
    pub element_topology_element_edges: Vec<[u32; 3]>,
    #[serde(default = "default_element_topology_element_orientations")]
    pub element_topology_element_orientations: Vec<[i8; 3]>,
    #[serde(default = "default_element_topology_element_areas_m2")]
    pub element_topology_element_areas_m2: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionMeshMapping {
    pub region_id: String,
    pub source_mesh_ids: Vec<String>,
    pub prepared_mesh_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshingQualityReport {
    pub min_scaled_jacobian: f64,
    pub mean_aspect_ratio: f64,
    pub inverted_element_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshingProvenance {
    pub algorithm: String,
    pub profile: MeshingProfile,
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshingPrepResult {
    pub schema_version: String,
    pub prepared_meshes: Vec<PreparedMeshDescriptor>,
    pub region_mappings: Vec<RegionMeshMapping>,
    pub quality: MeshingQualityReport,
    pub provenance: MeshingProvenance,
}
