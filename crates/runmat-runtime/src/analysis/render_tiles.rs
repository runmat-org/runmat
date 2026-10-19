//! Immutable, spatially ordered surface pieces for out-of-core result views.
//! The source topology remains exact; these pieces retain original node and
//! triangle identities so fields can be resolved without serializing a plot.

use std::collections::{BinaryHeap, HashMap, HashSet};

use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{
    AnalysisRenderMesh, AnalysisRenderRegion, AnalysisRenderTopology, AnalysisRenderTopologySource,
};

pub const MAX_TRIANGLES_PER_TILE: usize = 8192;
pub const MAX_TRIANGLES_PER_LOD_NODE: usize = 1024;
pub const DEFAULT_VIEW_TRIANGLE_BUDGET: usize = 250_000;
const LOD_FANOUT: usize = 4;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderTileIndex {
    pub schema_version: String,
    pub topology_schema_version: String,
    pub source: AnalysisRenderTopologySource,
    pub meshes: Vec<RenderTileMeshIndex>,
    pub volume_element_count: usize,
    pub field_summaries: Vec<RenderFieldSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderFieldSummary {
    pub field_id: String,
    pub minimum: f64,
    pub maximum: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderTileMeshIndex {
    pub mesh_id: String,
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub regions: Vec<AnalysisRenderRegion>,
    pub tiles: Vec<RenderTileDescriptor>,
    pub lod_nodes: Vec<RenderLodNode>,
    pub lod_root: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderLodNode {
    pub tile: RenderTileDescriptor,
    pub source_bounds: [[f64; 3]; 2],
    pub children: Vec<usize>,
    pub exact_tile_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderTileDescriptor {
    pub digest: Digest,
    pub bounds: [[f64; 3]; 2],
    pub triangle_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderTile {
    pub global_vertex_indices: Vec<u32>,
    pub vertices: Vec<[f64; 3]>,
    pub vertex_volume_node_indices: Vec<Option<usize>>,
    pub global_triangle_indices: Vec<u32>,
    pub triangles: Vec<[u32; 3]>,
    pub triangle_volume_element_indices: Vec<Option<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderViewRequest {
    pub camera_position: [f64; 3],
    pub camera_target: [f64; 3],
    pub camera_up: [f64; 3],
    pub fov_y_radians: f64,
    #[serde(default)]
    pub orthographic_height: Option<f64>,
    pub aspect_ratio: f64,
    pub viewport_height_px: f64,
    pub triangle_budget: usize,
    pub refine_above_px: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderViewPlan {
    pub nodes: Vec<(usize, usize)>,
    pub triangle_count: usize,
    pub exact: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderViewSelection {
    pub topology_schema_version: String,
    pub source: AnalysisRenderTopologySource,
    pub volume_element_count: usize,
    pub mesh_count: usize,
    pub field_summaries: Vec<RenderFieldSummary>,
    pub model_bounds: [[f64; 3]; 2],
    pub nodes: Vec<RenderSelectedNode>,
    pub exact: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderSelectedNode {
    pub mesh_index: usize,
    pub node_index: usize,
    pub mesh_id: String,
}

/// Exact tile index and its immutable content-addressed payloads.
pub type RenderTilePartition = (RenderTileIndex, Vec<(Digest, Vec<u8>)>);

mod codec;
mod hierarchy;
mod partition;
mod selection;
use codec::{bounds, morton_key};
pub use codec::{decode_tile, reconstruct};
use hierarchy::*;
pub use partition::partition;
#[cfg(test)]
mod tests;
