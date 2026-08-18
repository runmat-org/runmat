use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{StableDigest, SurfaceQualityTargets};
use serde::{Deserialize, Serialize};

use crate::{ExactFaceAcceptanceErrorKind, ExactFaceAcceptanceOptions, ExactFaceRefinementContext};

#[derive(Clone, Copy)]
pub struct ExactFaceJoinContext<'a> {
    pub refinement: ExactFaceRefinementContext<'a>,
    pub quality: SurfaceQualityTargets,
    pub acceptance: ExactFaceAcceptanceOptions,
}

impl<'a> ExactFaceJoinContext<'a> {
    pub fn new(
        refinement: ExactFaceRefinementContext<'a>,
        quality: SurfaceQualityTargets,
        acceptance: ExactFaceAcceptanceOptions,
    ) -> Self {
        Self {
            refinement,
            quality,
            acceptance,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactFaceJoinOptions {
    pub coordinate_tolerance_m: f64,
    pub maximum_nodes: u64,
    pub maximum_triangles: u64,
    pub maximum_boundary_segments: u64,
}

impl Default for ExactFaceJoinOptions {
    fn default() -> Self {
        Self {
            coordinate_tolerance_m: 1.0e-10,
            maximum_nodes: 100_000_000,
            maximum_triangles: 200_000_000,
            maximum_boundary_segments: 100_000_000,
        }
    }
}

pub const EXACT_FACE_MESH_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFaceMesh {
    pub schema_version: u16,
    pub source_face_id: PersistentEntityId,
    pub nodes: Vec<ExactFaceMeshNode>,
    pub triangles: Vec<ExactFaceMeshTriangle>,
    pub boundary_segments: Vec<ExactFaceMeshBoundarySegment>,
    pub joined_chart_cuts: Vec<ExactFaceMeshJoinedCut>,
    pub maximum_chordal_deviation_m: f64,
    pub maximum_normal_deviation_rad: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFaceMeshJoinedCut {
    pub cut_id: StableDigest,
    pub piece_count: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFaceMeshNode {
    pub node_id: StableDigest,
    pub point_m: [f64; 3],
    pub uses: Vec<ExactFaceMeshNodeUse>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFaceMeshNodeUse {
    pub source_face_id: PersistentEntityId,
    pub chart_id: StableDigest,
    pub uv: [f64; 2],
    pub evaluator_uv: [f64; 2],
    pub exact_edge_parameters: Vec<ExactFaceMeshEdgeParameter>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFaceMeshEdgeParameter {
    pub source_coedge_id: PersistentEntityId,
    pub source_edge_id: PersistentEntityId,
    pub parameter: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFaceMeshTriangle {
    pub triangle_id: StableDigest,
    pub chart_id: StableDigest,
    pub source_face_id: PersistentEntityId,
    pub node_ids: [StableDigest; 3],
    pub unit_normal: [f64; 3],
    pub physical_area_m2: f64,
    pub metric_edge_lengths: [f64; 3],
    pub minimum_metric_angle_rad: f64,
    pub physical_aspect_ratio: f64,
    pub chordal_deviation_m: f64,
    pub normal_deviation_rad: f64,
    pub acceptance_sample_count: u64,
    pub accepted_chordal_deviation_m: f64,
    pub accepted_normal_deviation_rad: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFaceMeshBoundarySegment {
    pub source_coedge_id: PersistentEntityId,
    pub source_edge_id: PersistentEntityId,
    pub node_ids: [StableDigest; 2],
    pub edge_parameters: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFaceJoinErrorKind {
    InvalidOptions,
    InvalidInput,
    Acceptance(ExactFaceAcceptanceErrorKind),
    ResourceLimit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFaceJoinError {
    pub kind: ExactFaceJoinErrorKind,
    pub source_face_id: Box<PersistentEntityId>,
    pub chart_id: Option<StableDigest>,
    pub reason: String,
}

impl ExactFaceJoinError {
    pub(super) fn new(
        kind: ExactFaceJoinErrorKind,
        source_face_id: &PersistentEntityId,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            source_face_id: Box::new(source_face_id.clone()),
            chart_id: None,
            reason: reason.into(),
        }
    }

    pub(super) fn with_chart(mut self, chart_id: StableDigest) -> Self {
        self.chart_id = Some(chart_id);
        self
    }
}

impl std::fmt::Display for ExactFaceJoinError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "exact face join {:?} for {:?} chart {:?}: {}",
            self.kind, self.source_face_id, self.chart_id, self.reason
        )
    }
}

impl std::error::Error for ExactFaceJoinError {}
