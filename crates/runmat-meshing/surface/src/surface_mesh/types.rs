use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::MeshingPartitionDescriptor;
use runmat_meshing_curve::SharedCurveSegmentSplit;
use serde::{Deserialize, Serialize};

use crate::{
    ExactFaceMesh, ExactFaceMeshBoundarySegment, ExactFaceMeshNode, ExactFaceMeshTriangle,
};

pub const EXACT_FACE_MESH_BATCH_SCHEMA_VERSION: u16 = 1;
pub const EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION: u16 = 1;
pub const EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION: u16 = 1;
pub const EXACT_SURFACE_MESH_SCHEMA_VERSION: u16 = 1;
pub const MAX_EXACT_FACE_PARTITIONS: usize = 63;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactSurfaceJoinOptions {
    pub coordinate_tolerance_m: f64,
    pub maximum_nodes: u64,
    pub maximum_triangles: u64,
    pub maximum_boundary_segments: u64,
}

impl Default for ExactSurfaceJoinOptions {
    fn default() -> Self {
        Self {
            coordinate_tolerance_m: 1.0e-10,
            maximum_nodes: 1_000_000_000,
            maximum_triangles: 2_000_000_000,
            maximum_boundary_segments: 1_000_000_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFaceMeshBatch {
    pub schema_version: u16,
    pub partition: MeshingPartitionDescriptor,
    pub faces: Vec<ExactFaceMesh>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFacePartitionResult {
    pub schema_version: u16,
    pub partition: MeshingPartitionDescriptor,
    pub outcome: ExactFacePartitionOutcome,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactFacePartitionOutcome {
    Converged {
        faces: Vec<ExactFaceMesh>,
    },
    RequiresCurveSplits {
        splits: Vec<SharedCurveSegmentSplit>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactSurfacePassResult {
    pub schema_version: u16,
    pub outcome: ExactSurfacePassOutcome,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactSurfacePassOutcome {
    RequiresCurveSplits {
        splits: Vec<SharedCurveSegmentSplit>,
    },
    Converged {
        surface: ExactSurfaceMesh,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactSurfaceMesh {
    pub schema_version: u16,
    pub face_ids: Vec<PersistentEntityId>,
    pub nodes: Vec<ExactFaceMeshNode>,
    pub triangles: Vec<ExactFaceMeshTriangle>,
    pub boundary_segments: Vec<ExactFaceMeshBoundarySegment>,
    pub shells: Vec<ExactSurfaceShellEvidence>,
    pub maximum_chordal_deviation_m: f64,
    pub maximum_normal_deviation_rad: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactSurfaceShellEvidence {
    pub source_shell_id: PersistentEntityId,
    pub face_count: u64,
    pub shared_edge_count: u64,
    pub open_edge_count: u64,
    pub nonmanifold_edge_count: u64,
    pub is_sheet_shell: bool,
    pub is_watertight: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactSurfaceMeshErrorKind {
    InvalidOptions,
    InvalidInput,
    InvalidEncoding,
    ResourceLimit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactSurfaceMeshError {
    pub kind: ExactSurfaceMeshErrorKind,
    pub source_face_id: Option<Box<PersistentEntityId>>,
    pub source_shell_id: Option<Box<PersistentEntityId>>,
    pub reason: String,
}

impl ExactSurfaceMeshError {
    pub(super) fn new(kind: ExactSurfaceMeshErrorKind, reason: impl Into<String>) -> Self {
        Self {
            kind,
            source_face_id: None,
            source_shell_id: None,
            reason: reason.into(),
        }
    }

    pub(super) fn with_face(mut self, face_id: &PersistentEntityId) -> Self {
        self.source_face_id = Some(Box::new(face_id.clone()));
        self
    }

    pub(super) fn with_shell(mut self, shell_id: &PersistentEntityId) -> Self {
        self.source_shell_id = Some(Box::new(shell_id.clone()));
        self
    }
}

impl std::fmt::Display for ExactSurfaceMeshError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "exact surface mesh {:?} for face {:?} shell {:?}: {}",
            self.kind, self.source_face_id, self.source_shell_id, self.reason
        )
    }
}

impl std::error::Error for ExactSurfaceMeshError {}
