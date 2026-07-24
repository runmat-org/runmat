use crate::validate::PlcValidationError;
use runmat_meshing_core::contracts::TopologyEntityId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlcBuildError {
    EmptySurface,
    MissingCurveBoundaryValidation,
    MissingSurfaceLoopCoverage,
    InconsistentSurfaceLoopCoverage {
        recovered_face_count: usize,
        surface_source_face_count: usize,
        boundary_loop_count: usize,
        hole_loop_count: usize,
        max_loops_per_face: usize,
        boundary_node_count: usize,
        recovered_source_edge_count: usize,
        protected_source_edge_count: usize,
        boundary_segment_count: usize,
    },
    InconsistentCadCurveBoundaryProvenance {
        reason: &'static str,
        recovered_source_edge_count: usize,
        protected_source_edge_count: usize,
        boundary_segment_count: usize,
        edge_report_count: usize,
    },
    MissingSurfaceNode {
        triangle_id: u32,
        node_id: u32,
    },
    NonFiniteSurfaceNode {
        node_id: u32,
    },
    NonFiniteSurfaceTriangle {
        triangle_id: u32,
    },
    NonPositiveSurfaceTriangleArea {
        triangle_id: u32,
    },
    InvalidSurfaceEntityId {
        entity_id: TopologyEntityId,
    },
    DuplicateFacet {
        element_id: u32,
    },
    AmbiguousProtectedBoundarySegment {
        node_ids: [u32; 2],
        first_source_edge_id: u32,
        second_source_edge_id: u32,
    },
    PartiallyProtectedBoundarySegment {
        node_ids: [u32; 2],
        source_edge_id: u32,
    },
    OpenBoundaryEdge {
        node_ids: [u32; 2],
        incidence_count: usize,
    },
    NonManifoldBoundaryEdge {
        node_ids: [u32; 2],
        incidence_count: usize,
    },
    ProtectedBoundaryValidation(Box<PlcValidationError>),
}

impl std::fmt::Display for PlcBuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySurface => write!(formatter, "surface mesh has no facets for PLC build"),
            Self::MissingCurveBoundaryValidation => write!(
                formatter,
                "surface mesh has protected source edges but no curve-boundary validation evidence"
            ),
            Self::MissingSurfaceLoopCoverage => write!(
                formatter,
                "surface mesh has protected source edges but no surface loop coverage evidence"
            ),
            Self::InconsistentSurfaceLoopCoverage {
                recovered_face_count,
                surface_source_face_count,
                boundary_loop_count,
                hole_loop_count,
                max_loops_per_face,
                boundary_node_count,
                recovered_source_edge_count,
                protected_source_edge_count,
                boundary_segment_count,
            } => write!(
                formatter,
                "surface loop coverage is inconsistent with PLC input: recovered faces {recovered_face_count}, surface source faces {surface_source_face_count}, boundary loops {boundary_loop_count}, hole loops {hole_loop_count}, max loops per face {max_loops_per_face}, boundary nodes {boundary_node_count}, recovered source edges {recovered_source_edge_count}, protected source edges {protected_source_edge_count}, boundary segments {boundary_segment_count}"
            ),
            Self::InconsistentCadCurveBoundaryProvenance {
                reason,
                recovered_source_edge_count,
                protected_source_edge_count,
                boundary_segment_count,
                edge_report_count,
            } => write!(
                formatter,
                "CAD curve boundary provenance is inconsistent with PLC input ({reason}): recovered source edges {recovered_source_edge_count}, protected source edges {protected_source_edge_count}, boundary segments {boundary_segment_count}, edge reports {edge_report_count}"
            ),
            Self::MissingSurfaceNode {
                triangle_id,
                node_id,
            } => write!(
                formatter,
                "surface triangle {triangle_id} references missing PLC node {node_id}"
            ),
            Self::NonFiniteSurfaceNode { node_id } => {
                write!(
                    formatter,
                    "surface node {node_id} has non-finite coordinates"
                )
            }
            Self::NonFiniteSurfaceTriangle { triangle_id } => write!(
                formatter,
                "surface triangle {triangle_id} has non-finite area or projection evidence"
            ),
            Self::NonPositiveSurfaceTriangleArea { triangle_id } => write!(
                formatter,
                "surface triangle {triangle_id} has non-positive area evidence"
            ),
            Self::InvalidSurfaceEntityId { entity_id } => write!(
                formatter,
                "surface contract entity {:?}:{} is not usable as a PLC numeric topology ID",
                entity_id.stage, entity_id.id
            ),
            Self::DuplicateFacet { element_id } => write!(
                formatter,
                "surface triangle {element_id} duplicates an existing PLC facet"
            ),
            Self::AmbiguousProtectedBoundarySegment {
                node_ids,
                first_source_edge_id,
                second_source_edge_id,
            } => write!(
                formatter,
                "surface boundary segment {}-{} has ambiguous source edges {} and {}",
                node_ids[0], node_ids[1], first_source_edge_id, second_source_edge_id
            ),
            Self::PartiallyProtectedBoundarySegment {
                node_ids,
                source_edge_id,
            } => write!(
                formatter,
                "surface boundary segment {}-{} is only partially owned by protected source edge {}",
                node_ids[0], node_ids[1], source_edge_id
            ),
            Self::OpenBoundaryEdge {
                node_ids,
                incidence_count,
            } => write!(
                formatter,
                "PLC edge {}-{} has incidence {incidence_count}, expected 2",
                node_ids[0], node_ids[1]
            ),
            Self::NonManifoldBoundaryEdge {
                node_ids,
                incidence_count,
            } => write!(
                formatter,
                "PLC edge {}-{} has non-manifold incidence {incidence_count}, expected 2",
                node_ids[0], node_ids[1]
            ),
            Self::ProtectedBoundaryValidation(error) => {
                write!(formatter, "built PLC failed validation: {error}")
            }
        }
    }
}

impl std::error::Error for PlcBuildError {}
