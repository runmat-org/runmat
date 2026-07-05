use crate::validate::PlcValidationError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlcBuildError {
    EmptySurface,
    MissingCurveBoundaryValidation,
    MissingSurfaceLoopCoverage,
    InconsistentSurfaceLoopCoverage {
        recovered_face_count: usize,
        surface_source_face_count: usize,
        recovered_source_edge_count: usize,
        protected_source_edge_count: usize,
    },
    MissingSurfaceNode {
        element_id: u32,
        node_id: u32,
    },
    NonFiniteSurfaceNode {
        node_id: u32,
    },
    NonFiniteSurfaceElement {
        element_id: u32,
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
    ProtectedBoundaryValidation(PlcValidationError),
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
                recovered_source_edge_count,
                protected_source_edge_count,
            } => write!(
                formatter,
                "surface loop coverage is inconsistent with PLC input: recovered faces {recovered_face_count}, surface source faces {surface_source_face_count}, recovered source edges {recovered_source_edge_count}, protected source edges {protected_source_edge_count}"
            ),
            Self::MissingSurfaceNode {
                element_id,
                node_id,
            } => write!(
                formatter,
                "surface element {element_id} references missing PLC node {node_id}"
            ),
            Self::NonFiniteSurfaceNode { node_id } => {
                write!(
                    formatter,
                    "surface node {node_id} has non-finite coordinates"
                )
            }
            Self::NonFiniteSurfaceElement { element_id } => write!(
                formatter,
                "surface element {element_id} has non-finite area or projection evidence"
            ),
            Self::DuplicateFacet { element_id } => write!(
                formatter,
                "surface element {element_id} duplicates an existing PLC facet"
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
