use crate::validate::PlcValidationError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlcBuildError {
    EmptySurface,
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
