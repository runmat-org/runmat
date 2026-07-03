use runmat_meshing_core::contracts::{PlcValidationSummary, TopologyEntityId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlcValidationError {
    ValidationSummaryNotVolumeReady {
        summary: PlcValidationSummary,
    },
    EmptyNodes,
    EmptyFacets,
    DuplicateNode {
        node_id: TopologyEntityId,
    },
    NonFiniteNode {
        node_id: TopologyEntityId,
    },
    FacetReferencesUnknownNode {
        facet_id: TopologyEntityId,
        node_id: TopologyEntityId,
    },
    FacetHasRepeatedNode {
        facet_id: TopologyEntityId,
    },
    FacetHasEmptyMaterialInterfaceId {
        facet_id: TopologyEntityId,
    },
    FacetHasRepeatedMaterialInterfaceId {
        facet_id: TopologyEntityId,
        material_interface_id: String,
    },
    ProtectedEdgeReferencesUnknownNode {
        edge_id: TopologyEntityId,
        node_id: TopologyEntityId,
    },
    ProtectedEdgeHasRepeatedNode {
        edge_id: TopologyEntityId,
    },
    UnreferencedNode {
        node_id: TopologyEntityId,
    },
    ProtectedEdgeNotOnBoundary {
        edge_id: TopologyEntityId,
        node_ids: [TopologyEntityId; 2],
    },
    OpenBoundaryEdge {
        node_ids: [TopologyEntityId; 2],
        incidence_count: usize,
    },
    NonManifoldBoundaryEdge {
        node_ids: [TopologyEntityId; 2],
        incidence_count: usize,
    },
    InconsistentBoundaryEdgeOrientation {
        node_ids: [TopologyEntityId; 2],
    },
}

impl std::fmt::Display for PlcValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ValidationSummaryNotVolumeReady { .. } => {
                write!(formatter, "PLC validation summary is not volume-ready")
            }
            Self::EmptyNodes => write!(formatter, "PLC has no nodes"),
            Self::EmptyFacets => write!(formatter, "PLC has no facets"),
            Self::DuplicateNode { node_id } => {
                write!(formatter, "PLC has duplicate node {}", node_id.id)
            }
            Self::NonFiniteNode { node_id } => {
                write!(
                    formatter,
                    "PLC node {} has non-finite coordinates",
                    node_id.id
                )
            }
            Self::FacetReferencesUnknownNode { facet_id, node_id } => write!(
                formatter,
                "PLC facet {} references unknown node {}",
                facet_id.id, node_id.id
            ),
            Self::FacetHasRepeatedNode { facet_id } => {
                write!(formatter, "PLC facet {} repeats a node", facet_id.id)
            }
            Self::FacetHasEmptyMaterialInterfaceId { facet_id } => write!(
                formatter,
                "PLC facet {} has an empty material interface id",
                facet_id.id
            ),
            Self::FacetHasRepeatedMaterialInterfaceId {
                facet_id,
                material_interface_id,
            } => write!(
                formatter,
                "PLC facet {} repeats material interface id {material_interface_id}",
                facet_id.id
            ),
            Self::ProtectedEdgeReferencesUnknownNode { edge_id, node_id } => write!(
                formatter,
                "PLC protected edge {} references unknown node {}",
                edge_id.id, node_id.id
            ),
            Self::ProtectedEdgeHasRepeatedNode { edge_id } => {
                write!(
                    formatter,
                    "PLC protected edge {} repeats a node",
                    edge_id.id
                )
            }
            Self::UnreferencedNode { node_id } => {
                write!(
                    formatter,
                    "PLC node {} is not referenced by any facet",
                    node_id.id
                )
            }
            Self::ProtectedEdgeNotOnBoundary { edge_id, node_ids } => write!(
                formatter,
                "PLC protected edge {} references non-boundary edge {}-{}",
                edge_id.id, node_ids[0].id, node_ids[1].id
            ),
            Self::OpenBoundaryEdge {
                node_ids,
                incidence_count,
            } => write!(
                formatter,
                "PLC edge {}-{} has incidence {incidence_count}, expected 2",
                node_ids[0].id, node_ids[1].id
            ),
            Self::NonManifoldBoundaryEdge {
                node_ids,
                incidence_count,
            } => write!(
                formatter,
                "PLC edge {}-{} has non-manifold incidence {incidence_count}, expected 2",
                node_ids[0].id, node_ids[1].id
            ),
            Self::InconsistentBoundaryEdgeOrientation { node_ids } => write!(
                formatter,
                "PLC edge {}-{} is not oppositely oriented by its incident facets",
                node_ids[0].id, node_ids[1].id
            ),
        }
    }
}

impl std::error::Error for PlcValidationError {}
