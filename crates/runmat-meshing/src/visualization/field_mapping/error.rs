use runmat_meshing_core::contracts::AnalysisFieldTopologyLocation;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldMappingError {
    FieldTopologyMissing {
        topology_id: String,
        location: AnalysisFieldTopologyLocation,
        element_kind: Option<String>,
    },
    FieldTopologyCountMismatch {
        topology_id: String,
        location: AnalysisFieldTopologyLocation,
        expected_entity_count: usize,
        actual_entity_count: usize,
    },
    FieldTopologyElementKindMismatch {
        topology_id: String,
        location: AnalysisFieldTopologyLocation,
        expected_element_kind: String,
        actual_element_kind: Option<String>,
    },
    ElementFieldLengthMismatch {
        element_value_count: usize,
        volume_element_count: usize,
    },
    NodeVectorFieldLengthMismatch {
        node_value_count: usize,
        node_count: usize,
    },
    NonFiniteElementValue {
        element_index: usize,
    },
    NonFiniteNodeVectorValue {
        node_index: usize,
        component_index: usize,
    },
    BoundaryFaceMissingAdjacentVolume {
        face_id: String,
    },
    BoundaryFaceReferencesUnknownVolume {
        face_id: String,
        volume_element_id: String,
    },
    BoundaryFaceReferencesUnknownNode {
        face_id: String,
        node_id: u32,
    },
    BoundaryFaceHasNoNodes {
        face_id: String,
    },
    BoundaryEdgeReferencesUnknownNode {
        edge_id: String,
        node_id: u32,
    },
}

impl std::fmt::Display for FieldMappingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FieldTopologyMissing {
                topology_id,
                location,
                element_kind,
            } => write!(
                formatter,
                "mesh field topology descriptor is missing: topology_id={topology_id} location={location:?} element_kind={}",
                element_kind.as_deref().unwrap_or("none")
            ),
            Self::FieldTopologyCountMismatch {
                topology_id,
                location,
                expected_entity_count,
                actual_entity_count,
            } => write!(
                formatter,
                "mesh field topology count mismatch: topology_id={topology_id} location={location:?} expected_entity_count={expected_entity_count} actual_entity_count={actual_entity_count}"
            ),
            Self::FieldTopologyElementKindMismatch {
                topology_id,
                location,
                expected_element_kind,
                actual_element_kind,
            } => write!(
                formatter,
                "mesh field topology element kind mismatch: topology_id={topology_id} location={location:?} expected_element_kind={expected_element_kind} actual_element_kind={}",
                actual_element_kind.as_deref().unwrap_or("none")
            ),
            Self::ElementFieldLengthMismatch {
                element_value_count,
                volume_element_count,
            } => write!(
                formatter,
                "element scalar field length {element_value_count} does not match volume element count {volume_element_count}"
            ),
            Self::NodeVectorFieldLengthMismatch {
                node_value_count,
                node_count,
            } => write!(
                formatter,
                "node vector field length {node_value_count} does not match mesh node count {node_count}"
            ),
            Self::NonFiniteElementValue { element_index } => {
                write!(formatter, "element scalar field value {element_index} is not finite")
            }
            Self::NonFiniteNodeVectorValue {
                node_index,
                component_index,
            } => write!(
                formatter,
                "node vector field value {node_index} component {component_index} is not finite"
            ),
            Self::BoundaryFaceMissingAdjacentVolume { face_id } => {
                write!(formatter, "boundary face {face_id} has no adjacent volume element")
            }
            Self::BoundaryFaceReferencesUnknownVolume {
                face_id,
                volume_element_id,
            } => write!(
                formatter,
                "boundary face {face_id} references unknown volume element {volume_element_id}"
            ),
            Self::BoundaryFaceReferencesUnknownNode { face_id, node_id } => write!(
                formatter,
                "boundary face {face_id} references unknown node {node_id}"
            ),
            Self::BoundaryFaceHasNoNodes { face_id } => {
                write!(formatter, "boundary face {face_id} has no nodes")
            }
            Self::BoundaryEdgeReferencesUnknownNode { edge_id, node_id } => write!(
                formatter,
                "boundary edge {edge_id} references unknown node {node_id}"
            ),
        }
    }
}

impl std::error::Error for FieldMappingError {}
