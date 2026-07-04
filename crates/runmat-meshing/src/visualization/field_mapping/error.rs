#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldMappingError {
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
