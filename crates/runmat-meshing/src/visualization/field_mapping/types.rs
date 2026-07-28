use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryFaceScalarValue {
    pub face_id: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryNodeVectorValue {
    pub node_id: u32,
    pub value: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryFaceVectorValue {
    pub face_id: String,
    pub value: [f64; 3],
}
