use serde::{Deserialize, Serialize};

use super::quantity::Position;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferencePoint {
    pub reference_point_id: String,
    pub position: Position,
}
