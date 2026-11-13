use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TopologyApplicability {
    NotApplicable,
    AnyGeometry,
    Vertex,
    Edge,
    Face,
    Solid,
    MeshNode,
    MeshElement,
    Boundary,
    Region,
    ConstraintGroup,
}
