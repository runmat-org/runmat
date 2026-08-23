use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VolumeElementKind {
    Tetrahedron4,
    Tetrahedron10,
}

impl VolumeElementKind {
    pub const fn node_count(self) -> usize {
        match self {
            Self::Tetrahedron4 => 4,
            Self::Tetrahedron10 => 10,
        }
    }

    pub const fn is_supported_for_solid_solve(self) -> bool {
        matches!(self, Self::Tetrahedron4)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryElementKind {
    Tri3,
    Tri6,
}

impl BoundaryElementKind {
    pub const fn node_count(self) -> usize {
        match self {
            Self::Tri3 => 3,
            Self::Tri6 => 6,
        }
    }

    pub const fn is_supported_for_boundary_mapping(self) -> bool {
        matches!(self, Self::Tri3)
    }
}
