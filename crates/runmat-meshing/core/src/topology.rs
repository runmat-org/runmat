use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VolumeElementKind {
    Tet4,
    Tet10,
    Hex8,
}

impl VolumeElementKind {
    pub const fn node_count(self) -> usize {
        match self {
            Self::Tet4 => 4,
            Self::Tet10 => 10,
            Self::Hex8 => 8,
        }
    }

    pub const fn is_supported_for_solid_solve(self) -> bool {
        matches!(self, Self::Tet4)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryElementKind {
    Tri3,
    Tri6,
    Quad4,
}

impl BoundaryElementKind {
    pub const fn node_count(self) -> usize {
        match self {
            Self::Tri3 => 3,
            Self::Tri6 => 6,
            Self::Quad4 => 4,
        }
    }

    pub const fn is_supported_for_boundary_mapping(self) -> bool {
        matches!(self, Self::Tri3)
    }
}
