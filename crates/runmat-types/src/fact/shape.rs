use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShapeFact {
    Unknown,
    Scalar,
    Ranked { rank: usize },
    Shaped { dims: Vec<DimensionFact> },
}

impl ShapeFact {
    pub fn rank(&self) -> Option<usize> {
        match self {
            Self::Unknown => None,
            Self::Scalar => Some(2),
            Self::Ranked { rank } => Some(*rank),
            Self::Shaped { dims } => Some(dims.len()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DimensionFact {
    Known(usize),
    Symbolic(DimensionSymbol),
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DimensionSymbol(pub String);
