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

    pub fn element_count(&self) -> Option<usize> {
        match self {
            Self::Unknown | Self::Ranked { .. } => None,
            Self::Scalar => Some(1),
            Self::Shaped { dims } => dims.iter().try_fold(1usize, |count, dim| {
                let DimensionFact::Known(value) = dim else {
                    return None;
                };
                count.checked_mul(*value)
            }),
        }
    }

    pub fn known_dims(&self) -> Option<Vec<Option<usize>>> {
        match self {
            Self::Unknown => None,
            Self::Scalar => Some(vec![Some(1), Some(1)]),
            Self::Ranked { rank } => Some(vec![None; *rank]),
            Self::Shaped { dims } => Some(
                dims.iter()
                    .map(|dim| match dim {
                        DimensionFact::Known(value) => Some(*value),
                        DimensionFact::Symbolic(_) | DimensionFact::Unknown => None,
                    })
                    .collect(),
            ),
        }
    }

    /// Whether two proven shapes denote the same MATLAB array geometry.
    /// Scalar and explicit all-one shapes are equivalent, and trailing
    /// singleton dimensions beyond MATLAB's minimum rank of two are not
    /// semantically distinct.
    pub fn is_proven_equivalent(&self, other: &Self) -> bool {
        if self == other {
            return true;
        }
        let (Some(mut left), Some(mut right)) = (self.known_dims(), other.known_dims()) else {
            return false;
        };
        if left.iter().chain(&right).any(Option::is_none) {
            return false;
        }
        trim_trailing_singletons(&mut left);
        trim_trailing_singletons(&mut right);
        left == right
    }
}

fn trim_trailing_singletons(dimensions: &mut Vec<Option<usize>>) {
    while dimensions.len() > 2 && dimensions.last() == Some(&Some(1)) {
        dimensions.pop();
    }
    dimensions.resize(2, Some(1));
}

impl From<Vec<Option<usize>>> for ShapeFact {
    fn from(dims: Vec<Option<usize>>) -> Self {
        Self::Shaped {
            dims: dims
                .into_iter()
                .map(|dim| dim.map_or(DimensionFact::Unknown, DimensionFact::Known))
                .collect(),
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
