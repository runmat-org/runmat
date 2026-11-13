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

    /// True only when every completion of the known dimensions has an element
    /// count other than one. This stays allocation-free for symbolic ranks.
    pub fn is_definitely_nonscalar(&self) -> bool {
        matches!(self, Self::Shaped { dims } if dims.iter().any(|dim| matches!(dim, DimensionFact::Known(extent) if *extent != 1)))
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
    if dimensions.len() < 2 {
        dimensions.resize(2, Some(1));
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_shapes_can_be_definitely_nonscalar_without_known_element_count() {
        for extent in [0, 2] {
            let shape = ShapeFact::Shaped {
                dims: vec![DimensionFact::Known(extent), DimensionFact::Unknown],
            };
            assert_eq!(shape.element_count(), None);
            assert!(shape.is_definitely_nonscalar());
        }
        for shape in [
            ShapeFact::Unknown,
            ShapeFact::Scalar,
            ShapeFact::Ranked { rank: usize::MAX },
            ShapeFact::Shaped {
                dims: vec![DimensionFact::Known(1), DimensionFact::Unknown],
            },
            ShapeFact::Shaped {
                dims: vec![DimensionFact::Known(1), DimensionFact::Known(1)],
            },
        ] {
            assert!(!shape.is_definitely_nonscalar(), "{shape:?}");
        }
    }

    #[test]
    fn shape_equivalence_discards_only_trailing_singletons() {
        let matrix = ShapeFact::from(vec![Some(2), Some(1)]);
        let nd = ShapeFact::from(vec![Some(2), Some(1), Some(3)]);
        let trailing = ShapeFact::from(vec![Some(2), Some(1), Some(3), Some(1)]);
        assert!(!matrix.is_proven_equivalent(&nd));
        assert!(nd.is_proven_equivalent(&trailing));
    }
}
