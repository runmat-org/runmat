use super::*;
/// Scalar/default/all selections own no temporary dimension vector.
pub(super) enum ResolvedDims {
    One(Option<usize>),
    All(usize),
    Many(Vec<usize>),
}
impl ResolvedDims {
    pub fn resolve(shape: &[usize], selection: &DimSelection) -> Self {
        match selection {
            DimSelection::Auto => Self::One(if shape.is_empty() {
                None
            } else {
                Some(shape.iter().position(|&n| n != 1).unwrap_or(0))
            }),
            DimSelection::Dim(axis) => Self::One(axis.checked_sub(1).filter(|i| *i < shape.len())),
            DimSelection::All => Self::All(shape.len()),
            DimSelection::Vec(values) => {
                let mut axes = values
                    .iter()
                    .filter_map(|i| i.checked_sub(1).filter(|i| *i < shape.len()))
                    .collect::<Vec<_>>();
                axes.sort_unstable();
                axes.dedup();
                Self::Many(axes)
            }
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        let len = self.len();
        (0..len).map(move |index| match self {
            Self::One(Some(axis)) => *axis,
            Self::All(_) => index,
            Self::Many(values) => values[index],
            Self::One(None) => unreachable!("empty selection"),
        })
    }
    pub fn len(&self) -> usize {
        match self {
            Self::One(value) => usize::from(value.is_some()),
            Self::All(n) => *n,
            Self::Many(values) => values.len(),
        }
    }
    pub fn contains(&self, axis: usize) -> bool {
        self.iter().any(|i| i == axis)
    }
}
