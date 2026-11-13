use super::TiedRankMode;
use runmat_types::ShapeFact;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum ReductionAxis {
    Default,
    All,
    Dimension(usize),
    Dimensions(Vec<usize>),
    Unknown,
}

fn default_axis(dims: &[Option<usize>]) -> Option<usize> {
    for (index, dimension) in dims.iter().enumerate() {
        match dimension {
            Some(size) if *size > 1 => return Some(index),
            None => return None,
            _ => {}
        }
    }
    Some(0)
}

pub(super) fn reduced_shape(
    input: &ShapeFact,
    request_count: Option<usize>,
    axis: ReductionAxis,
) -> ShapeFact {
    let Some(count) = request_count else {
        // An unknown scalar request may be a probability or a count. Its
        // selected extent is dynamic, but a known axis still fixes the rank.
        let rank = if input.element_count() == Some(0) {
            Some(2)
        } else {
            input.rank()
        };
        return match axis {
            ReductionAxis::All => ShapeFact::Ranked { rank: 2 },
            ReductionAxis::Default => {
                rank.map_or(ShapeFact::Unknown, |rank| ShapeFact::Ranked { rank })
            }
            ReductionAxis::Dimension(selected) => {
                rank.zip(selected.checked_add(1))
                    .map_or(ShapeFact::Unknown, |(rank, selected)| ShapeFact::Ranked {
                        rank: rank.max(selected),
                    })
            }
            ReductionAxis::Dimensions(axes) => rank
                .zip(axes.last().and_then(|selected| selected.checked_add(1)))
                .map_or(ShapeFact::Unknown, |(rank, selected)| ShapeFact::Ranked {
                    rank: rank.max(selected),
                }),
            ReductionAxis::Unknown => ShapeFact::Unknown,
        };
    };
    if axis == ReductionAxis::All {
        return ShapeFact::from(vec![Some(if count == 1 { 1 } else { count }), Some(1)]);
    }
    // A ranked fact can carry a caller-selected rank far beyond practical
    // host memory. Preserve that rank symbolically; known_dims() would build
    // a Vec with one slot per dimension.
    if let ShapeFact::Ranked { rank } = input {
        return match axis {
            ReductionAxis::Dimension(selected) => selected
                .checked_add(1)
                .map(|selected_rank| ShapeFact::Ranked {
                    rank: (*rank).max(selected_rank),
                })
                .unwrap_or(ShapeFact::Unknown),
            ReductionAxis::Dimensions(ref axes) => axes
                .last()
                .and_then(|selected| selected.checked_add(1))
                .map(|selected_rank| ShapeFact::Ranked {
                    rank: (*rank).max(selected_rank),
                })
                .unwrap_or(ShapeFact::Unknown),
            ReductionAxis::Default | ReductionAxis::All | ReductionAxis::Unknown => {
                ShapeFact::Unknown
            }
        };
    }
    let Some(mut dims) = input.known_dims() else {
        return ShapeFact::Unknown;
    };
    // The runtime normalizes any empty tensor to a 0-by-1 working shape.
    if input.element_count() == Some(0) {
        dims = vec![Some(0), Some(1)];
    }
    if let ReductionAxis::Dimensions(axes) = &axis {
        let Some(&first) = axes.first() else {
            return ShapeFact::Unknown;
        };
        let Some(rank) = axes.last().and_then(|last| last.checked_add(1)) else {
            return ShapeFact::Unknown;
        };
        if rank > dims.len() {
            return ShapeFact::Ranked { rank };
        }
        for &axis in axes {
            dims[axis] = Some(1);
        }
        dims[first] = Some(count);
        return ShapeFact::from(dims);
    }
    let selected = match axis {
        ReductionAxis::Default => default_axis(&dims),
        ReductionAxis::Dimension(dim) => Some(dim),
        ReductionAxis::All | ReductionAxis::Dimensions(_) | ReductionAxis::Unknown => None,
    };
    let Some(selected) = selected else {
        return ShapeFact::Unknown;
    };
    let Some(rank) = selected.checked_add(1) else {
        return ShapeFact::Unknown;
    };
    if rank > dims.len() {
        return ShapeFact::Ranked { rank };
    }
    dims[selected] = Some(count);
    ShapeFact::from(dims)
}

pub(super) fn tie_adjustment_shape(shape: &ShapeFact, mode: TiedRankMode) -> ShapeFact {
    // Rank alone cannot distinguish vectors from matrices or establish any
    // tie-adjustment extent. Keep it symbolic rather than allocating unknown
    // dimensions in proportion to a caller-provided rank.
    if matches!(shape, ShapeFact::Ranked { .. }) {
        return ShapeFact::Unknown;
    }
    let Some(mut dims) = shape.known_dims() else {
        return ShapeFact::Unknown;
    };
    if dims.iter().any(Option::is_none) {
        return ShapeFact::Unknown;
    }
    let is_vector = dims
        .iter()
        .filter(|dim| dim.is_some_and(|size| size > 1))
        .count()
        <= 1;
    if is_vector {
        return if mode == TiedRankMode::Kendall {
            ShapeFact::from(vec![Some(3), Some(1)])
        } else {
            ShapeFact::Scalar
        };
    }
    let axis = if dims.len() <= 2 { 0 } else { 1 };
    dims[axis] = Some(if mode == TiedRankMode::Kendall { 3 } else { 1 });
    ShapeFact::from(dims)
}
