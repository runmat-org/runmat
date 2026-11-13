use crate::{DimensionFact, InferenceDiagnostic, RangeStepFact, ShapeFact};

fn aligned_dims(shape: &ShapeFact, rank: usize) -> Option<Vec<DimensionFact>> {
    let mut dims = match shape {
        ShapeFact::Unknown => return None,
        ShapeFact::Scalar => vec![DimensionFact::Known(1), DimensionFact::Known(1)],
        ShapeFact::Ranked { rank } => vec![DimensionFact::Unknown; *rank],
        ShapeFact::Shaped { dims } => dims.clone(),
    };
    // A one-dimensional value is a row when implicit expansion reaches a
    // matrix (or higher rank), matching concrete broadcast and Runtime.
    if rank >= 2 && dims.len() == 1 {
        dims.insert(0, DimensionFact::Known(1));
    }
    dims.resize(rank, DimensionFact::Known(1));
    Some(dims)
}

pub fn broadcast_shape(
    left: &ShapeFact,
    right: &ShapeFact,
) -> Result<ShapeFact, InferenceDiagnostic> {
    broadcast_shape_checked(left, right).map_err(BroadcastShapeError::into_diagnostic)
}

/// The same broadcast projection used by public inference, with a typed
/// failure that does not format or allocate a diagnostic on the finite
/// facts-only path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BroadcastShapeError {
    pub left: usize,
    pub right: usize,
    pub dimension: usize,
}

impl BroadcastShapeError {
    pub fn into_diagnostic(self) -> InferenceDiagnostic {
        InferenceDiagnostic::error(
            "RM-TYPE-BROADCAST",
            format!(
                "dimensions {} and {} are not compatible for implicit expansion",
                self.left, self.right
            ),
        )
        .at_dimension(self.dimension)
    }
}

pub fn broadcast_shape_checked(
    left: &ShapeFact,
    right: &ShapeFact,
) -> Result<ShapeFact, BroadcastShapeError> {
    let Some(rank) = left.rank().zip(right.rank()).map(|(a, b)| a.max(b)) else {
        return Ok(ShapeFact::Unknown);
    };
    let Some(left) = aligned_dims(left, rank) else {
        return Ok(ShapeFact::Unknown);
    };
    let Some(right) = aligned_dims(right, rank) else {
        return Ok(ShapeFact::Unknown);
    };
    let mut output = Vec::with_capacity(rank);
    for (dimension, (left, right)) in left.iter().zip(&right).enumerate() {
        let joined = match (left, right) {
            (DimensionFact::Known(a), DimensionFact::Known(b)) if a == b => {
                DimensionFact::Known(*a)
            }
            (DimensionFact::Known(1), other) | (other, DimensionFact::Known(1)) => other.clone(),
            (DimensionFact::Known(a), DimensionFact::Known(b)) => {
                return Err(BroadcastShapeError {
                    left: *a,
                    right: *b,
                    dimension,
                });
            }
            (DimensionFact::Symbolic(a), DimensionFact::Symbolic(b)) if a == b => left.clone(),
            (DimensionFact::Known(value), DimensionFact::Unknown)
            | (DimensionFact::Unknown, DimensionFact::Known(value))
                if *value != 1 =>
            {
                DimensionFact::Known(*value)
            }
            _ => DimensionFact::Unknown,
        };
        output.push(joined);
    }
    Ok(ShapeFact::Shaped { dims: output })
}

pub fn matrix_shape(left: &ShapeFact, right: &ShapeFact) -> Result<ShapeFact, InferenceDiagnostic> {
    if left.element_count() == Some(1) {
        return Ok(right.clone());
    }
    if right.element_count() == Some(1) {
        return Ok(left.clone());
    }
    reject_proven_nonmatrix(left, "left matrix operand")?;
    reject_proven_nonmatrix(right, "right matrix operand")?;
    let Some(left) = matrix_dims(left) else {
        return Ok(ShapeFact::Unknown);
    };
    let Some(right) = matrix_dims(right) else {
        return Ok(ShapeFact::Unknown);
    };
    if let (DimensionFact::Known(a), DimensionFact::Known(b)) = (&left.1, &right.0) {
        if a != b {
            return Err(InferenceDiagnostic::error(
                "RM-TYPE-MATMUL",
                format!("matrix inner dimensions {a} and {b} do not agree"),
            ));
        }
    }
    Ok(ShapeFact::Shaped {
        dims: vec![left.0, right.1],
    })
}

pub fn left_divide_shape(
    left: &ShapeFact,
    right: &ShapeFact,
) -> Result<ShapeFact, InferenceDiagnostic> {
    if left.element_count() == Some(1) {
        return Ok(right.clone());
    }
    reject_proven_nonmatrix(left, "left-division coefficient")?;
    reject_proven_nonmatrix(right, "left-division right-hand side")?;
    let Some(left) = matrix_dims(left) else {
        return Ok(ShapeFact::Unknown);
    };
    let Some(right) = matrix_dims(right) else {
        return Ok(ShapeFact::Unknown);
    };
    if let (DimensionFact::Known(a), DimensionFact::Known(b)) = (&left.0, &right.0) {
        if a != b {
            return Err(InferenceDiagnostic::error(
                "RM-TYPE-MLDIVIDE",
                format!("left-division row dimensions {a} and {b} do not agree"),
            ));
        }
    }
    Ok(ShapeFact::Shaped {
        dims: vec![left.1, right.1],
    })
}

pub fn right_divide_shape(
    left: &ShapeFact,
    right: &ShapeFact,
) -> Result<ShapeFact, InferenceDiagnostic> {
    if right.element_count() == Some(1) {
        return Ok(left.clone());
    }
    reject_proven_nonmatrix(left, "right-division left-hand side")?;
    reject_proven_nonmatrix(right, "right-division coefficient")?;
    let Some(left) = matrix_dims(left) else {
        return Ok(ShapeFact::Unknown);
    };
    let Some(right) = matrix_dims(right) else {
        return Ok(ShapeFact::Unknown);
    };
    if let (DimensionFact::Known(a), DimensionFact::Known(b)) = (&left.1, &right.1) {
        if a != b {
            return Err(InferenceDiagnostic::error(
                "RM-TYPE-MRDIVIDE",
                format!("right-division column dimensions {a} and {b} do not agree"),
            ));
        }
    }
    Ok(ShapeFact::Shaped {
        dims: vec![left.0, right.0],
    })
}

fn matrix_dims(shape: &ShapeFact) -> Option<(DimensionFact, DimensionFact)> {
    let dims = aligned_dims(shape, shape.rank()?.max(2))?;
    if dims
        .iter()
        .skip(2)
        .any(|dim| !matches!(dim, DimensionFact::Known(1)))
    {
        return None;
    }
    Some((dims[0].clone(), dims[1].clone()))
}

pub(crate) fn reject_proven_nonmatrix(
    shape: &ShapeFact,
    description: &str,
) -> Result<(), InferenceDiagnostic> {
    let ShapeFact::Shaped { dims } = shape else {
        return Ok(());
    };
    if dims
        .iter()
        .skip(2)
        .any(|dimension| matches!(dimension, DimensionFact::Known(value) if *value != 1))
    {
        return Err(InferenceDiagnostic::error(
            "RM-TYPE-MATRIX-RANK",
            format!("{description} must be two-dimensional"),
        ));
    }
    Ok(())
}

pub fn transpose_shape(shape: &ShapeFact) -> ShapeFact {
    let Some(rank) = shape.rank() else {
        return ShapeFact::Unknown;
    };
    let Some(mut dims) = aligned_dims(shape, rank.max(2)) else {
        return ShapeFact::Unknown;
    };
    dims.swap(0, 1);
    ShapeFact::Shaped { dims }
}

pub fn reshape_shape(
    source: &ShapeFact,
    mut dimensions: Vec<DimensionFact>,
) -> Result<ShapeFact, InferenceDiagnostic> {
    let unknown = dimensions
        .iter()
        .enumerate()
        .filter_map(|(index, dimension)| {
            matches!(dimension, DimensionFact::Unknown).then_some(index)
        })
        .collect::<Vec<_>>();
    if unknown.len() == 1 {
        if let Some(source_count) = source.element_count() {
            let known_product =
                dimensions
                    .iter()
                    .try_fold(1usize, |product, dimension| match dimension {
                        DimensionFact::Known(value) => product.checked_mul(*value),
                        DimensionFact::Symbolic(_) => None,
                        DimensionFact::Unknown => Some(product),
                    });
            if let Some(known_product) = known_product {
                if known_product != 0 && source_count % known_product == 0 {
                    dimensions[unknown[0]] = DimensionFact::Known(source_count / known_product);
                }
            }
        }
    }
    let output = ShapeFact::Shaped { dims: dimensions };
    if let (Some(source_count), Some(output_count)) =
        (source.element_count(), output.element_count())
    {
        if source_count != output_count {
            return Err(InferenceDiagnostic::error(
                "RM-TYPE-RESHAPE",
                format!(
                    "reshape cannot change element count from {source_count} to {output_count}"
                ),
            ));
        }
    }
    Ok(output)
}

pub fn permute_shape(
    source: &ShapeFact,
    one_based_order: &[usize],
) -> Result<ShapeFact, InferenceDiagnostic> {
    if source
        .rank()
        .is_some_and(|rank| one_based_order.len() < rank)
    {
        return Err(InferenceDiagnostic::error(
            "RM-TYPE-PERMUTE",
            format!(
                "permutation has only {} dimensions but source rank is {}",
                one_based_order.len(),
                source.rank().expect("known rank")
            ),
        ));
    }
    let output_rank = one_based_order.len();
    let mut seen = vec![false; output_rank];
    for dimension in one_based_order {
        if *dimension == 0 || *dimension > output_rank || seen[*dimension - 1] {
            return Err(InferenceDiagnostic::error(
                "RM-TYPE-PERMUTE",
                "permutation must contain each source dimension exactly once",
            ));
        }
        seen[*dimension - 1] = true;
    }
    let Some(dims) = structural_dims(source, output_rank) else {
        return Ok(ShapeFact::Unknown);
    };
    Ok(ShapeFact::Shaped {
        dims: one_based_order
            .iter()
            .map(|dimension| dims[*dimension - 1].clone())
            .collect(),
    })
}

/// Remove only proven singleton axes. An unknown higher-rank extent may be
/// one, so its output rank cannot be asserted merely by retaining that axis.
pub fn squeeze_shape(source: &ShapeFact) -> ShapeFact {
    let Some(dims) = source.known_dims() else {
        return source.clone();
    };
    if dims.len() <= 2 {
        return source.clone();
    }
    if dims.iter().any(Option::is_none) {
        return ShapeFact::Unknown;
    }
    let mut dims = dims
        .into_iter()
        .filter(|dim| *dim != Some(1))
        .collect::<Vec<_>>();
    if dims.is_empty() {
        dims = vec![Some(1), Some(1)];
    } else if dims.len() == 1 {
        dims.push(Some(1));
    }
    ShapeFact::from(dims)
}

pub fn repmat_shape(source: &ShapeFact, repetitions: &[DimensionFact]) -> ShapeFact {
    let Some(source_rank) = source.rank() else {
        // Repetitions provide a lower bound, not an upper bound on source rank.
        return ShapeFact::Unknown;
    };
    let rank = source_rank.max(repetitions.len()).max(2);
    let Some(source) = structural_dims(source, rank) else {
        return ShapeFact::Ranked { rank };
    };
    let mut repetitions = repetitions.to_vec();
    repetitions.resize(rank, DimensionFact::Known(1));
    ShapeFact::Shaped {
        dims: source
            .iter()
            .zip(repetitions)
            .map(|(dimension, repetition)| match (dimension, repetition) {
                (DimensionFact::Known(0), _) | (_, DimensionFact::Known(0)) => {
                    DimensionFact::Known(0)
                }
                (DimensionFact::Known(a), DimensionFact::Known(b)) => a
                    .checked_mul(b)
                    .map_or(DimensionFact::Unknown, DimensionFact::Known),
                _ => DimensionFact::Unknown,
            })
            .collect(),
    }
}

// Structural transforms pad trailing axes, unlike implicit expansion's
// leading-axis promotion of one-dimensional operands.
fn structural_dims(source: &ShapeFact, rank: usize) -> Option<Vec<DimensionFact>> {
    let mut dimensions = match source {
        ShapeFact::Unknown => return None,
        ShapeFact::Scalar => vec![DimensionFact::Known(1); 2],
        ShapeFact::Ranked { rank } => vec![DimensionFact::Unknown; *rank],
        ShapeFact::Shaped { dims } => dims.clone(),
    };
    dimensions.resize(rank, DimensionFact::Known(1));
    Some(dimensions)
}

pub fn reduction_shape(source: &ShapeFact, one_based_dimension: Option<usize>) -> ShapeFact {
    let Some(rank) = source.rank() else {
        return ShapeFact::Unknown;
    };
    let Some(mut dims) = aligned_dims(source, rank) else {
        return ShapeFact::Unknown;
    };
    let dimension = one_based_dimension
        .and_then(|dimension| dimension.checked_sub(1))
        .or_else(|| {
            dims.iter()
                .position(|dimension| !matches!(dimension, DimensionFact::Known(1)))
        })
        .unwrap_or(0);
    if dimension >= dims.len() {
        dims.resize(dimension + 1, DimensionFact::Known(1));
    }
    dims[dimension] = DimensionFact::Known(1);
    ShapeFact::Shaped { dims }
}

pub fn range_shape(start: Option<f64>, step: Option<f64>, end: Option<f64>) -> ShapeFact {
    range_shape_with_step(
        start,
        step.map_or(RangeStepFact::Implicit, RangeStepFact::Known),
        end,
    )
}

fn range_shape_with_step(start: Option<f64>, step: RangeStepFact, end: Option<f64>) -> ShapeFact {
    let step = match step {
        RangeStepFact::Implicit => Some(1.0),
        RangeStepFact::Known(step) => Some(step),
        RangeStepFact::Unknown => None,
    };
    let length = match (start, step, end) {
        (Some(start), Some(step), Some(end)) if step != 0.0 => {
            let raw = (end - start) / step;
            if !raw.is_finite() || raw < 0.0 {
                Some(0)
            } else {
                let rounded = raw.round();
                let steps = if (raw - rounded).abs() <= 1e-9 {
                    rounded
                } else {
                    raw.floor()
                };
                Some(steps as usize + 1)
            }
        }
        _ => None,
    };
    ShapeFact::Shaped {
        dims: vec![
            DimensionFact::Known(1),
            length.map_or(DimensionFact::Unknown, DimensionFact::Known),
        ],
    }
}

pub fn infer_range(
    start: Option<f64>,
    step: RangeStepFact,
    end: Option<f64>,
) -> crate::FactInference {
    if step == RangeStepFact::Known(0.0) {
        return crate::FactInference {
            fact: crate::ValueFact::unknown(crate::DynamicReason::UnsupportedRepresentation),
            diagnostics: vec![crate::InferenceDiagnostic::error(
                "RM-TYPE-RANGE-STEP",
                "range step cannot be zero",
            )],
        };
    }
    crate::FactInference::exact(crate::ValueFact::proven(
        crate::ValueKindFact::Numeric(crate::NumericFact {
            class: crate::NumericClass::Double,
            domain: crate::NumericDomain::Real,
        }),
        range_shape_with_step(start, step, end),
        crate::StorageFact::Dense,
    ))
}

#[cfg(test)]
mod broadcast_tests {
    use super::*;

    fn shaped(dims: &[usize]) -> ShapeFact {
        ShapeFact::Shaped {
            dims: dims.iter().copied().map(DimensionFact::Known).collect(),
        }
    }

    #[test]
    fn typed_broadcast_failure_renders_the_existing_public_diagnostic() {
        let left = shaped(&[2, 3]);
        let right = shaped(&[2, 4]);
        let typed = broadcast_shape_checked(&left, &right).unwrap_err();
        assert_eq!(typed.left, 3);
        assert_eq!(typed.right, 4);
        assert_eq!(typed.dimension, 1);
        assert_eq!(
            typed.into_diagnostic(),
            broadcast_shape(&left, &right).unwrap_err()
        );
    }

    #[test]
    fn vector_uses_row_shorthand_only_when_broadcast_rank_reaches_two() {
        assert_eq!(
            broadcast_shape(&shaped(&[4]), &shaped(&[4])).unwrap(),
            shaped(&[4])
        );
        assert_eq!(
            broadcast_shape(&shaped(&[4]), &ShapeFact::Scalar).unwrap(),
            shaped(&[1, 4])
        );
        assert_eq!(
            broadcast_shape(&ShapeFact::Scalar, &shaped(&[4])).unwrap(),
            shaped(&[1, 4])
        );
        assert_eq!(
            broadcast_shape(&shaped(&[4]), &shaped(&[2, 4])).unwrap(),
            shaped(&[2, 4])
        );
        assert_eq!(
            broadcast_shape(&shaped(&[2, 4]), &shaped(&[4])).unwrap(),
            shaped(&[2, 4])
        );
        assert_eq!(
            broadcast_shape(&shaped(&[4]), &shaped(&[4, 1])).unwrap(),
            shaped(&[4, 4])
        );
        assert_eq!(
            broadcast_shape(&shaped(&[4, 1]), &shaped(&[4])).unwrap(),
            shaped(&[4, 4])
        );
        assert_eq!(
            broadcast_shape(&shaped(&[4, 1]), &ShapeFact::Scalar).unwrap(),
            shaped(&[4, 1])
        );
        assert!(broadcast_shape(&shaped(&[4]), &shaped(&[2, 5])).is_err());
        // Structural transforms pad trailing axes rather than broadcasting a
        // one-dimensional source as a row.
        assert_eq!(
            permute_shape(&shaped(&[4]), &[2, 1]).unwrap(),
            shaped(&[1, 4])
        );
        assert_eq!(
            repmat_shape(&shaped(&[4]), &[DimensionFact::Known(2)]),
            shaped(&[8, 1])
        );
        assert_eq!(
            repmat_shape(&shaped(&[0, 3]), &[DimensionFact::Unknown]),
            ShapeFact::from(vec![Some(0), Some(3)])
        );
        assert!(permute_shape(&ShapeFact::Unknown, &[1, 1]).is_err());
        assert_eq!(
            repmat_shape(
                &ShapeFact::Unknown,
                &[DimensionFact::Known(2), DimensionFact::Known(3)]
            ),
            ShapeFact::Unknown
        );
        assert_eq!(squeeze_shape(&shaped(&[1, 4, 1])), shaped(&[4, 1]));
    }

    #[test]
    fn vector_row_shorthand_preserves_higher_rank_and_empty_axes() {
        assert_eq!(
            broadcast_shape(&shaped(&[4]), &shaped(&[2, 1, 3])).unwrap(),
            shaped(&[2, 4, 3])
        );
        assert_eq!(
            broadcast_shape(&shaped(&[0]), &ShapeFact::Scalar).unwrap(),
            shaped(&[1, 0])
        );
        assert_eq!(
            broadcast_shape(&shaped(&[0]), &shaped(&[2, 0])).unwrap(),
            shaped(&[2, 0])
        );
        assert_eq!(
            broadcast_shape(&ShapeFact::Ranked { rank: 1 }, &shaped(&[2, 4])).unwrap(),
            shaped(&[2, 4])
        );
    }
}
