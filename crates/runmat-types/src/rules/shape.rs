use crate::{DimensionFact, InferenceDiagnostic, RangeStepFact, ShapeFact};

fn aligned_dims(shape: &ShapeFact, rank: usize) -> Option<Vec<DimensionFact>> {
    let mut dims = match shape {
        ShapeFact::Unknown => return None,
        ShapeFact::Scalar => vec![DimensionFact::Known(1), DimensionFact::Known(1)],
        ShapeFact::Ranked { rank } => vec![DimensionFact::Unknown; *rank],
        ShapeFact::Shaped { dims } => dims.clone(),
    };
    dims.resize(rank, DimensionFact::Known(1));
    Some(dims)
}

pub fn broadcast_shape(
    left: &ShapeFact,
    right: &ShapeFact,
) -> Result<ShapeFact, InferenceDiagnostic> {
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
                return Err(InferenceDiagnostic::error(
                    "RM-TYPE-BROADCAST",
                    format!(
                        "dimensions {} and {} are not compatible for implicit expansion",
                        a, b
                    ),
                )
                .at_dimension(dimension));
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
    let Some(rank) = source.rank() else {
        return Ok(ShapeFact::Unknown);
    };
    if one_based_order.len() < rank {
        return Err(InferenceDiagnostic::error(
            "RM-TYPE-PERMUTE",
            format!(
                "permutation has only {} dimensions but source rank is {rank}",
                one_based_order.len()
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
    let Some(dims) = aligned_dims(source, output_rank) else {
        return Ok(ShapeFact::Unknown);
    };
    Ok(ShapeFact::Shaped {
        dims: one_based_order
            .iter()
            .map(|dimension| dims[*dimension - 1].clone())
            .collect(),
    })
}

pub fn repmat_shape(source: &ShapeFact, repetitions: &[DimensionFact]) -> ShapeFact {
    let rank = source.rank().unwrap_or(0).max(repetitions.len()).max(2);
    let Some(source) = aligned_dims(source, rank) else {
        return ShapeFact::Ranked { rank };
    };
    let mut repetitions = repetitions.to_vec();
    repetitions.resize(rank, DimensionFact::Known(1));
    ShapeFact::Shaped {
        dims: source
            .iter()
            .zip(repetitions)
            .map(|(dimension, repetition)| match (dimension, repetition) {
                (DimensionFact::Known(a), DimensionFact::Known(b)) => a
                    .checked_mul(b)
                    .map_or(DimensionFact::Unknown, DimensionFact::Known),
                _ => DimensionFact::Unknown,
            })
            .collect(),
    }
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
