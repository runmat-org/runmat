//! Literal decoding and metadata admission share the structural domain owner.
use super::*;

pub(super) fn controls_shape(
    request: &CallRequest,
    source: &ValueFact,
    operation: ArrayShapeTransform,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    let permutation = matches!(
        operation,
        ArrayShapeTransform::Permute | ArrayShapeTransform::InversePermute
    );
    for (index, control) in request.arguments.iter().enumerate().skip(1) {
        if !shape_transform_control_supported(control, operation, request.arguments.len() > 2) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SHAPE-CONTROL",
                "unsupported structural control class, shape or residency",
                index,
            ));
        }
    }
    let mut dimensions = control_dimensions(request, operation, diagnostics);
    if permutation {
        let mut source_shape = source.shape.clone();
        if matches!(source.kind, ValueKindFact::Struct(_)) {
            if let Some(mut dims) = source_shape.known_dims() {
                while dims.len() > 2 && dims.last() == Some(&Some(1)) {
                    dims.pop();
                }
                source_shape = ShapeFact::from(dims);
            }
        }
        let Some(dimensions) = dimensions else {
            return ShapeFact::Unknown;
        };
        let rank = dimensions.len();
        if rank == 0 {
            diagnostics.push(argument_error(
                "RM-CATALOG-SHAPE-ORDER",
                "order must not be empty",
                1,
            ));
        }
        let order = dimensions
            .iter()
            .map(|dimension| match dimension {
                DimensionFact::Known(value) => Some(*value),
                _ => None,
            })
            .collect::<Option<Vec<_>>>();
        if let Some(mut order) = order {
            if operation == ArrayShapeTransform::InversePermute {
                let mut inverse = vec![0; order.len()];
                for (index, &dimension) in order.iter().enumerate() {
                    if dimension == 0 || dimension > order.len() || inverse[dimension - 1] != 0 {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-SHAPE-ORDER",
                            "order must be a permutation",
                            1,
                        ));
                        return ShapeFact::Unknown;
                    }
                    inverse[dimension - 1] = index + 1;
                }
                order = inverse;
            }
            return match permute_shape(&source_shape, &order) {
                Ok(shape) => shape,
                Err(error) => {
                    diagnostics.push(error);
                    ShapeFact::Unknown
                }
            };
        }
        if source_shape
            .rank()
            .is_some_and(|source_rank| rank < source_rank)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SHAPE-RANK",
                "order must cover the source rank",
                1,
            ));
        }
        return ShapeFact::Ranked { rank };
    }
    let Some(mut dimensions) = dimensions.take() else {
        return ShapeFact::Unknown;
    };
    if dimensions.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-SHAPE-CONTROL",
            "size controls must not be empty",
            1,
        ));
        return ShapeFact::Unknown;
    }
    if operation == ArrayShapeTransform::Tile {
        if matches!(source.kind, ValueKindFact::Struct(_)) {
            // The existing structure owner pads omitted repetitions with one,
            // including its scalar-structure construction form.
            return infer_repmat(source, &dimensions).fact.shape;
        }
        if matches!(
            source.kind,
            ValueKindFact::Character | ValueKindFact::Cell(_)
        ) {
            if dimensions
                .iter()
                .skip(2)
                .any(|dimension| matches!(dimension, DimensionFact::Known(value) if *value > 1))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SHAPE-TILE",
                    "character and cell tiling supports two dimensions",
                    1,
                ));
            }
            let first = dimensions[0].clone();
            let second = dimensions.get(1).cloned().unwrap_or_else(|| first.clone());
            let mut base = source.clone();
            base.shape =
                source
                    .shape
                    .known_dims()
                    .map_or(ShapeFact::Ranked { rank: 2 }, |mut dims| {
                        dims.resize(2, Some(1));
                        dims.truncate(2);
                        ShapeFact::from(dims)
                    });
            return infer_repmat(&base, &[first, second]).fact.shape;
        }
        if dimensions.len() == 1 {
            dimensions.resize(
                source.shape.rank().unwrap_or(2).max(2),
                dimensions[0].clone(),
            );
        }
        return infer_repmat(source, &dimensions).fact.shape;
    }
    let inferred = infer_reshape(source, dimensions);
    diagnostics.extend(inferred.diagnostics);
    let mut shape = inferred.fact.shape;
    // Char/cell/structure array carriers preserve their existing two-D minimum.
    if matches!(
        source.kind,
        ValueKindFact::Character | ValueKindFact::Cell(_) | ValueKindFact::Struct(_)
    ) {
        if let Some(dims) = shape.known_dims() {
            if dims.len() == 1 {
                shape = ShapeFact::from(vec![dims[0], Some(1)]);
            }
        }
    }
    shape
}

fn control_dimensions(
    request: &CallRequest,
    operation: ArrayShapeTransform,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<Vec<DimensionFact>> {
    if request.arguments.len() == 2 {
        let literal = request.literals.literal_args.get(1);
        match literal {
            Some(LiteralValue::Vector(values)) => {
                return Some(
                    values
                        .iter()
                        .map(|value| dimension(value, operation, false, diagnostics))
                        .collect(),
                )
            }
            Some(LiteralValue::Matrix(rows))
                if rows.len() == 1 || rows.iter().all(|row| row.len() == 1) =>
            {
                return Some(
                    rows.iter()
                        .flatten()
                        .map(|value| dimension(value, operation, false, diagnostics))
                        .collect(),
                );
            }
            Some(LiteralValue::Matrix(rows)) if operation == ArrayShapeTransform::Tile => {
                let columns = rows.first().map_or(0, Vec::len);
                if rows.iter().any(|row| row.len() != columns) {
                    return None;
                }
                return Some(
                    (0..columns)
                        .flat_map(|column| rows.iter().map(move |row| &row[column]))
                        .map(|value| dimension(value, operation, false, diagnostics))
                        .collect(),
                );
            }
            Some(LiteralValue::Empty) => return Some(Vec::new()),
            Some(value) if !matches!(value, LiteralValue::Unknown) => {
                return Some(vec![dimension(
                    value,
                    operation,
                    request.arguments[1].storage == StorageFact::Scalar,
                    diagnostics,
                )])
            }
            _ => {}
        }
        return request.arguments[1]
            .shape
            .element_count()
            .map(|count| vec![DimensionFact::Unknown; count]);
    }
    Some(
        request
            .arguments
            .iter()
            .enumerate()
            .skip(1)
            .map(|(index, _)| {
                request
                    .literals
                    .literal_args
                    .get(index)
                    .map_or(DimensionFact::Unknown, |value| {
                        dimension(value, operation, true, diagnostics)
                    })
            })
            .collect(),
    )
}

fn dimension(
    value: &LiteralValue,
    operation: ArrayShapeTransform,
    scalar_form: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> DimensionFact {
    if matches!(value, LiteralValue::Empty) && operation == ArrayShapeTransform::Reshape {
        return DimensionFact::Unknown;
    }
    if let LiteralValue::Integer { .. } = value {
        return match value
            .integer_value_i128()
            .and_then(|value| usize::try_from(value).ok())
        {
            Some(value) => DimensionFact::Known(value),
            None => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SHAPE-DIMENSION",
                    "invalid integer structural control",
                    1,
                ));
                DimensionFact::Unknown
            }
        };
    }
    if let LiteralValue::Bool(value) = value {
        return DimensionFact::Known(usize::from(*value));
    }
    if let Some(value) = LiteralContext::numeric_from_literal(value) {
        let decoded = if operation == ArrayShapeTransform::Tile {
            runmat_types::dimension::decode_replication_factor(value).ok()
        } else if matches!(
            operation,
            ArrayShapeTransform::Permute | ArrayShapeTransform::InversePermute
        ) {
            let rounded = value.round();
            (value.is_finite()
                && (rounded - value).abs() <= f64::EPSILON
                && rounded >= 1.0
                && dimension_fits_platform(rounded))
            .then_some(rounded as usize)
        } else if scalar_form {
            runmat_types::dimension::decode_scalar_dimension(value, true).ok()
        } else {
            decode_shape_dimension(value).ok()
        };
        return match decoded {
            Some(value) => DimensionFact::Known(value),
            None => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SHAPE-DIMENSION",
                    "invalid structural control",
                    1,
                ));
                DimensionFact::Unknown
            }
        };
    }
    DimensionFact::Unknown
}

/// Metadata admission only; structural payload decoding remains runtime-owned.
pub fn shape_transform_control_supported(
    control: &ValueFact,
    operation: ArrayShapeTransform,
    scalar_form: bool,
) -> bool {
    let numeric = matches!(
        control.kind,
        ValueKindFact::Unknown | ValueKindFact::Logical
    ) || control
        .numeric()
        .is_some_and(|numeric| numeric.domain == NumericDomain::Real);
    if !numeric || control.storage == StorageFact::Sparse {
        return false;
    }
    let permutation = matches!(
        operation,
        ArrayShapeTransform::Permute | ArrayShapeTransform::InversePermute
    );
    let device = matches!(control.residency, ResidencyFact::Device { .. });
    if permutation && device {
        return false;
    }
    let count = control.shape.element_count();
    if scalar_form {
        return count.is_none_or(|count| {
            count == 1 || (operation == ArrayShapeTransform::Reshape && count == 0 && !device)
        });
    }
    if count == Some(0) {
        return false;
    }
    // Existing host repmat also accepts matrix controls by flattening in its
    // compatibility adapter. Resident controls and reshape/order require vectors.
    if operation == ArrayShapeTransform::Tile && !device {
        return true;
    }
    control.shape.known_dims().is_none_or(|dims| {
        count == Some(1)
            || dims.len() <= 1
            || ((dims.len() == 2 || !permutation)
                && (dims[0].is_none_or(|n| n == 1) || dims[1].is_none_or(|n| n == 1)))
    })
}
