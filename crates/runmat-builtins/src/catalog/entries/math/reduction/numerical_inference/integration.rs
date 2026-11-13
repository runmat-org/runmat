use super::*;

#[derive(Clone, Copy)]
pub(crate) enum Mode {
    Final,
    Cumulative,
}

pub(crate) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    mode: Mode,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let roles = roles(request);
    let output = if let Some((y_position, x_position, dim_position)) = roles {
        let y = &request.arguments[y_position];
        if !dense_samples(y) && y.kind != ValueKindFact::Unknown {
            diagnostics.push(invalid(
                entry,
                "Y must be dense real or floating complex sampled data",
            ));
        }
        let axis = dim_position.map_or_else(
            || default_axis(&y.shape, true),
            |position| {
                if request.arguments[position].shape.element_count() == Some(0) {
                    default_axis(&y.shape, true)
                } else {
                    literal(request, position).and_then(positive_axis)
                }
            },
        );
        if let Some(position) = dim_position {
            let dimension = &request.arguments[position];
            if dimension
                .shape
                .element_count()
                .is_some_and(|count| count > 1)
                || literal(request, position).is_some_and(|value| {
                    !matches!(value, LiteralValue::Unknown | LiteralValue::Empty)
                }) && axis.is_none()
            {
                diagnostics.push(invalid(
                    entry,
                    "dimension must be a positive integer scalar or an accepted empty placeholder",
                ));
            }
        }
        if let Some(position) = x_position {
            let x = &request.arguments[position];
            if !matches!(
                x.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            ) || x.storage == StorageFact::Sparse
            {
                diagnostics.push(invalid(
                    entry,
                    "X must be dense real sample spacing or coordinates",
                ));
            }
            if let (Some(x_count), Some(y_count), Some(axis)) =
                (x.shape.element_count(), y.shape.element_count(), axis)
            {
                let axis_count = y
                    .shape
                    .known_dims()
                    .and_then(|dimensions| dimensions.get(axis - 1).copied().unwrap_or(Some(1)));
                if x_count > 1
                    && x_count != y_count
                    && axis_count.is_some_and(|count| count != x_count)
                {
                    diagnostics.push(invalid(entry, "X must be scalar, match Y, or supply one coordinate per selected-axis sample"));
                }
            }
        }
        let shape = match mode {
            Mode::Final => reduced_shape(&y.shape, axis),
            Mode::Cumulative => y.shape.clone(),
        };
        let mut output = numeric_output(y, shape);
        if x_position
            .is_some_and(|position| request.arguments[position].residency != ResidencyFact::Host)
        {
            output.residency = ResidencyFact::Unknown;
            output.storage = StorageFact::Unknown;
        }
        output
    } else {
        if request.arguments.is_empty() || request.arguments.len() > 3 {
            diagnostics.push(invalid(entry, "expected Y, Y/dim, X/Y, or X/Y/dim"));
        }
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    finish(request, entry, vec![output], diagnostics)
}

fn roles(request: &CallRequest) -> Option<(usize, Option<usize>, Option<usize>)> {
    match request.arguments.as_slice() {
        [_] => Some((0, None, None)),
        [_, _, _] => Some((1, Some(0), Some(2))),
        [first, second] => {
            // Preserve the existing public two-argument disambiguation: a
            // scalar first argument is X; otherwise a scalar numeric or
            // empty second argument selects Y/dim.
            match first.shape.element_count() {
                Some(1) => Some((1, Some(0), None)),
                Some(_) => match second.shape.element_count() {
                    Some(count)
                        if (count == 0
                            && matches!(
                                second.kind,
                                ValueKindFact::Numeric(_) | ValueKindFact::Logical
                            ))
                            || (count == 1
                                && matches!(
                                    second.kind,
                                    ValueKindFact::Numeric(NumericFact {
                                        domain: NumericDomain::Real,
                                        ..
                                    })
                                )) =>
                    {
                        Some((0, None, Some(1)))
                    }
                    Some(_) => Some((1, Some(0), None)),
                    None => None,
                },
                None => None,
            }
        }
        _ => None,
    }
}
