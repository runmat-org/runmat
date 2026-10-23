use super::*;

fn text(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::String(value)
        | LiteralValue::Character(value)
        | LiteralValue::Keyword(value) => Some(value),
        _ => None,
    }
}

pub(super) fn validate_shapes(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let invalid = |value: &ValueFact| {
        let Some(dims) = value.shape.known_dims() else {
            return false;
        };
        let rows = dims.first().copied().unwrap_or(Some(1));
        let cols = dims.get(1).copied().unwrap_or(Some(1));
        (rows.is_some_and(|n| n != 1) && cols.is_some_and(|n| n != 1))
            || dims
                .iter()
                .skip(2)
                .any(|extent| extent.is_some_and(|n| n != 1))
    };
    let mut check = |value: &ValueFact| {
        if invalid(value) {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.MOVING_WINDOW.INVALID_ARGUMENT",
                "window and SamplePoints controls must be scalars or row/column vectors",
            ));
        }
    };
    if let Some(window) = request.arguments.get(1) {
        check(window);
    }
    for index in 2..request.arguments.len() {
        if request
            .literals
            .literal_args
            .get(index)
            .and_then(text)
            .is_some_and(|option| option.eq_ignore_ascii_case("SamplePoints"))
        {
            if let Some(points) = request.arguments.get(index + 1) {
                check(points);
            }
        }
    }
}

fn count(value: &LiteralValue) -> Option<usize> {
    if let LiteralValue::Integer { .. } = value {
        return usize::try_from(value.integer_value_i128()?).ok();
    }
    let value = crate::ResolveContext::numeric_from_literal(value)?;
    (value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value < usize::MAX as f64)
        .then_some(value as usize)
}

fn scalar(value: &LiteralValue) -> &LiteralValue {
    match value {
        LiteralValue::Vector(values) if values.len() == 1 => &values[0],
        LiteralValue::Matrix(rows) if rows.len() == 1 && rows[0].len() == 1 => &rows[0][0],
        other => other,
    }
}

fn count_trim(window: &LiteralValue) -> Option<usize> {
    let pair = match window {
        LiteralValue::Vector(values) if values.len() == 2 => Some((&values[0], &values[1])),
        LiteralValue::Matrix(rows) if rows.len() == 1 && rows[0].len() == 2 => {
            Some((&rows[0][0], &rows[0][1]))
        }
        LiteralValue::Matrix(rows) if rows.len() == 2 && rows.iter().all(|row| row.len() == 1) => {
            Some((&rows[0][0], &rows[1][0]))
        }
        _ => None,
    };
    if let Some((before, after)) = pair {
        count(before)?
            .checked_add(count(after)?)?
            .checked_add(1)?
            .checked_sub(1)
    } else {
        count(scalar(window))?.checked_sub(1)
    }
}

pub(super) fn shape(
    source: &ShapeFact,
    request: &CallRequest,
    operation: MovingOperation,
) -> Option<ShapeFact> {
    let literals = &request.literals.literal_args;
    let window = literals.get(1)?;
    let mut axis = None;
    let mut normalization = !operation.dispersion();
    let mut discard = false;
    let mut non_shrink = false;
    let mut samples = false;
    let mut index = 2;
    while index < request.arguments.len() {
        let value = literals.get(index)?;
        if let Some(option) = text(value) {
            if ["omitnan", "omitmissing", "includenan", "includemissing"]
                .iter()
                .any(|known| option.eq_ignore_ascii_case(known))
            {
                index += 1;
                continue;
            }
            if option.eq_ignore_ascii_case("Endpoints") {
                let endpoint = literals.get(index + 1)?;
                non_shrink =
                    !text(endpoint).is_some_and(|text| text.eq_ignore_ascii_case("shrink"));
                discard = if let Some(endpoint) = text(endpoint) {
                    if endpoint.eq_ignore_ascii_case("discard") {
                        true
                    } else if endpoint.eq_ignore_ascii_case("shrink")
                        || endpoint.eq_ignore_ascii_case("fill")
                    {
                        false
                    } else {
                        return None;
                    }
                } else {
                    crate::ResolveContext::numeric_from_literal(scalar(endpoint))?;
                    false
                };
                index += 2;
                continue;
            }
            if option.eq_ignore_ascii_case("SamplePoints") {
                if samples {
                    return None;
                }
                literals.get(index + 1)?;
                samples = true;
                index += 2;
                continue;
            }
            return None;
        }
        if !normalization {
            let value = scalar(value);
            if matches!(value, LiteralValue::Empty)
                || matches!(value, LiteralValue::Vector(values) if values.is_empty())
                || matches!(count(value), Some(0 | 1))
            {
                normalization = true;
                index += 1;
                continue;
            }
            return None;
        }
        if axis.is_some() {
            return None;
        }
        axis = Some(count(scalar(value))?.checked_sub(1)?);
        index += 1;
    }
    if samples && non_shrink {
        return None;
    }
    let trim = if samples { 0 } else { count_trim(window)? };
    let mut dims = source.known_dims()?;
    let axis = if let Some(axis) = axis {
        axis
    } else {
        // An unknown earlier extent can change the default first nonsingleton.
        let mut selected = None;
        for (index, dim) in dims.iter().enumerate() {
            match dim {
                Some(1) => {}
                Some(_) => {
                    selected = Some(index);
                    break;
                }
                None => return None,
            }
        }
        selected.unwrap_or(0)
    };
    let rank = dims.len().max(axis.checked_add(1)?);
    if rank > dims.len() {
        // A huge literal dimension must not allocate its runtime shape in the
        // compiler; preserve the known rank without materializing all extents.
        return Some(ShapeFact::Ranked { rank });
    }
    if discard {
        dims[axis] = dims[axis].map(|extent| extent.saturating_sub(trim));
    }
    Some(ShapeFact::from(dims))
}
