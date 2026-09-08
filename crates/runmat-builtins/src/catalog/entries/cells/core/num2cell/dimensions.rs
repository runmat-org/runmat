use runmat_types::{CallRequest, DimensionFact, LiteralValue, ShapeFact};

pub(super) fn grouped(request: &CallRequest) -> Option<Vec<usize>> {
    if request.arguments.len() == 1 {
        return Some(Vec::new());
    }
    if let Some(values) = request.literals.numeric_vector_at(1) {
        return values.into_iter().collect();
    }
    request
        .literals
        .numeric_at(1)
        .and_then(positive_integer)
        .map(|dim| vec![dim])
}

pub(super) fn shape_may_be_vector(shape: &ShapeFact) -> bool {
    match shape {
        ShapeFact::Scalar | ShapeFact::Ranked { .. } | ShapeFact::Unknown => true,
        ShapeFact::Shaped { dims } => {
            dims.iter()
                .filter(|dim| matches!(dim, DimensionFact::Known(extent) if *extent > 1))
                .count()
                <= 1
        }
    }
}

pub(super) fn literal_values(literal: &LiteralValue) -> Result<Option<Vec<usize>>, ()> {
    match literal {
        LiteralValue::Unknown => Ok(None),
        LiteralValue::Empty => Ok(Some(Vec::new())),
        LiteralValue::Vector(values) => values
            .iter()
            .map(literal_dimension)
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        LiteralValue::Number(value) => positive_integer(*value)
            .map(|value| Some(vec![value]))
            .ok_or(()),
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .ok()
            .and_then(positive_integer)
            .map(|value| Some(vec![value]))
            .ok_or(()),
        LiteralValue::Integer { text, .. } => text
            .parse::<usize>()
            .ok()
            .filter(|value| *value >= 1)
            .map(|value| Some(vec![value]))
            .ok_or(()),
        LiteralValue::Matrix(_)
        | LiteralValue::Complex { .. }
        | LiteralValue::Bool(_)
        | LiteralValue::Character(_)
        | LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_) => Err(()),
    }
}

fn positive_integer(value: f64) -> Option<usize> {
    (value.is_finite() && value >= 1.0 && value.fract() == 0.0 && value < usize::MAX as f64)
        .then_some(value as usize)
}

fn literal_dimension(literal: &LiteralValue) -> Result<usize, ()> {
    literal_values(literal)?
        .and_then(|values| (values.len() == 1).then(|| values[0]))
        .ok_or(())
}
